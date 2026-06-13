from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.cuda.amp import GradScaler
from tqdm import trange

from world_model_lab.env import (
    CONTEXT_FRAMES,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    BreakoutEnv,
    frames_to_context,
    make_context,
)
from world_model_lab.model import TinyDenoiser, make_input, parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny conditional diffusion world model.")
    parser.add_argument("--steps", type=int, default=40_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=14)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--sigma-min", type=float, default=0.02)
    parser.add_argument("--sigma-max", type=float, default=1.0)
    parser.add_argument("--random-action-rate", type=float, default=0.22)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/tiny-denoiser"))
    parser.add_argument("--save-every", type=int, default=2_000)
    parser.add_argument("--sample-every", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class BatchWorlds:
    def __init__(self, batch_size: int, seed: int, random_action_rate: float):
        self.random_action_rate = random_action_rate
        self.envs = [BreakoutEnv(seed + idx * 9973) for idx in range(batch_size)]
        self.contexts: list[deque[np.ndarray]] = [make_context(env) for env in self.envs]

    def next_batch(self):
        contexts = np.empty(
            (len(self.envs), CONTEXT_FRAMES * 3, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8
        )
        targets = np.empty((len(self.envs), 3, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        actions = np.empty((len(self.envs),), dtype=np.int64)

        for idx, env in enumerate(self.envs):
            action = env.scripted_action(self.random_action_rate)
            contexts[idx] = frames_to_context(self.contexts[idx])
            env.step(action)
            frame = env.render_rgb()
            targets[idx] = np.transpose(frame, (2, 0, 1))
            self.contexts[idx].append(frame)
            actions[idx] = action

        return contexts, actions, targets


def normalize_uint8(array: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(array).to(device=device, dtype=torch.float32, non_blocking=True)
    return tensor / 127.5 - 1.0


def save_checkpoint(
    path: Path,
    model: TinyDenoiser,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "width": args.width,
            "blocks": args.blocks,
            "args": vars(args),
        },
        path,
    )


@torch.no_grad()
def save_sample_grid(
    path: Path,
    model: TinyDenoiser,
    batch: tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
    sigma: float,
):
    model.eval()
    context_np, actions_np, target_np = batch
    context = normalize_uint8(context_np[:8], device)
    target = normalize_uint8(target_np[:8], device)
    actions = torch.from_numpy(actions_np[:8]).to(device=device)
    sigma_tensor = torch.full((target.shape[0],), sigma, device=device)
    noisy = target + sigma_tensor.view(-1, 1, 1, 1) * torch.randn_like(target)
    pred = model(make_input(context, noisy, actions, sigma_tensor)).clamp(-1, 1)

    target_img = ((target.cpu() + 1) * 127.5).byte().numpy()
    pred_img = ((pred.cpu() + 1) * 127.5).byte().numpy()
    rows = []
    for idx in range(target_img.shape[0]):
        left = np.transpose(target_img[idx], (1, 2, 0))
        right = np.transpose(pred_img[idx], (1, 2, 0))
        rows.append(np.concatenate([left, right], axis=1))
    grid = np.concatenate(rows, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(path)
    model.train()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_dir: Path = args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_dir / "train_args.json").open("w", encoding="utf-8") as handle:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, handle, indent=2)

    device = torch.device(args.device)
    model = TinyDenoiser(width=args.width, blocks=args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=device.type == "cuda")
    worlds = BatchWorlds(args.batch_size, args.seed, args.random_action_rate)

    print(f"device={device}")
    print(f"parameters={parameter_count(model):,}")
    print(f"checkpoint_dir={checkpoint_dir}")

    ema_loss = None
    started_at = time.time()
    progress = trange(1, args.steps + 1, dynamic_ncols=True)
    for step in progress:
        context_np, actions_np, target_np = worlds.next_batch()
        context = normalize_uint8(context_np, device)
        target = normalize_uint8(target_np, device)
        actions = torch.from_numpy(actions_np).to(device=device)

        log_sigma = torch.empty((args.batch_size,), device=device).uniform_(
            math.log(args.sigma_min), math.log(args.sigma_max)
        )
        sigma = log_sigma.exp()
        noisy = target + sigma.view(-1, 1, 1, 1) * torch.randn_like(target)
        model_input = make_input(context, noisy, actions, sigma)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            pred = model(model_input)
            loss = torch.mean((pred - target) ** 2)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_value = float(loss.detach().cpu())
        ema_loss = loss_value if ema_loss is None else 0.97 * ema_loss + 0.03 * loss_value
        progress.set_postfix(loss=f"{ema_loss:.5f}", elapsed=f"{time.time() - started_at:.0f}s")

        if step % args.sample_every == 0 or step == 1:
            save_sample_grid(
                checkpoint_dir / "samples" / f"step-{step:06d}.png",
                model,
                (context_np, actions_np, target_np),
                device,
                sigma=min(args.sigma_max, 0.75),
            )

        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, step, args)
            save_checkpoint(checkpoint_dir / f"step-{step:06d}.pt", model, optimizer, step, args)

    save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, args.steps, args)
    print(f"done: {checkpoint_dir / 'last.pt'}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()

