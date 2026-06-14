from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from world_model_lab.env import FRAME_HEIGHT, FRAME_WIDTH, BreakoutEnv, make_context
from world_model_lab.model import TinyDenoiser, make_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a short real-vs-neural rollout GIF.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/rollout.gif"))
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--denoise-steps", type=int, default=4)
    parser.add_argument("--sigma-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    chw = np.transpose(frame, (2, 0, 1))[None]
    return torch.from_numpy(chw).to(device=device, dtype=torch.float32) / 127.5 - 1


def context_tensor(context: list[np.ndarray], device: torch.device) -> torch.Tensor:
    chw = np.concatenate([np.transpose(frame, (2, 0, 1)) for frame in context], axis=0)[None]
    return torch.from_numpy(chw).to(device=device, dtype=torch.float32) / 127.5 - 1


def to_uint8(tensor: torch.Tensor) -> np.ndarray:
    image = ((tensor[0].detach().cpu().clamp(-1, 1) + 1) * 127.5).byte().numpy()
    return np.transpose(image, (1, 2, 0))


@torch.no_grad()
def sample_next(
    model: TinyDenoiser,
    context: list[np.ndarray],
    action: int,
    denoise_steps: int,
    sigma_max: float,
    device: torch.device,
) -> np.ndarray:
    current = to_tensor(context[-1], device) + torch.randn(
        (1, 3, FRAME_HEIGHT, FRAME_WIDTH), device=device
    ) * sigma_max
    sigmas = np.geomspace(sigma_max, 0.02, max(1, denoise_steps)).tolist() + [0.0]
    ctx = context_tensor(context, device)
    actions = torch.tensor([action], device=device, dtype=torch.long)
    for idx, sigma_value in enumerate(sigmas[:-1]):
        sigma = torch.tensor([sigma_value], device=device, dtype=torch.float32)
        pred = model(make_input(ctx, current, actions, sigma))
        next_sigma = sigmas[idx + 1]
        if next_sigma == 0:
            current = pred
        else:
            current = pred + (next_sigma / max(0.02, sigma_value)) * (current - pred)
    return to_uint8(current)


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TinyDenoiser(width=int(checkpoint["width"]), blocks=int(checkpoint["blocks"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    real_env = BreakoutEnv(args.seed)
    neural_env = BreakoutEnv(args.seed)
    real_context = list(make_context(real_env))
    neural_context = list(make_context(neural_env))

    frames = []
    for _ in range(args.steps):
        action = real_env.scripted_action(random_rate=0.08)
        real_env.step(action)
        real_frame = real_env.render_rgb()
        real_context = (real_context + [real_frame])[-4:]

        neural_frame = sample_next(
            model, neural_context, action, args.denoise_steps, args.sigma_max, device
        )
        neural_context = (neural_context + [neural_frame])[-4:]

        separator = np.zeros((FRAME_HEIGHT, 2, 3), dtype=np.uint8)
        frame = np.concatenate([real_frame, separator, neural_frame], axis=1)
        frames.append(np.repeat(np.repeat(frame, 4, axis=0), 4, axis=1))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, fps=20)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
