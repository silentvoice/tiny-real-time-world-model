from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch

from world_model_lab.model import TinyDenoiser, dummy_input, parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained denoiser checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("public/model/tiny_denoiser.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    width = int(checkpoint.get("width", 96))
    blocks = int(checkpoint.get("blocks", 14))

    model = TinyDenoiser(width=width, blocks=blocks)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"parameters={parameter_count(model):,}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input("cpu"),
        args.output,
        input_names=["input"],
        output_names=["pred"],
        opset_version=args.opset,
        do_constant_folding=True,
    )

    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print(f"exported {args.output}")


if __name__ == "__main__":
    main()
