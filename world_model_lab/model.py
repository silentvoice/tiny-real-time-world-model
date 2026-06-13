from __future__ import annotations

import torch
from torch import nn

from world_model_lab.env import FRAME_HEIGHT, FRAME_WIDTH

INPUT_CHANNELS = 3 + 4 * 3 + 3 + 1


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=padding, dilation=dilation),
            nn.SiLU(),
            nn.Conv2d(width, width, kernel_size=3, padding=padding, dilation=dilation),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class TinyDenoiser(nn.Module):
    """Browser-friendly denoiser using only common ONNX ops."""

    def __init__(self, width: int = 96, blocks: int = 14):
        super().__init__()
        dilations = [1, 2, 4, 8]
        self.in_proj = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, width, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(width, dilations[i % len(dilations)]) for i in range(blocks)]
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, 3, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.blocks(x)
        return self.out_proj(x)


def make_input(
    context: torch.Tensor,
    noisy_next: torch.Tensor,
    actions: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Build NCHW model input from normalized context/target tensors."""

    batch = context.shape[0]
    height = context.shape[2]
    width = context.shape[3]
    action_planes = torch.zeros(
        (batch, 3, height, width), dtype=context.dtype, device=context.device
    )
    action_planes.scatter_(
        1,
        actions.view(batch, 1, 1, 1).expand(batch, 1, height, width),
        1.0,
    )
    sigma_plane = sigma.view(batch, 1, 1, 1).expand(batch, 1, height, width)
    return torch.cat([noisy_next, context, action_planes, sigma_plane], dim=1)


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def dummy_input(device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.zeros((1, INPUT_CHANNELS, FRAME_HEIGHT, FRAME_WIDTH), device=device)

