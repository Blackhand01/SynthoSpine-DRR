"""PyTorch layers for the X23D-like baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyEncoder2d(nn.Module):
    """Small 2D encoder that maps 224x224 views to 112x112 feature maps."""

    def __init__(self, in_channels: int, base_channels: int, out_channels: int) -> None:
        super().__init__()
        self.stem = ConvBlock2d(in_channels, base_channels)
        self.down = nn.Sequential(nn.MaxPool2d(2), ConvBlock2d(base_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.stem(x))


class TinyRefiner3d(nn.Module):
    """Small 3D refiner producing one occupancy channel."""

    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(ConvBlock3d(in_channels, hidden_channels), nn.Conv3d(hidden_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BackProjector(nn.Module):
    """Project 2D feature maps into a 3D grid using projection matrices."""

    def __init__(self, volume_shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.volume_shape = volume_shape

    def forward(self, features: torch.Tensor, projections: torch.Tensor) -> torch.Tensor:
        batch, views, channels, height, width = features.shape
        points = self._world_grid(features.device, features.dtype)
        sampled = []
        for batch_idx in range(batch):
            view_features = []
            for view_idx in range(views):
                grid = self._sample_grid(points, projections[batch_idx, view_idx], height, width)
                sample = F.grid_sample(features[batch_idx, view_idx].unsqueeze(0), grid, align_corners=True)
                view_features.append(sample.reshape(channels, *self.volume_shape))
            sampled.append(torch.stack(view_features, dim=0))
        return torch.stack(sampled, dim=0)

    def _world_grid(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        depth, height, width = self.volume_shape
        z = torch.linspace(1.0, 2.0, depth, device=device, dtype=dtype)
        y = torch.linspace(-0.5, 0.5, height, device=device, dtype=dtype)
        x = torch.linspace(-0.5, 0.5, width, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack([xx, yy, zz, torch.ones_like(xx)], dim=-1).reshape(-1, 4)

    def _sample_grid(self, points: torch.Tensor, projection: torch.Tensor, h: int, w: int) -> torch.Tensor:
        projected = points @ projection.T
        xy = projected[:, :2] / projected[:, 2:3].clamp_min(1e-6)
        x_norm = (xy[:, 0] / max(w - 1, 1)) * 2.0 - 1.0
        y_norm = (xy[:, 1] / max(h - 1, 1)) * 2.0 - 1.0
        depth, height, width = self.volume_shape
        return torch.stack([x_norm, y_norm], dim=-1).reshape(1, depth * height, width, 2)
