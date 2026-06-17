"""X23D-like sparse multi-view reconstruction baseline."""

from __future__ import annotations

import torch
from torch import nn

from ..core.config import ModelConfig
from .layers import BackProjector, TinyEncoder2d, TinyRefiner3d


class SparseXrayTo3D(nn.Module):
    """Minimal geometry-aware baseline."""

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.encoder = TinyEncoder2d(config.image.channels, config.base_channels, config.feature_channels)
        volume_shape = (config.volume.depth, config.volume.height, config.volume.width)
        self.backproject = BackProjector(volume_shape)
        self.refiner = TinyRefiner3d(config.feature_channels, config.feature_channels)
        self.volume_prior = nn.Parameter(torch.zeros(1, 1, *volume_shape))

    def forward(self, images: torch.Tensor, projections: torch.Tensor) -> torch.Tensor:
        """Return occupancy probabilities with shape [B, 1, D, H, W]."""

        batch, views, channels, height, width = images.shape
        features = self.encoder(images.reshape(batch * views, channels, height, width))
        _, feature_channels, feature_h, feature_w = features.shape
        features = features.reshape(batch, views, feature_channels, feature_h, feature_w)
        fused = self.backproject(features, projections).mean(dim=1)
        return torch.sigmoid(self.refiner(fused) + self.volume_prior)


def count_parameters(model: nn.Module) -> int:
    """Return trainable parameter count."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
