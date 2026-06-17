"""Configuration objects for the Spine2Space prototype."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageConfig:
    """2D view preprocessing defaults."""

    height: int = 224
    width: int = 224
    channels: int = 1


@dataclass(frozen=True)
class VolumeConfig:
    """3D occupancy defaults."""

    depth: int = 64
    height: int = 64
    width: int = 64


@dataclass(frozen=True)
class ModelConfig:
    """Small sparse multi-view baseline configuration."""

    views: int = 4
    image: ImageConfig = ImageConfig()
    volume: VolumeConfig = VolumeConfig()
    base_channels: int = 8
    feature_channels: int = 16
