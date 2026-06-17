"""Training losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def dice_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss for occupancy probabilities."""

    pred = prediction.reshape(prediction.shape[0], -1)
    truth = target.reshape(target.shape[0], -1)
    intersection = (pred * truth).sum(dim=1)
    score = (2.0 * intersection + eps) / (pred.sum(dim=1) + truth.sum(dim=1) + eps)
    return 1.0 - score.mean()


def bce_dice_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE plus soft Dice loss."""

    return F.binary_cross_entropy(prediction, target) + dice_loss(prediction, target)


def balanced_bce_dice_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Class-balanced BCE plus Dice for sparse occupancy targets."""

    positive = target.sum().clamp_min(1.0)
    negative = (target.numel() - positive).clamp_min(1.0)
    pos_weight = (negative / positive).clamp(max=100.0)
    bce = F.binary_cross_entropy(prediction, target, weight=torch.where(target > 0, pos_weight, 1.0))
    return bce + dice_loss(prediction, target)
