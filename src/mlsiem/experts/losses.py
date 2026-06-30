"""Loss-функции для несбалансированной мультиклассовой детекции.

Дисбаланс UNSW экстремальный (benign 96.8%, worms 0.01%), поэтому plain
cross-entropy выучивает «всё benign». Focal loss (Lin et al., 2017) давит вклад
лёгких примеров (доминирующий benign) и поднимает редкие классы; опциональные
per-class веса (alpha) — дополнительный рычаг (PLAN.md, раздел 3).
"""

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)  # вероятность истинного класса
        return ((1 - pt) ** self.gamma * ce).mean()


def inverse_frequency_alpha(class_counts: list[int]) -> torch.Tensor:
    """Веса классов ∝ 1/частота, нормированные к среднему 1 (для alpha в focal)."""
    counts = torch.tensor(class_counts, dtype=torch.float64).clamp(min=1)
    inv = counts.sum() / counts
    return (inv / inv.mean()).float()
