"""Тесты focal loss и весов классов."""

import torch

from mlsiem.models.losses import FocalLoss, inverse_frequency_alpha


def test_focal_reduces_to_ce_when_gamma_zero():
    logits = torch.randn(16, 5)
    target = torch.randint(0, 5, (16,))
    focal = FocalLoss(gamma=0.0)(logits, target)
    ce = torch.nn.functional.cross_entropy(logits, target)
    torch.testing.assert_close(focal, ce, atol=1e-6, rtol=1e-5)


def test_focal_downweights_easy_examples():
    # уверенно-правильный (лёгкий) пример должен давать меньший focal-вклад, чем CE
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    target = torch.tensor([0])
    focal = FocalLoss(gamma=2.0)(logits, target)
    ce = torch.nn.functional.cross_entropy(logits, target)
    assert focal < ce


def test_inverse_frequency_alpha_favors_rare():
    alpha = inverse_frequency_alpha([1000, 10])
    assert alpha[1] > alpha[0]            # редкий класс — больший вес
    assert abs(float(alpha.mean()) - 1.0) < 1e-5  # нормировка к среднему 1


def test_alpha_weighted_focal_runs():
    alpha = inverse_frequency_alpha([900, 90, 10])
    loss = FocalLoss(gamma=2.0, alpha=alpha)
    out = loss(torch.randn(8, 3), torch.randint(0, 3, (8,)))
    assert out.ndim == 0 and out.item() >= 0
