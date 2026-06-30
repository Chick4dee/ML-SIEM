"""Тесты Conv1D автоэнкодера: формы, узкое горлышко, маскированная ошибка."""

import torch

from mlsiem.experts.autoencoder import (
    AEConfig,
    Conv1DAutoencoder,
    reconstruction_error,
)


def _model(n_features=8, window=16):
    torch.manual_seed(0)
    return Conv1DAutoencoder(AEConfig(n_features=n_features, window=window)), n_features, window


def test_forward_shape_preserved():
    m, f, n = _model()
    x = torch.randn(4, n, f)
    recon = m(x)
    assert recon.shape == x.shape


def test_bottleneck_is_narrow():
    m, f, n = _model(n_features=62, window=16)
    x = torch.randn(2, n, 62)
    z = m.latent(x)
    # горлышко должно быть СИЛЬНО уже входа (урок v1: широкое реконструирует атаки)
    assert z.shape == (2, 4, 16)            # bottleneck_ch=4
    assert z.numel() // 2 < (n * 62) // 4   # latent ≪ размер входа


def test_reconstruction_error_ignores_padding():
    m, f, n = _model()
    x = torch.randn(2, n, f)
    recon = x.clone()
    # внесём большую ошибку ТОЛЬКО в паддинг-позиции (первые 4) — маска их скрывает
    recon[:, :4, :] += 100.0
    mask = torch.ones(2, n, dtype=torch.bool)
    mask[:, :4] = False
    err = reconstruction_error(recon, x, mask)
    assert torch.allclose(err, torch.zeros(2), atol=1e-5)


def test_reconstruction_error_positive_on_real_mismatch():
    m, f, n = _model()
    x = torch.randn(2, n, f)
    recon = x + 1.0
    mask = torch.ones(2, n, dtype=torch.bool)
    err = reconstruction_error(recon, x, mask)
    assert torch.all(err > 0)


def test_backward_runs():
    m, f, n = _model()
    x = torch.randn(4, n, f)
    mask = torch.ones(4, n, dtype=torch.bool)
    err = reconstruction_error(m(x), x, mask).mean()
    err.backward()
    assert all(p.grad is not None for p in m.parameters())
