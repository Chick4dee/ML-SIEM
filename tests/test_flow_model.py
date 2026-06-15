"""Тесты Flow CNN+LSTM: формы, эмбеддинги, инвариантность к паддингу прогрева."""

import torch

from mlsiem.models.flow_cnn_lstm.model import FlowCNNLSTM, FlowModelConfig


def _cfg(n_numeric=4, n_classes=3):
    # последняя позиция в векторе фич — категориальная (как в нашем пайплайне)
    return FlowModelConfig(
        n_numeric=n_numeric,
        n_classes=n_classes,
        cat_cardinalities=[10],
        cat_emb_dims=[4],
        cat_positions=[n_numeric],   # числовые 0..n-1, категория на позиции n
        conv_channels=16,
        lstm_hidden=32,
    )


def _model():
    torch.manual_seed(0)
    m = FlowCNNLSTM(_cfg())
    m.eval()
    return m


def test_forward_shape():
    m = _model()
    b, n, f = 8, 16, 5  # 4 числовых + 1 категориальный
    x = torch.randn(b, n, f)
    x[:, :, 4] = torch.randint(0, 10, (b, n)).float()
    mask = torch.ones(b, n, dtype=torch.bool)
    out = m(x, mask)
    assert out.shape == (b, 3)


def test_padding_invariance():
    """Окно с ведущим паддингом должно дать тот же выход, что те же потоки
    без паддинга — маска прогрева обязана полностью игнорировать паддинг."""
    m = _model()
    torch.manual_seed(1)
    # три реальных потока (фичи 4 числовых + 1 категория)
    real = torch.randn(1, 3, 5)
    real[:, :, 4] = torch.tensor([[1.0, 2.0, 3.0]])

    # вариант A: окно N=4 с одним паддингом слева
    x4 = torch.zeros(1, 4, 5)
    x4[:, 1:, :] = real
    mask4 = torch.tensor([[False, True, True, True]])

    # вариант B: окно N=3 без паддинга
    mask3 = torch.tensor([[True, True, True]])

    out_a = m(x4, mask4)
    out_b = m(real, mask3)
    torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-4)


def test_oov_index_zero_is_valid_embedding():
    m = _model()
    x = torch.randn(2, 5, 5)
    x[:, :, 4] = 0.0  # все категории = OOV (0)
    mask = torch.ones(2, 5, dtype=torch.bool)
    out = m(x, mask)
    assert out.shape == (2, 3)
    assert not torch.isnan(out).any()


def test_backward_runs():
    torch.manual_seed(0)
    m = FlowCNNLSTM(_cfg())
    m.train()
    x = torch.randn(4, 8, 5)
    x[:, :, 4] = torch.randint(0, 10, (4, 8)).float()
    mask = torch.ones(4, 8, dtype=torch.bool)
    mask[:, 0] = False  # немного прогрева
    out = m(x, mask)
    loss = torch.nn.functional.cross_entropy(out, torch.tensor([0, 1, 2, 0]))
    loss.backward()
    grads = [p.grad is not None for p in m.parameters()]
    assert all(grads)
