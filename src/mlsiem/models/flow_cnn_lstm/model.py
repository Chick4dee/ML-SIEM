"""Эксперт 1 — Flow CNN+LSTM (PLAN.md, раздел 4).

Поток фич окна: [B, N, F]. По каждому потоку (позиции окна) Conv1D ловит
локальные паттерны между признаками; LSTM обходит окно во времени, учитывая
маску прогрева (паддинг-позиции упаковываются прочь, не влияя на скрытое
состояние); голова даёт мультикласс.

Категориальные признаки (application_name/category) приходят как целочисленные
индексы (0 = OOV) и встраиваются эмбеддингами, затем конкатенируются с
числовыми — обучаемое представление сервиса вместо one-hot.

Под 8 ГБ VRAM: компактный, совместим с AMP (PLAN.md, раздел 10.6).
"""

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class FlowModelConfig:
    n_numeric: int
    n_classes: int
    # (размер_словаря, размерность_эмбеддинга) на каждый категориальный признак;
    # позиции категориальных в общем векторе фич — в cat_positions
    cat_cardinalities: list[int] = field(default_factory=list)
    cat_emb_dims: list[int] = field(default_factory=list)
    cat_positions: list[int] = field(default_factory=list)
    conv_channels: int = 64
    conv_kernel: int = 3
    lstm_hidden: int = 128
    lstm_layers: int = 1
    dropout: float = 0.2


class FlowCNNLSTM(nn.Module):
    def __init__(self, cfg: FlowModelConfig):
        super().__init__()
        self.cfg = cfg
        self.cat_positions = list(cfg.cat_positions)
        self.numeric_positions = [
            i for i in range(cfg.n_numeric + len(self.cat_positions))
            if i not in set(self.cat_positions)
        ]

        self.embeddings = nn.ModuleList([
            nn.Embedding(card, dim)
            for card, dim in zip(cfg.cat_cardinalities, cfg.cat_emb_dims, strict=True)
        ])
        per_step_in = len(self.numeric_positions) + sum(cfg.cat_emb_dims)

        # Conv1D по оси признаков внутри одного потока (TimeDistributed).
        self.conv = nn.Sequential(
            nn.Conv1d(1, cfg.conv_channels, cfg.conv_kernel, padding=cfg.conv_kernel // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # → [.., conv_channels, 1]
        )
        self.conv_in = per_step_in

        self.lstm = nn.LSTM(
            input_size=cfg.conv_channels,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.lstm_hidden, cfg.n_classes),
        )

    def _embed_step(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, F_raw] → [B, N, per_step_in] (числовые + эмбеддинги категорий)."""
        parts = [x[:, :, self.numeric_positions]]
        for emb, pos in zip(self.embeddings, self.cat_positions, strict=True):
            idx = x[:, :, pos].long().clamp(min=0)
            parts.append(emb(idx))
        return torch.cat(parts, dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: [B, N, F_raw], mask: [B, N] bool (True=реальный поток). → [B, n_classes]."""
        b, n, _ = x.shape
        step = self._embed_step(x)                       # [B, N, per_step_in]

        conv_in = step.reshape(b * n, 1, step.shape[-1])  # каждый поток отдельно
        conv_out = self.conv(conv_in).reshape(b, n, self.cfg.conv_channels)

        # Окна паддятся СЛЕВА (реальные потоки в конце), а pack_padded_sequence
        # ждёт паддинг справа. Сдвигаем реальные потоки в начало, сохраняя их
        # хронологический порядок, затем упаковываем по реальной длине —
        # паддинг не влияет на скрытое состояние LSTM.
        lengths = mask.sum(dim=1).clamp(min=1)
        left_pad = n - lengths                            # сколько паддинга слева
        roll_idx = (torch.arange(n, device=x.device) + left_pad[:, None]) % n
        rolled = conv_out.gather(
            1, roll_idx[:, :, None].expand(-1, -1, self.cfg.conv_channels)
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            rolled, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        return self.head(h_n[-1])                         # последний слой LSTM
