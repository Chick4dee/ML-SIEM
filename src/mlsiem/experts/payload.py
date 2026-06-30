"""Эксперт 3 — char-CNN по содержимому HTTP-запроса (PLAN.md, раздел 4).

Закрывает зону, слепую для flow/AE: web-атаки (SQLi/XSS/brute), которые на
flow-фичах неотличимы от нормы — сигнал в СИМВОЛАХ запроса (`'`, `;`, `DROP`,
`<script>`, `../`). Символьная свёртка ловит эти локальные паттерны независимо
от их позиции.

Токенизация — посимвольная (char-level): не зависит от словаря слов, устойчива к
обфускации и percent-encoding. 0 = PAD, 1 = UNK (OOV-символ на serve).
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

PAD_IDX = 0
UNK_IDX = 1


class CharTokenizer:
    """Символ ↔ индекс; кодирование текста в фикс. длину (truncate/pad)."""

    def __init__(self, chars: list[str], max_len: int):
        self.max_len = max_len
        self.chars = list(chars)
        self.stoi = {c: i + 2 for i, c in enumerate(self.chars)}  # 0 PAD, 1 UNK

    @classmethod
    def fit(cls, texts, max_len: int, max_vocab: int = 200) -> "CharTokenizer":
        cnt = Counter(ch for t in texts for ch in t)
        chars = sorted(c for c, _ in cnt.most_common(max_vocab))
        return cls(chars, max_len)

    @property
    def vocab_size(self) -> int:
        return len(self.chars) + 2

    def encode(self, text: str) -> np.ndarray:
        idx = np.zeros(self.max_len, dtype=np.int64)  # PAD=0
        for i, ch in enumerate(text[: self.max_len]):
            idx[i] = self.stoi.get(ch, UNK_IDX)
        return idx

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"chars": self.chars, "max_len": self.max_len},
                                   ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["chars"], d["max_len"])


@dataclass
class CharCNNConfig:
    vocab_size: int
    n_classes: int
    emb_dim: int = 32
    channels: int = 128
    kernels: tuple[int, ...] = field(default_factory=lambda: (3, 5, 7))
    dropout: float = 0.3


class CharCNN(nn.Module):
    def __init__(self, cfg: CharCNNConfig):
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=PAD_IDX)
        self.convs = nn.ModuleList([
            nn.Conv1d(cfg.emb_dim, cfg.channels, k, padding=k // 2) for k in cfg.kernels
        ])
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.channels * len(cfg.kernels), cfg.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L] индексы символов → [B, n_classes]."""
        emb = self.embedding(x).transpose(1, 2)          # [B, emb, L]
        # свёртки разных ядер + global max-pool (позиция паттерна не важна)
        pooled = [torch.relu(conv(emb)).amax(dim=2) for conv in self.convs]
        return self.head(torch.cat(pooled, dim=1))
