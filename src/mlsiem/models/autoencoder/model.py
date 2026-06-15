"""Эксперт 4 — Conv1D автоэнкодер аномалий (PLAN.md, разделы 4, 10.6).

Учится реконструировать benign-окна; высокая ошибка реконструкции ⇒ аномалия
(волюметрические/тайминговые атаки, zero-day). Роль — open-set страховка, НЕ
универсальный классификатор: скрытные атаки (brute/web/bot) на flow неотличимы
от нормы, их закрывают другие эксперты.

Рецепт из v1 (что работает / что НЕ повторять):
- Conv1D по оси времени окна; НЕ Dense/Flatten (убивает время), НЕ GlobalAvgPool.
- Достаточно УЗКОЕ горлышко — широкое реконструирует и атаки тоже.
- Маскированная ошибка: позиции прогрева (паддинг) не штрафуем.
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AEConfig:
    n_features: int
    window: int = 16
    # каналы энкодера до горлышка; горлышко = bottleneck_ch × window признаков
    enc_channels: tuple[int, ...] = (32, 16)
    bottleneck_ch: int = 4   # узкое: 4 × 16 = 64 (latent ~64, урок v1)

    @property
    def latent_size(self) -> int:
        return self.bottleneck_ch * self.window


class Conv1DAutoencoder(nn.Module):
    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg
        chans = [cfg.n_features, *cfg.enc_channels, cfg.bottleneck_ch]

        enc = []
        for i in range(len(chans) - 1):
            enc.append(nn.Conv1d(chans[i], chans[i + 1], kernel_size=3, padding=1))
            if i < len(chans) - 2:
                enc.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc)

        dec_chans = list(reversed(chans))
        dec = []
        for i in range(len(dec_chans) - 1):
            dec.append(nn.Conv1d(dec_chans[i], dec_chans[i + 1], kernel_size=3, padding=1))
            if i < len(dec_chans) - 2:
                dec.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, F] → reconstruction [B, N, F] (время сохраняется, без Flatten)."""
        z = self.encoder(x.transpose(1, 2))   # [B, bottleneck_ch, N]
        recon = self.decoder(z)               # [B, F, N]
        return recon.transpose(1, 2)          # [B, N, F]

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.transpose(1, 2))


def reconstruction_error(
    recon: torch.Tensor, x: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Маскированная MSE на окно: [B]. Паддинг-позиции не учитываются."""
    se = (recon - x) ** 2                       # [B, N, F]
    m = mask.unsqueeze(-1).to(se.dtype)         # [B, N, 1]
    per_window = (se * m).sum(dim=(1, 2)) / (m.sum(dim=(1, 2)) * x.shape[-1]).clamp(min=1)
    return per_window
