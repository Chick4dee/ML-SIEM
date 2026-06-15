"""Отбор, нормализация и кодирование фич с сохранением fitted-артефактов.

Контракты PLAN.md (раздел 10.5):
- #8: scaler/encoder фитятся ТОЛЬКО на train, сохраняются и переиспользуются на
  serve; незнакомые категории (OOV) → индекс 0; общий код train/serve — здесь.
- #1/#12: в признаки НЕ идут идентификаторы (IP/MAC/OUI/порты) и абсолютные
  таймстемпы — иначе модель заучит топологию и время лаборатории вместо
  признаков атак. Оставляем только переносимые статистики потока.

Числовые фичи: log1p (все неотрицательны — счётчики/байты/длительности/размеры)
для гашения скоса, затем z-нормализация по train-статистике.
Категориальные: ординальный индекс по словарю train, 0 зарезервирован под OOV.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

# Поля, которые НИКОГДА не признаки (идентификаторы + абсолютное время + мета).
EXCLUDED = frozenset({
    "id", "expiration_id",
    "src_ip", "src_mac", "src_oui", "src_port",
    "dst_ip", "dst_mac", "dst_oui", "dst_port",
    "ip_version", "vlan_id", "tunnel_id",
    "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
    "src2dst_first_seen_ms", "src2dst_last_seen_ms",
    "dst2src_first_seen_ms", "dst2src_last_seen_ms",
    # payload-метаданные — зона Эксперта 3, не flow-признаки
    "requested_server_name", "client_fingerprint", "server_fingerprint",
    "user_agent", "content_type",
    # таргеты и служебное
    "label", "label_class", "canonical", "scenario",
})

# Категориальные фичи (переносимая идентичность сервиса по nDPI).
CATEGORICAL = ("application_name", "application_category_name")

_TIME_COL = "bidirectional_first_seen_ms"
_STD_FLOOR = 1e-6


def select_feature_columns(df: pl.DataFrame) -> list[str]:
    """Числовые фичи = все колонки, кроме исключённых и категориальных."""
    return [
        c for c, dt in df.schema.items()
        if c not in EXCLUDED and c not in CATEGORICAL and dt.is_numeric()
    ]


class FeaturePreprocessor:
    """Фитится на train, сохраняется, переиспользуется на serve (контракт #8)."""

    def __init__(self, numeric: list[str], means: list[float], stds: list[float],
                 vocabularies: dict[str, list[str]]):
        self.numeric = numeric
        self.means = np.asarray(means, np.float64)
        self.stds = np.asarray(stds, np.float64)
        self.vocabularies = vocabularies

    @classmethod
    def fit(cls, df: pl.DataFrame, numeric: list[str] | None = None) -> "FeaturePreprocessor":
        numeric = numeric or select_feature_columns(df)
        logged = df.select(
            [pl.col(c).cast(pl.Float64).fill_null(0.0).log1p().alias(c) for c in numeric]
        )
        means = [logged[c].mean() for c in numeric]
        stds = [max(logged[c].std() or 0.0, _STD_FLOOR) for c in numeric]
        vocabularies = {
            col: df[col].drop_nulls().unique().sort().to_list()
            for col in CATEGORICAL if col in df.columns
        }
        return cls(numeric, means, stds, vocabularies)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Возвращает DataFrame только из фич: нормализованные числовые +
        категориальные как ординальный индекс (0 = OOV). Порядок колонок
        стабилен (numeric, затем categorical) — контракт для windowing."""
        num = df.select(
            [pl.col(c).cast(pl.Float64).fill_null(0.0).log1p().alias(c) for c in self.numeric]
        ).to_numpy()
        num = (num - self.means) / self.stds

        out = pl.from_numpy(num, schema=self.numeric)

        cat_cols = []
        for col, vocab in self.vocabularies.items():
            mapping = {v: i + 1 for i, v in enumerate(vocab)}  # 0 — OOV
            idx = df[col].replace_strict(mapping, default=0, return_dtype=pl.Int32)
            cat_cols.append(idx.alias(col))
        if cat_cols:
            out = out.with_columns(cat_cols)
        return out

    @property
    def feature_names(self) -> list[str]:
        return [*self.numeric, *self.vocabularies.keys()]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "numeric": self.numeric,
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "vocabularies": self.vocabularies,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePreprocessor":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["numeric"], d["means"], d["stds"], d["vocabularies"])


def time_split(
    df: pl.DataFrame,
    *,
    val_frac: float = 0.2,
    time_col: str = _TIME_COL,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Хронологический сплit без ликеджа (контракт #12): ранние потоки — train,
    поздние — val. Случайный сплит при stride=1 утёк бы (соседние окна почти
    идентичны)."""
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac должен быть в (0, 1)")
    ordered = df.sort(time_col)
    cut = int(len(ordered) * (1.0 - val_frac))
    return ordered.head(cut), ordered.tail(len(ordered) - cut)
