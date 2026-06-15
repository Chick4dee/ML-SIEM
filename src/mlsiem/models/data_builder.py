"""Сборка train/val датасетов из размеченных parquet (общий код для обучения).

Шаги: склейка parquet → хронологический сплит (без ликеджа) → fit препроцессора
на train → transform обеих частей → FlowWindowDataset. Препроцессор и энкодер
фитятся ТОЛЬКО на train и возвращаются для сохранения как serve-артефакты (#8).
"""

import glob
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from mlsiem.features.preprocessing import FeaturePreprocessor, time_split
from mlsiem.models.dataset import FlowWindowDataset, LabelEncoder

_META_COLS = ("src_ip", "bidirectional_first_seen_ms", "canonical")


@dataclass
class BuiltData:
    train: FlowWindowDataset
    val: FlowWindowDataset
    preprocessor: FeaturePreprocessor
    encoder: LabelEncoder
    feature_names: list[str]
    cat_positions: list[int]
    cat_cardinalities: list[int]
    class_counts: list[int]   # по train, в порядке encoder.classes


def _attach_meta(feat: pl.DataFrame, src: pl.DataFrame) -> pl.DataFrame:
    return feat.with_columns([src[c] for c in _META_COLS])


def build_datasets(
    parquet_glob: str,
    *,
    val_frac: float = 0.2,
    window: int = 16,
    label_col: str = "canonical",
) -> BuiltData:
    files = sorted(glob.glob(parquet_glob))
    if not files:
        raise FileNotFoundError(f"нет parquet по шаблону: {parquet_glob}")
    raw = pl.concat([pl.read_parquet(f) for f in files])
    # потоки без метки (unmatched при разметке) не учим — не превращаем в шум
    raw = raw.filter(pl.col(label_col).is_not_null())

    train_raw, val_raw = time_split(raw, val_frac=val_frac)
    pre = FeaturePreprocessor.fit(train_raw)
    enc = LabelEncoder.fit(raw[label_col])

    train_feat = _attach_meta(pre.transform(train_raw), train_raw)
    val_feat = _attach_meta(pre.transform(val_raw), val_raw)

    train_ds = FlowWindowDataset(train_feat, pre.feature_names, enc,
                                 label_col=label_col, window=window)
    val_ds = FlowWindowDataset(val_feat, pre.feature_names, enc,
                               label_col=label_col, window=window)

    # позиции и мощности категориальных фич (для эмбеддингов модели)
    names = pre.feature_names
    cat_positions = [names.index(c) for c in pre.vocabularies]
    cat_cardinalities = [len(pre.vocabularies[c]) + 1 for c in pre.vocabularies]  # +1 OOV

    counts_map = {r[label_col]: r["count"]
                  for r in train_raw[label_col].value_counts().iter_rows(named=True)}
    class_counts = [counts_map.get(c, 0) for c in enc.classes]

    return BuiltData(
        train=train_ds, val=val_ds, preprocessor=pre, encoder=enc,
        feature_names=names, cat_positions=cat_positions,
        cat_cardinalities=cat_cardinalities, class_counts=class_counts,
    )


def save_artifacts(data: BuiltData, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data.preprocessor.save(out / "preprocessor.json")
    data.encoder.save(out / "label_encoder.json")
