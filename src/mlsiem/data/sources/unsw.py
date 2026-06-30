"""Пайплайн UNSW-NB15: PCAP → размеченные parquet.

Отличия от CTU-13 (см. ctu13.py):
- Ground truth (`NUSW-NB15_GT.csv`) перечисляет ТОЛЬКО атаки; всё, что не
  сматчилось, — benign-фон (генератор трафика IXIA), а не unknown. Поэтому
  несматченные потоки помечаются normal, не null.
- Время GT — epoch-СЕКУНДЫ (не строки): зона не нужна, epoch совпадает с PCAP
  напрямую (проверено: GT и pcap дня 22-1 выровнены без сдвига).
- 9 грязных категорий атак (регистр, пробелы, ед./мн.) → нормализация +
  единая таксономия (taxonomy.py).

Запуск:

    python -m mlsiem.data.sources.unsw --dataset "A:/datasets/UNSW-NB15/OneDrive_2_12.06.2026" ^
        --output data/processed/unsw
"""

import argparse
from pathlib import Path

import polars as pl

from mlsiem.data.extractor import extract_flows
from mlsiem.data.labeling import _GT_COLS, _PROTO_NUMBERS, label_flows
from mlsiem.data.taxonomy import to_canonical

GT_FILENAME = "CSV Files/NUSW-NB15_GT.csv"
PCAP_GLOB = "pcap files/pcaps */*.pcap"

BENIGN_LABEL = "normal"


def load_unsw_gt(path) -> pl.DataFrame:
    """Читает NUSW-NB15_GT.csv в нормализованный вид (_GT_COLS).

    label_class — категория атаки, очищенная (strip+lower); label — подробное
    «Attack Name». Время epoch-секунды → ms.
    """
    raw = pl.read_csv(path, truncate_ragged_lines=True, infer_schema_length=0)
    return raw.select(
        (pl.col("Start time").cast(pl.Int64) * 1000).alias("gt_start_ms"),
        (pl.col("Last time").cast(pl.Int64) * 1000).alias("gt_end_ms"),
        pl.col("Source IP").alias("src_ip"),
        pl.col("Source Port").cast(pl.Int64, strict=False).alias("src_port"),
        pl.col("Destination IP").alias("dst_ip"),
        pl.col("Destination Port").cast(pl.Int64, strict=False).alias("dst_port"),
        pl.col("Protocol").str.to_lowercase().str.strip_chars()
        .replace_strict(_PROTO_NUMBERS, default=None).alias("protocol"),
        pl.col("Attack Name").alias("label"),
        pl.col("Attack category").str.strip_chars().str.to_lowercase().alias("label_class"),
    ).select(_GT_COLS)


def add_canonical(labeled: pl.DataFrame, source: str) -> pl.DataFrame:
    """Добавляет колонку canonical из label_class по таксономии источника."""
    mapping = {
        c: to_canonical(source, c)
        for c in labeled["label_class"].drop_nulls().unique().to_list()
    }
    return labeled.with_columns(
        pl.col("label_class").replace_strict(mapping, default=None).alias("canonical")
    )


def process_pcap(pcap: Path, gt: pl.DataFrame) -> pl.DataFrame:
    """Один PCAP: извлечь фичи, разметить, несматченное → benign, добавить canonical."""
    flows = extract_flows(pcap)
    labeled = label_flows(flows, gt).with_columns(
        pl.col("label").fill_null(BENIGN_LABEL),
        pl.col("label_class").fill_null(BENIGN_LABEL),
    )
    return add_canonical(labeled, "unsw")


def main() -> None:
    parser = argparse.ArgumentParser(description="UNSW-NB15 → размеченные parquet")
    parser.add_argument("--dataset", required=True, help="папка OneDrive_2_… с CSV и pcap")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    gt = load_unsw_gt(dataset / GT_FILENAME)
    print(f"GT загружен: {len(gt)} записей атак", flush=True)

    stats = []
    for pcap in sorted(dataset.glob(PCAP_GLOB)):
        labeled = process_pcap(pcap, gt)
        day = pcap.parent.name.replace("pcaps ", "")
        name = f"{day}__{pcap.stem}.parquet"
        labeled.write_parquet(output / name)

        n = len(labeled)
        attacks = int((labeled["label_class"] != BENIGN_LABEL).sum())
        stats.append({"file": name, "flows": n, "attacks": attacks,
                      "attack_rate": round(attacks / n, 4) if n else 0.0})
        print(f"{name}: {n} потоков, атак {attacks} ({100 * attacks / n:.1f}%)", flush=True)

    summary = pl.DataFrame(stats)
    summary.write_csv(output / "summary.csv")
    tot = summary["flows"].sum()
    att = summary["attacks"].sum()
    print(f"итого: {tot} потоков, атак {att} ({100 * att / tot:.2f}%)")


if __name__ == "__main__":
    main()
