"""Пайплайн CTU-13: PCAP всех сценариев → размеченные parquet.

Запуск (из корня репозитория, в venv):

    python -m mlsiem.data.sources.ctu13 --dataset A:/datasets/CTU-13/CTU-13-Dataset ^
        --output data/processed/ctu13

Каждый сценарий: переэкстракция nfstream → перенос меток из .binetflow →
``scenario_NN.parquet``; в конце — сводка ``summary.csv`` по качеству матчинга.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from mlsiem.data.extractor import extract_flows
from mlsiem.data.labeling import label_flows, load_binetflow


@dataclass
class ScenarioFiles:
    scenario: int
    pcap: Path
    binetflow: Path


def discover_scenarios(dataset_dir: str | Path) -> list[ScenarioFiles]:
    """Находит пары (pcap, binetflow) в папках-сценариях CTU-13 (имена «1»…«13»)."""
    dataset_dir = Path(dataset_dir)
    found: list[ScenarioFiles] = []
    for d in dataset_dir.iterdir():
        if not d.is_dir() or not d.name.isdigit():
            continue
        pcaps = sorted(d.glob("*.pcap"))
        gts = sorted(d.glob("*.binetflow"))
        if len(pcaps) != 1 or len(gts) != 1:
            raise FileNotFoundError(
                f"сценарий {d.name}: ожидался ровно один pcap и один binetflow, "
                f"найдено {len(pcaps)} и {len(gts)}"
            )
        found.append(ScenarioFiles(int(d.name), pcaps[0], gts[0]))
    if not found:
        raise FileNotFoundError(f"в {dataset_dir} не найдено папок-сценариев CTU-13")
    return sorted(found, key=lambda s: s.scenario)


def process_scenario(files: ScenarioFiles, output_dir: str | Path) -> dict:
    """Один сценарий: извлечь, разметить, сохранить. Возвращает статистику."""
    flows = extract_flows(files.pcap)
    gt = load_binetflow(files.binetflow)
    labeled = label_flows(flows, gt).with_columns(
        pl.lit(files.scenario, dtype=pl.Int32).alias("scenario")
    )

    output = Path(output_dir) / f"scenario_{files.scenario:02d}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    labeled.write_parquet(output)

    n = len(labeled)
    matched = int(labeled["label_class"].is_not_null().sum())
    return {
        "scenario": files.scenario,
        "flows": n,
        "matched": matched,
        "match_rate": round(matched / n, 4) if n else 0.0,
        "botnet": int((labeled["label_class"] == "botnet").sum()),
        "normal": int((labeled["label_class"] == "normal").sum()),
        "background": int((labeled["label_class"] == "background").sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CTU-13 → размеченные parquet")
    parser.add_argument("--dataset", required=True, help="папка CTU-13-Dataset")
    parser.add_argument("--output", required=True, help="папка для parquet и summary.csv")
    args = parser.parse_args()

    stats = []
    for files in discover_scenarios(args.dataset):
        s = process_scenario(files, args.output)
        stats.append(s)
        print(
            f"сценарий {s['scenario']:>2}: {s['flows']:>7} потоков, "
            f"матч {s['match_rate'] * 100:5.1f}%, botnet {s['botnet']}",
            flush=True,
        )

    summary = pl.DataFrame(stats).sort("scenario")
    summary.write_csv(Path(args.output) / "summary.csv")
    total = summary["flows"].sum()
    total_matched = summary["matched"].sum()
    print(f"итого: {total} потоков, матч {100 * total_matched / total:.1f}%")


if __name__ == "__main__":
    main()
