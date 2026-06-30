"""Переэкстракция фич из PCAP единым экстрактором (nfstream).

Контракт train/serve (PLAN.md, раздел 10.5, #1): и обучение (PCAP публичных
датасетов), и live-захват проходят через NFStreamer с одинаковыми настройками —
фичи идентичны по построению. Настройки меняются только здесь и только синхронно
для всех потребителей.

Windows: требуется установленный Npcap (WinPcap API-compatible mode). Вызывающий
код обязан работать под guard ``if __name__ == "__main__"`` — nfstream порождает
дочерние процессы (multiprocessing/spawn).
"""

from pathlib import Path

import polars as pl
from nfstream import NFStreamer

# Единые настройки NFStreamer для train (PCAP) и serve (live).
STREAMER_SETTINGS: dict = {
    # статистические фичи потока (мин/макс/среднее/std размеров и межпакетных интервалов)
    "statistical_analysis": True,
    # SPLT-анализ отключён: длины/направления первых пакетов пока не используем
    "splt_analysis": 0,
    # nDPI-диссекция первых N пакетов — даёт application_name/category
    "n_dissections": 20,
}


def extract_flows(source: str | Path, **overrides) -> pl.DataFrame:
    """Прогоняет источник (PCAP-файл или сетевой интерфейс) через nfstream.

    Возвращает потоки таблицей polars; колонки — фичи nfstream как есть,
    отбор/нормализация — ответственность следующего шага пайплайна.
    """
    settings = {**STREAMER_SETTINGS, **overrides}
    streamer = NFStreamer(source=str(source), **settings)
    return pl.from_pandas(streamer.to_pandas())


def extract_to_parquet(source: str | Path, output: str | Path) -> Path:
    """PCAP → parquet. Возвращает путь к записанному файлу."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    extract_flows(source).write_parquet(output)
    return output
