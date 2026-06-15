"""Парсер CSIC 2010 HTTP → (текст запроса, метка) для Payload-эксперта (Э3).

Формат CSIC: текстовые файлы из блоков-запросов (request-line + заголовки +
тело для POST), блоки разделены пустыми строками. Сигнал web-атаки (SQLi/XSS/
brute) — в URL (query) и теле POST; заголовки почти константны (шум), их
отбрасываем. Текст запроса = METHOD + URL + тело (raw, percent-encoding как есть
— char-CNN учит и %27/DROP-паттерны напрямую).

Метки: normalTrafficTraining + normalTrafficTest = normal; anomalousTrafficTest
= anomalous (стандартная постановка CSIC).
"""

import re
from pathlib import Path

import polars as pl

_REQ_RE = re.compile(r"^(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH) \S+ HTTP/", re.IGNORECASE)

NORMAL_FILES = ("normalTrafficTraining.txt", "normalTrafficTest.txt")
ANOMALOUS_FILES = ("anomalousTrafficTest.txt",)


def _blocks(lines: list[str]) -> list[list[str]]:
    """Группирует строки в блоки-запросы по строке-началу запроса."""
    blocks: list[list[str]] = []
    cur: list[str] = []
    for line in lines:
        if _REQ_RE.match(line):
            if cur:
                blocks.append(cur)
            cur = [line]
        elif cur:
            cur.append(line)
    if cur:
        blocks.append(cur)
    return blocks


def _block_to_text(block: list[str]) -> str:
    """METHOD + URL + тело POST (если есть) в одну строку."""
    parts = block[0].split(" ")
    method = parts[0]
    url = parts[1] if len(parts) > 1 else ""
    body = ""
    # тело идёт после первой пустой строки внутри блока (конец заголовков)
    if "" in block[1:]:
        i = block.index("", 1)
        body = " ".join(line for line in block[i + 1:] if line.strip())
    text = f"{method} {url} {body}".strip()
    return re.sub(r"\s+", " ", text)


def parse_file(path: str | Path) -> list[str]:
    """Файл CSIC → список текстов запросов."""
    lines = Path(path).read_text(encoding="latin-1").splitlines()
    return [_block_to_text(b) for b in _blocks(lines)]


def load_csic(dataset_dir: str | Path) -> pl.DataFrame:
    """Собирает CSIC в DataFrame {text, label} (label: normal / anomalous)."""
    dataset_dir = Path(dataset_dir)
    rows: list[dict] = []
    for fname in NORMAL_FILES:
        p = dataset_dir / fname
        if p.exists():
            rows += [{"text": t, "label": "normal"} for t in parse_file(p)]
    for fname in ANOMALOUS_FILES:
        p = dataset_dir / fname
        if p.exists():
            rows += [{"text": t, "label": "anomalous"} for t in parse_file(p)]
    if not rows:
        raise FileNotFoundError(f"в {dataset_dir} не найдено файлов CSIC")
    return pl.DataFrame(rows)
