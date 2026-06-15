"""Единый словарь классов атак (PLAN.md, раздел 10.5, #10).

У каждого датасета своя номенклатура: CTU-13 — botnet/normal/background,
UNSW-NB15 — 9 категорий с грязными подписями (регистр, пробелы, ед./мн. число).
Мета-слой и общий мультиклассовый выход требуют единой таксономии — сюда
сводятся подписи всех источников.
"""

# Канонические классы проекта. benign — отрицательный класс; background —
# непроверенный реальный фон CTU-13 (не гарантированно benign, держим отдельно).
CANONICAL = frozenset({
    "benign",
    "background",
    "botnet",
    "dos",
    "exploits",
    "fuzzers",
    "generic",
    "reconnaissance",
    "backdoor",
    "analysis",
    "shellcode",
    "worms",
})

# Подписи источников → канонический класс. Ключи приводятся к нижнему регистру
# и обрезаются по пробелам перед поиском (см. to_canonical).
_SOURCE_MAP = {
    "ctu13": {
        "botnet": "botnet",
        "normal": "benign",
        "background": "background",
    },
    "unsw": {
        "normal": "benign",
        "fuzzers": "fuzzers",
        "analysis": "analysis",
        "backdoor": "backdoor",
        "backdoors": "backdoor",
        "dos": "dos",
        "exploits": "exploits",
        "generic": "generic",
        "reconnaissance": "reconnaissance",
        "shellcode": "shellcode",
        "worms": "worms",
    },
}


def to_canonical(source: str, raw_label: str | None) -> str | None:
    """Канонический класс по подписи источника. None → None (несматченное)."""
    if raw_label is None:
        return None
    key = raw_label.strip().lower()
    try:
        mapping = _SOURCE_MAP[source]
    except KeyError as exc:
        raise ValueError(f"неизвестный источник: {source!r}") from exc
    if key not in mapping:
        raise ValueError(f"подпись {raw_label!r} не отображена для источника {source!r}")
    return mapping[key]
