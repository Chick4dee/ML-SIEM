"""Сшивка nfstream-потоков с ground-truth разметкой (PLAN.md, раздел 10.5, #2).

Переэкстракция nfstream нарезает потоки иначе, чем экстрактор авторов датасета
(Argus и т.п.), поэтому метки переносятся маппингом: 5-tuple + ближайшее время
старта в пределах допуска, с учётом обеих ориентаций потока.

Подводный камень времени: PCAP хранит UTC, а текстовые выгрузки датасетов —
часто локальное время стенда (CTU-13 — Прага, CEST). Таймзона источника
указывается явно; наивный парсинг сдвинул бы все метки на часы.
"""

import polars as pl

# Таймзона стенда CTU University (захваты 2011 г.)
CTU13_TIMEZONE = "Europe/Prague"

_PROTO_NUMBERS = {"tcp": 6, "udp": 17, "icmp": 1, "igmp": 2}

_GT_COLS = ["gt_start_ms", "src_ip", "src_port", "dst_ip", "dst_port", "protocol",
            "label", "label_class"]


def _parse_port(col: str) -> pl.Expr:
    """Порт из binetflow: десятичный, шестнадцатеричный (ICMP) или пустой."""
    raw = pl.col(col).cast(pl.String).str.strip_chars()
    return (
        pl.when(raw.str.starts_with("0x"))
        .then(raw.str.slice(2).str.to_integer(base=16, strict=False))
        .otherwise(raw.str.to_integer(base=10, strict=False))
        .alias(col)
    )


def _label_class() -> pl.Expr:
    label = pl.col("Label")
    return (
        pl.when(label.str.contains("(?i)botnet")).then(pl.lit("botnet"))
        .when(label.str.contains("(?i)normal")).then(pl.lit("normal"))
        .otherwise(pl.lit("background"))
        .alias("label_class")
    )


def load_binetflow(path, *, timezone: str = CTU13_TIMEZONE) -> pl.DataFrame:
    """Читает .binetflow (Argus CSV из CTU-13) в нормализованный вид.

    Возвращает колонки: gt_start_ms (UTC epoch ms), src_ip, src_port, dst_ip,
    dst_port, protocol (номер IANA), label (сырая строка), label_class
    (botnet / normal / background).
    """
    return (
        pl.scan_csv(path, schema_overrides={"Sport": pl.String, "Dport": pl.String})
        .with_columns(
            pl.col("StartTime")
            .str.strptime(pl.Datetime("ms"), "%Y/%m/%d %H:%M:%S%.f")
            .dt.replace_time_zone(timezone, ambiguous="earliest")
            .dt.convert_time_zone("UTC")
            .dt.timestamp("ms")
            .alias("gt_start_ms"),
            pl.col("Proto").str.to_lowercase()
            .replace_strict(_PROTO_NUMBERS, default=None)
            .alias("protocol"),
            _parse_port("Sport"),
            _parse_port("Dport"),
            _label_class(),
        )
        .rename({"SrcAddr": "src_ip", "DstAddr": "dst_ip",
                 "Sport": "src_port", "Dport": "dst_port", "Label": "label"})
        .select(_GT_COLS)
        .collect()
    )


def label_flows(
    flows: pl.DataFrame,
    gt: pl.DataFrame,
    *,
    tolerance_ms: int = 60_000,
) -> pl.DataFrame:
    """Переносит метки gt на nfstream-потоки.

    Матч: одинаковый 5-tuple (в любой из двух ориентаций) и минимальная
    |Δt стартов| в пределах tolerance_ms. Потоки без пары получают
    label/label_class = null — решение об их судьбе принимает следующий шаг
    (для бенчмарка обычно отбрасываются, чтобы не учить на шуме).
    """
    keys = ["src_ip", "src_port", "dst_ip", "dst_port", "protocol"]
    gt_fwd = gt.select(_GT_COLS)
    gt_rev = gt_fwd.rename({"src_ip": "dst_ip", "dst_ip": "src_ip",
                            "src_port": "dst_port", "dst_port": "src_port"})
    gt_both = pl.concat([gt_fwd, gt_rev.select(_GT_COLS)])

    indexed = flows.with_row_index("_flow_idx")
    matched = (
        indexed.select(["_flow_idx", "bidirectional_first_seen_ms", *keys])
        .join(gt_both, on=keys, how="inner")
        .with_columns(
            (pl.col("bidirectional_first_seen_ms") - pl.col("gt_start_ms"))
            .abs().alias("_dt_ms")
        )
        .filter(pl.col("_dt_ms") <= tolerance_ms)
        .sort("_dt_ms")
        .unique(subset="_flow_idx", keep="first")
        .select(["_flow_idx", "label", "label_class"])
    )
    return (
        indexed.join(matched, on="_flow_idx", how="left")
        .drop("_flow_idx")
    )
