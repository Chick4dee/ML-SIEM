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

# Протоколы, у которых порты — настоящие (для остальных, как ICMP, nfstream
# пишет нули, а Argus — hex-код типа/кода: сравнивать порты бессмысленно)
_PORTED_PROTOCOLS = (6, 17)

_GT_COLS = ["gt_start_ms", "gt_end_ms", "src_ip", "src_port", "dst_ip", "dst_port",
            "protocol", "label", "label_class"]


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

    Возвращает колонки: gt_start_ms / gt_end_ms (UTC epoch ms; конец = старт +
    Dur), src_ip, src_port, dst_ip, dst_port, protocol (номер IANA), label
    (сырая строка), label_class (botnet / normal / background).
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
        .with_columns(
            (pl.col("gt_start_ms") + (pl.col("Dur") * 1000).cast(pl.Int64))
            .alias("gt_end_ms")
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

    Матч: одинаковый 5-tuple (в любой из двух ориентаций) и старт потока,
    ближайший к ИНТЕРВАЛУ [gt_start, gt_end] GT-записи (расстояние 0, если
    старт внутри интервала; иначе — до ближайшей границы, не более
    tolerance_ms). Интервал, а не старт-к-старту: nfstream режет длинные
    потоки по active_timeout на сегменты, чьи старты на десятки минут позже
    начала GT-записи (DDoS-флуды CTU-13 — именно такой случай).

    Протоколы без портов (ICMP и т.п.) матчатся по 3-tuple (ip, ip, protocol):
    у nfstream их порты нулевые, у Argus в портах закодирован тип/код ICMP —
    сравнение портов дало бы гарантированный промах.

    Потоки без пары получают label/label_class = null — решение об их судьбе
    принимает следующий шаг (для бенчмарка обычно отбрасываются, чтобы не
    учить на шуме).
    """
    gt_fwd = gt.select(_GT_COLS)
    gt_rev = gt_fwd.rename({"src_ip": "dst_ip", "dst_ip": "src_ip",
                            "src_port": "dst_port", "dst_port": "src_port"})
    gt_both = pl.concat([gt_fwd, gt_rev.select(_GT_COLS)])

    ported = pl.col("protocol").is_in(_PORTED_PROTOCOLS)
    indexed = flows.with_row_index("_flow_idx")
    flow_cols = ["_flow_idx", "bidirectional_first_seen_ms",
                 "src_ip", "src_port", "dst_ip", "dst_port", "protocol"]
    matched = pl.concat([
        _nearest_match(
            indexed.filter(ported).select(flow_cols),
            gt_both.filter(ported),
            ["src_ip", "src_port", "dst_ip", "dst_port", "protocol"],
            tolerance_ms,
        ),
        _nearest_match(
            indexed.filter(~ported).select(flow_cols),
            gt_both.filter(~ported),
            ["src_ip", "dst_ip", "protocol"],
            tolerance_ms,
        ),
    ])
    return (
        indexed.join(matched, on="_flow_idx", how="left")
        .drop("_flow_idx")
    )


def _nearest_match(
    flows_keyed: pl.DataFrame,
    gt_keyed: pl.DataFrame,
    keys: list[str],
    tolerance_ms: int,
) -> pl.DataFrame:
    """Ближайшая по времени GT-запись на каждый поток при совпадении keys."""
    return (
        flows_keyed.join(gt_keyed.select([*keys, "gt_start_ms", "gt_end_ms",
                                          "label", "label_class"]),
                         on=keys, how="inner")
        .with_columns(
            pl.max_horizontal(
                pl.col("gt_start_ms") - pl.col("bidirectional_first_seen_ms"),
                pl.col("bidirectional_first_seen_ms") - pl.col("gt_end_ms"),
                pl.lit(0, dtype=pl.Int64),
            ).alias("_dt_ms")
        )
        .filter(pl.col("_dt_ms") <= tolerance_ms)
        .sort("_dt_ms")
        .unique(subset="_flow_idx", keep="first")
        .select(["_flow_idx", "label", "label_class"])
    )
