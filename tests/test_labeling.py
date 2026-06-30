"""Тесты разметки: таймзона, ориентация потока, допуск по времени, hex-порты."""

import polars as pl

from mlsiem.data.labeling import label_flows, load_binetflow

BINETFLOW_HEADER = "StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label"  # noqa: E501


def _write_binetflow(tmp_path, rows: list[str]):
    path = tmp_path / "gt.binetflow"
    path.write_text("\n".join([BINETFLOW_HEADER, *rows]), encoding="utf-8")
    return path


def _flows(rows: list[tuple]) -> pl.DataFrame:
    cols = ["bidirectional_first_seen_ms", "src_ip", "src_port", "dst_ip", "dst_port", "protocol"]
    return pl.DataFrame([dict(zip(cols, r, strict=True)) for r in rows])


def test_load_binetflow_converts_prague_time_to_utc(tmp_path):
    # 11:04:24 CEST (лето, UTC+2) == 09:04:24 UTC
    path = _write_binetflow(
        tmp_path,
        ["2011/08/10 11:04:24.000000,1.0,tcp,10.0.0.1,1577,   ->,10.0.0.2,80,S_RA,0,0,4,276,156,flow=From-Botnet-V42-TCP"],  # noqa: E501
    )
    gt = load_binetflow(path)

    import datetime as dt

    expected = dt.datetime(2011, 8, 10, 9, 4, 24, tzinfo=dt.UTC).timestamp() * 1000
    assert gt["gt_start_ms"][0] == int(expected)
    assert gt["gt_end_ms"][0] == int(expected) + 1000  # Dur = 1.0 c
    assert gt["protocol"][0] == 6
    assert gt["label_class"][0] == "botnet"


def test_load_binetflow_parses_hex_ports_and_classes(tmp_path):
    path = _write_binetflow(
        tmp_path,
        [
            "2011/08/10 11:00:00.000000,1.0,icmp,10.0.0.1,0x000c,   ->,10.0.0.2,0x0008,ECO,0,0,1,60,60,flow=Background-ICMP",  # noqa: E501
            "2011/08/10 11:00:01.000000,1.0,udp,10.0.0.3,53,   ->,10.0.0.4,5353,CON,0,0,2,120,60,flow=From-Normal-V42-UDP",  # noqa: E501
        ],
    )
    gt = load_binetflow(path)

    assert gt["src_port"][0] == 12
    assert gt["dst_port"][0] == 8
    assert gt["label_class"].to_list() == ["background", "normal"]


def test_label_flows_matches_both_orientations_and_nearest_time():
    gt = pl.DataFrame(
        {
            "gt_start_ms": [1000, 5000, 100_000],
            "gt_end_ms": [2000, 6000, 101_000],
            "src_ip": ["10.0.0.1", "10.0.0.1", "10.0.0.9"],
            "src_port": [1111, 1111, 9999],
            "dst_ip": ["10.0.0.2", "10.0.0.2", "10.0.0.8"],
            "dst_port": [80, 80, 443],
            "protocol": [6, 6, 6],
            "label": ["A-early", "A-late", "C"],
            "label_class": ["botnet", "background", "normal"],
        }
    )
    flows = _flows(
        [
            (4500, "10.0.0.1", 1111, "10.0.0.2", 80, 6),  # ближе к A-late (Δ=500 vs 3500)
            (1200, "10.0.0.2", 80, "10.0.0.1", 1111, 6),  # обратная ориентация → A-early
            (500, "10.0.0.5", 5, "10.0.0.6", 6, 6),       # нет пары
        ]
    )
    out = label_flows(flows, gt)

    assert out["label"].to_list() == ["A-late", "A-early", None]
    assert out["label_class"].to_list() == ["background", "botnet", None]


def test_label_flows_matches_flood_segment_inside_gt_interval():
    # длинная GT-запись (час флуда): сегмент nfstream стартует в середине —
    # старт-к-старту дал бы Δ=30 мин >> допуска, интервальный матч обязан найти
    gt = pl.DataFrame(
        {
            "gt_start_ms": [0],
            "gt_end_ms": [3_600_000],
            "src_ip": ["10.0.0.1"],
            "src_port": [1],
            "dst_ip": ["10.0.0.2"],
            "dst_port": [2],
            "protocol": [17],
            "label": ["flood"],
            "label_class": ["botnet"],
        }
    )
    flows = _flows([(1_800_000, "10.0.0.1", 1, "10.0.0.2", 2, 17)])
    out = label_flows(flows, gt, tolerance_ms=60_000)
    assert out["label_class"][0] == "botnet"


def test_label_flows_matches_icmp_by_3tuple_ignoring_ports():
    # nfstream: порты ICMP = 0; Argus: в портах hex-код типа (0x0008) —
    # по 5-tuple такие потоки не сматчатся никогда, нужен 3-tuple
    gt = pl.DataFrame(
        {
            "gt_start_ms": [1000],
            "gt_end_ms": [2000],
            "src_ip": ["10.0.0.1"],
            "src_port": [8],  # из 0x0008
            "dst_ip": ["10.0.0.2"],
            "dst_port": [0],
            "protocol": [1],
            "label": ["icmp-ping"],
            "label_class": ["botnet"],
        }
    )
    flows = _flows(
        [
            (1500, "10.0.0.1", 0, "10.0.0.2", 0, 1),   # ICMP: матч по 3-tuple
            (1500, "10.0.0.1", 0, "10.0.0.2", 0, 6),   # TCP с теми же IP: порты
                                                        # обязаны совпадать — мимо
        ]
    )
    out = label_flows(flows, gt)
    assert out["label_class"].to_list() == ["botnet", None]


def test_label_flows_respects_tolerance():
    gt = pl.DataFrame(
        {
            "gt_start_ms": [0],
            "gt_end_ms": [1000],
            "src_ip": ["10.0.0.1"],
            "src_port": [1],
            "dst_ip": ["10.0.0.2"],
            "dst_port": [2],
            "protocol": [6],
            "label": ["X"],
            "label_class": ["botnet"],
        }
    )
    flows = _flows([(120_000, "10.0.0.1", 1, "10.0.0.2", 2, 6)])

    strict = label_flows(flows, gt, tolerance_ms=60_000)
    loose = label_flows(flows, gt, tolerance_ms=300_000)
    assert strict["label"][0] is None
    assert loose["label"][0] == "X"


def test_label_flows_keeps_flow_count_and_columns():
    gt = pl.DataFrame(
        {
            "gt_start_ms": [0],
            "gt_end_ms": [1000],
            "src_ip": ["10.0.0.1"],
            "src_port": [1],
            "dst_ip": ["10.0.0.2"],
            "dst_port": [2],
            "protocol": [6],
            "label": ["X"],
            "label_class": ["botnet"],
        }
    )
    flows = _flows([(100, "10.0.0.1", 1, "10.0.0.2", 2, 6)]).with_columns(
        pl.lit(42).alias("extra_feature")
    )
    out = label_flows(flows, gt)

    assert len(out) == len(flows)
    assert "extra_feature" in out.columns
    assert out["label"][0] == "X"
