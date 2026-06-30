"""Тесты загрузчика UNSW GT и логики benign-по-умолчанию."""

import polars as pl

from mlsiem.data.labeling import label_flows
from mlsiem.data.sources.unsw import add_canonical, load_unsw_gt

GT_HEADER = ("Start time,Last time,Attack category,Attack subcategory,Protocol,"
             "Source IP,Source Port,Destination IP,Destination Port,Attack Name,"
             "Attack Reference,.")


def _write_gt(tmp_path, rows: list[str]):
    path = tmp_path / "gt.csv"
    path.write_text("\n".join([GT_HEADER, *rows]), encoding="utf-8")
    return path


def _flows(rows: list[tuple]) -> pl.DataFrame:
    cols = ["bidirectional_first_seen_ms", "src_ip", "src_port", "dst_ip", "dst_port", "protocol"]
    return pl.DataFrame([dict(zip(cols, r, strict=True)) for r in rows])


def test_load_unsw_gt_seconds_to_ms_and_normalizes(tmp_path):
    path = _write_gt(tmp_path, [
        "1421927414,1421927416, Reconnaissance ,HTTP,tcp,175.45.176.0,13284,149.171.126.16,80,Recon X,-,.",  # noqa: E501
    ])
    gt = load_unsw_gt(path)

    assert gt["gt_start_ms"][0] == 1421927414_000
    assert gt["gt_end_ms"][0] == 1421927416_000  # секунды → ms
    assert gt["protocol"][0] == 6
    assert gt["label_class"][0] == "reconnaissance"  # strip + lower
    assert gt["label"][0] == "Recon X"


def test_unmatched_flows_become_benign(tmp_path):
    path = _write_gt(tmp_path, [
        "1000,1001,Exploits,Browser,tcp,175.45.176.2,23357,149.171.126.16,80,Exploit Y,-,.",
    ])
    gt = load_unsw_gt(path)
    flows = _flows([
        (1_000_500, "175.45.176.2", 23357, "149.171.126.16", 80, 6),  # атака (матч)
        (2_000_000, "59.166.0.5", 1234, "149.171.126.9", 53, 17),      # фон → benign
    ])
    labeled = label_flows(flows, gt).with_columns(
        pl.col("label").fill_null("normal"),
        pl.col("label_class").fill_null("normal"),
    )
    out = add_canonical(labeled, "unsw")

    assert out["label_class"].to_list() == ["exploits", "normal"]
    assert out["canonical"].to_list() == ["exploits", "benign"]
