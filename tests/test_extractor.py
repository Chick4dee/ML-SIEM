"""Интеграционный тест переэкстракции: эталонный PCAP → потоки с фичами."""

from pathlib import Path

import pytest

pytest.importorskip("nfstream", reason="нужен extras [capture] (+ Npcap на Windows)")

from mlsiem.features.extractor import extract_flows, extract_to_parquet  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "google_ssl.pcap"


def test_extract_flows_returns_expected_features():
    df = extract_flows(FIXTURE)

    assert len(df) == 1
    flow = df.row(0, named=True)
    assert flow["src_ip"] == "172.31.3.224"
    assert flow["application_name"] == "TLS"
    assert flow["bidirectional_packets"] == 28
    # статистические фичи включены (контракт STREAMER_SETTINGS)
    assert "bidirectional_mean_ps" in df.columns
    # event-time поля на месте — нужны windowing-модулю
    assert flow["bidirectional_first_seen_ms"] > 0


def test_extract_to_parquet_roundtrip(tmp_path):
    import polars as pl

    out = extract_to_parquet(FIXTURE, tmp_path / "flows.parquet")
    assert out.exists()
    df = pl.read_parquet(out)
    assert len(df) == 1
    assert df["dst_ip"][0] == "216.58.212.100"
