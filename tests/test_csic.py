"""Тесты парсера CSIC: разбор блоков, извлечение URL+тела, метки."""

import polars as pl

from mlsiem.features.csic import _block_to_text, _blocks, load_csic, parse_file


def test_blocks_split_on_request_line():
    lines = [
        "GET http://h/a HTTP/1.1", "Host: h", "",
        "POST http://h/b HTTP/1.1", "Host: h", "Content-Length: 3", "", "x=1",
    ]
    blocks = _blocks(lines)
    assert len(blocks) == 2
    assert blocks[0][0].startswith("GET")
    assert blocks[1][0].startswith("POST")


def test_get_text_is_method_and_url():
    block = ["GET http://h/p?id=1 HTTP/1.1", "Host: h", "Cookie: x"]
    assert _block_to_text(block) == "GET http://h/p?id=1"


def test_post_text_includes_body():
    block = ["POST http://h/p HTTP/1.1", "Host: h", "Content-Length: 7", "", "id=DROP"]
    text = _block_to_text(block)
    assert text.startswith("POST http://h/p")
    assert "id=DROP" in text


def test_sqli_payload_preserved(tmp_path):
    raw = ("GET http://h/p?q=%27%3B+DROP+TABLE+usuarios HTTP/1.1\n"
           "Host: h\n\n")
    f = tmp_path / "anomalousTrafficTest.txt"
    f.write_text(raw, encoding="latin-1")
    texts = parse_file(f)
    assert len(texts) == 1
    assert "DROP+TABLE" in texts[0]


def test_load_csic_labels(tmp_path):
    (tmp_path / "normalTrafficTraining.txt").write_text(
        "GET http://h/ok HTTP/1.1\nHost: h\n\n", encoding="latin-1")
    (tmp_path / "anomalousTrafficTest.txt").write_text(
        "GET http://h/bad?q=1 HTTP/1.1\nHost: h\n\n", encoding="latin-1")
    df = load_csic(tmp_path)
    assert set(df["label"].unique()) == {"normal", "anomalous"}
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 2
