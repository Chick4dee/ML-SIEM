"""Тесты обнаружения сценариев CTU-13."""

import pytest

from mlsiem.features.ctu13 import discover_scenarios


def _make_scenario(root, name: str, pcaps: int = 1, gts: int = 1):
    d = root / name
    d.mkdir()
    for i in range(pcaps):
        (d / f"capture-{i}.pcap").touch()
    for i in range(gts):
        (d / f"capture-{i}.binetflow").touch()
    return d


def test_discover_finds_and_sorts_numerically(tmp_path):
    for name in ["10", "2", "1"]:
        _make_scenario(tmp_path, name)
    (tmp_path / "README.txt").touch()
    _make_scenario(tmp_path, "not_a_scenario")  # не цифровое имя — игнорируется

    found = discover_scenarios(tmp_path)

    assert [s.scenario for s in found] == [1, 2, 10]
    assert found[0].pcap.suffix == ".pcap"
    assert found[0].binetflow.suffix == ".binetflow"


def test_discover_rejects_ambiguous_scenario(tmp_path):
    _make_scenario(tmp_path, "1", pcaps=2)
    with pytest.raises(FileNotFoundError, match="ровно один"):
        discover_scenarios(tmp_path)


def test_discover_rejects_empty_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match="не найдено"):
        discover_scenarios(tmp_path)
