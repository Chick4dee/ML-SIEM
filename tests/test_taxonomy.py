"""Тесты единой таксономии классов."""

import pytest

from mlsiem.features.taxonomy import CANONICAL, to_canonical


def test_unsw_dirty_labels_normalized():
    # грязные подписи UNSW: регистр, пробелы, мн. число
    assert to_canonical("unsw", " Fuzzers ") == "fuzzers"
    assert to_canonical("unsw", "Backdoor") == "backdoor"
    assert to_canonical("unsw", "Backdoors") == "backdoor"
    assert to_canonical("unsw", "normal") == "benign"


def test_ctu13_labels():
    assert to_canonical("ctu13", "botnet") == "botnet"
    assert to_canonical("ctu13", "normal") == "benign"
    assert to_canonical("ctu13", "background") == "background"


def test_none_passes_through():
    assert to_canonical("unsw", None) is None


def test_all_mapped_targets_are_canonical():
    for src in ("unsw", "ctu13"):
        for raw in ("normal", "botnet", "Fuzzers", "DoS", "background", "Worms"):
            try:
                result = to_canonical(src, raw)
            except ValueError:
                continue
            assert result is None or result in CANONICAL


def test_unknown_source_and_label_raise():
    with pytest.raises(ValueError, match="источник"):
        to_canonical("kaggle", "normal")
    with pytest.raises(ValueError, match="не отображена"):
        to_canonical("unsw", "ddos-of-the-future")
