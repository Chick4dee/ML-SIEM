"""Smoke-тест каркаса: пакет импортируется, версия на месте."""

import mlsiem


def test_package_importable():
    assert mlsiem.__version__ == "0.1.0"
