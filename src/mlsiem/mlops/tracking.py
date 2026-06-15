"""Настройка трекинга экспериментов MLflow (локальный SQLite-бэкенд).

MLflow 3.x вывел file-store (`./mlruns`) в maintenance-режим, поэтому метаданные
пишем в SQLite (`mlflow.db` в корне репо), а артефакты — в `mlartifacts/`. Оба
в git не идут (см. .gitignore). UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
"""

from pathlib import Path

import mlflow

DEFAULT_EXPERIMENT = "mlsiem"


def _repo_root() -> Path:
    # src/mlsiem/mlops/tracking.py → корень на 3 уровня выше
    return Path(__file__).resolve().parents[3]


def setup_tracking(experiment: str = DEFAULT_EXPERIMENT) -> str:
    """Указывает MLflow на локальный SQLite-стор и выбирает эксперимент.

    Возвращает tracking URI.
    """
    root = _repo_root()
    uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(root / "mlartifacts").as_uri())
    mlflow.set_experiment(experiment)
    return uri
