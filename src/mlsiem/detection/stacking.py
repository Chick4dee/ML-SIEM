"""Мета-слой v2 — обучаемый stacking над выходами экспертов (PLAN.md, раздел 4).

Чем это лучше правил из meta.py (фаза 4)?
Наивное «ИЛИ» там флагует атаку, если ХОТЬ ОДИН эксперт сработал. Минус мы уже
увидели на оценке: мета унаследовала ложные срабатывания самого шумного эксперта
(AE), precision просел. Stacking решает это иначе: поверх выходов экспертов
ставится ещё одна, маленькая модель, которая на размеченных данных САМА учится,
кому и когда верить. Условно «Flow уверенно говорит exploits — доверяем Flow;
Flow мечется, но AE кричит про аномалию — это похоже на атаку, но неизвестно
какую». Эти правила не прописаны руками, а выведены из данных.

Важный момент про честность (контракт #10.5): стэкинг ОБЯЗАН учиться на выходах
экспертов, посчитанных на данных, которых эксперты НЕ видели при обучении. Иначе
на своих тренировочных примерах эксперты «слишком уверены», и стэкинг выучит
неправильные веса доверия. Поэтому весь pipeline ниже работает на hold-out.

Пока в стэке два эксперта (Flow + AE) — оба применимы к flow-окну. Payload/Host
живут на других данных (текст запроса, логи хоста); они присоединятся к стэку,
когда появится совместная разметка полигона (фаза 19). Конструкция к этому готова:
добавить эксперта — значит дописать его признаки в build_meta_features.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def _entropy(probs: np.ndarray) -> np.ndarray:
    """Энтропия распределения вероятностей по строкам — мера «растерянности»
    эксперта. Низкая = эксперт уверен в одном классе, высокая = размазал
    вероятности и сам не знает. Это сильный сигнал для стэкинга: уверенному
    эксперту можно доверять больше."""
    p = np.clip(probs, 1e-9, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def build_meta_features(
    flow_probs: np.ndarray,
    ae_scores: np.ndarray,
    ae_threshold: float,
) -> np.ndarray:
    """Собирает из выходов экспертов матрицу признаков для стэкинг-классификатора.

    На каждое окно склеиваем:
      - все вероятности классов от Flow (он уже «голосует» за конкретный класс);
      - насколько AE превысил свой порог — относительная величина аномалии
        (score/threshold; >1 значит «аномально»). Делим на порог, а не берём
        сырой score, чтобы число было сопоставимо между прогонами;
      - max вероятность Flow — насколько он вообще уверен в своём лучшем классе;
      - энтропию Flow — «растерянность» (см. выше);
      - бинарный флаг «AE сработал» — иногда стэкингу удобнее грубый да/нет, а не
        только непрерывная величина.
    """
    n_classes = flow_probs.shape[1]
    ae_ratio = (ae_scores / max(ae_threshold, 1e-9)).reshape(-1, 1)
    flow_max = flow_probs.max(axis=1, keepdims=True)
    flow_entropy = _entropy(flow_probs).reshape(-1, 1)
    ae_flag = (ae_scores > ae_threshold).astype(np.float64).reshape(-1, 1)
    # порядок столбцов фиксирован — стэкинг при инференсе ждёт ровно такой же
    feats = np.hstack([flow_probs, ae_ratio, flow_max, flow_entropy, ae_flag])
    assert feats.shape[1] == n_classes + 4
    return feats


class StackingMeta:
    """Тонкая обёртка над sklearn-классификатором + знание о классах/пороге.

    Сам классификатор — градиентный бустинг: он хорошо ловит нелинейные «правила
    доверия» (типа «если Flow уверен И AE молчит — это класс X»), которых линейная
    модель не увидит. Для нашего небольшого набора признаков он быстрый.
    """

    def __init__(self, classifier, classes: list[str], ae_threshold: float):
        self.classifier = classifier
        self.classes = list(classes)          # имена классов в порядке их id
        self.ae_threshold = ae_threshold

    @classmethod
    def fit(cls, meta_features: np.ndarray, labels: np.ndarray,
            classes: list[str], ae_threshold: float) -> "StackingMeta":
        # random_state фиксируем, чтобы прогон был воспроизводимым
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                              random_state=0)
        clf.fit(meta_features, labels)
        return cls(clf, classes, ae_threshold)

    def predict_proba(self, meta_features: np.ndarray) -> np.ndarray:
        """Калиброванные вероятности классов от стэкинга — [N, n_classes]."""
        return self.classifier.predict_proba(meta_features)

    def save(self, path: str | Path) -> Path:
        # sklearn-модель кладём рядом в .joblib, а метаданные — в json,
        # чтобы артефакт был самодостаточным и читаемым
        import joblib
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.classifier, path / "stacking.joblib")
        (path / "stacking_meta.json").write_text(
            json.dumps({"classes": self.classes, "ae_threshold": self.ae_threshold},
                       ensure_ascii=False),
            encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "StackingMeta":
        import joblib
        path = Path(path)
        clf = joblib.load(path / "stacking.joblib")
        meta = json.loads((path / "stacking_meta.json").read_text(encoding="utf-8"))
        return cls(clf, meta["classes"], meta["ae_threshold"])
