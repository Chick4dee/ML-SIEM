"""Мета-слой v1 на ПРАВИЛАХ (PLAN.md, раздел 4, мета-слой; stacking — фаза 7).

Объединяет вердикт Flow-эксперта (класс + вероятности) и сигнал автоэнкодера
(аномалия) в единый структурированный вердикт {class, severity, confidence}.

Логика (детерминированная, объяснимая):
- Flow уверен в атаке (max-prob ≥ τ) → берём его класс. Если AE тоже флагует —
  это corroboration: повышаем confidence (сеть+аномалия на одном окне).
- Flow НЕ уверен (или говорит benign), но AE high → **open-set / UNKNOWN**
  (zero-day страховка: supervised не знает, аномалия есть).
- Иначе → benign.

severity-движок: класс → критичность (для приоритизации в SOAR).
"""

from dataclasses import dataclass

import numpy as np

BENIGN = "benign"
UNKNOWN = "UNKNOWN"

# Критичность класса (PLAN.md: severity-движок). UNKNOWN — средняя (неизвестность).
SEVERITY_BY_CLASS: dict[str, str] = {
    "benign": "none",
    "reconnaissance": "low",
    "analysis": "low",
    "fuzzers": "low",
    "generic": "medium",
    "dos": "high",
    "exploits": "high",
    "shellcode": "high",
    "backdoor": "critical",
    "worms": "critical",
    "botnet": "critical",
    UNKNOWN: "medium",
}
_DEFAULT_SEVERITY = "medium"


@dataclass
class Verdict:
    cls: str
    severity: str
    confidence: float
    sources: tuple[str, ...]   # кто дал вклад: ("flow",), ("ae",), ("flow","ae")


def severity_of(cls: str) -> str:
    return SEVERITY_BY_CLASS.get(cls, _DEFAULT_SEVERITY)


def _anomaly_confidence(ae_score: float, threshold: float) -> float:
    """Уверенность аномалии: 0.5 на пороге, растёт с величиной превышения."""
    excess = max(0.0, (ae_score - threshold) / max(threshold, 1e-9))
    return float(min(1.0, 0.5 + 0.5 * excess))


def combine_verdict(
    flow_probs: np.ndarray,
    classes: list[str],
    ae_score: float,
    ae_threshold: float,
    *,
    tau: float = 0.5,
    corroboration_bonus: float = 0.1,
) -> Verdict:
    """Единый вердикт по одному окну из выходов Flow и AE."""
    flow_idx = int(np.argmax(flow_probs))
    flow_cls = classes[flow_idx]
    flow_conf = float(flow_probs[flow_idx])
    ae_anom = ae_score > ae_threshold

    if flow_cls != BENIGN and flow_conf >= tau:
        # supervised уверен в атаке
        conf = flow_conf
        sources = ("flow",)
        if ae_anom:  # corroboration сеть+аномалия → выше уверенность
            conf = min(1.0, conf + corroboration_bonus)
            sources = ("flow", "ae")
        return Verdict(flow_cls, severity_of(flow_cls), conf, sources)

    if ae_anom:
        # flow не уверен / benign, а аномалия есть → open-set UNKNOWN
        return Verdict(UNKNOWN, severity_of(UNKNOWN),
                       _anomaly_confidence(ae_score, ae_threshold), ("ae",))

    # никто не флагует
    benign_conf = flow_conf if flow_cls == BENIGN else 1.0 - flow_conf
    return Verdict(BENIGN, severity_of(BENIGN), benign_conf, ("flow",))


def combine_verdicts_batch(
    flow_probs: np.ndarray,
    classes: list[str],
    ae_scores: np.ndarray,
    ae_threshold: float,
    *,
    tau: float = 0.5,
) -> list[Verdict]:
    """Векторный вход [B,C] и [B] → список вердиктов (для оценки)."""
    return [
        combine_verdict(flow_probs[i], classes, float(ae_scores[i]), ae_threshold, tau=tau)
        for i in range(len(ae_scores))
    ]
