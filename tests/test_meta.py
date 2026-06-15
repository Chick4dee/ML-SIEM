"""Тесты мета-слоя на правилах: все ветки объединения Flow+AE."""

import numpy as np

from mlsiem.detection.meta import (
    BENIGN,
    UNKNOWN,
    Verdict,
    combine_verdict,
    severity_of,
)

CLASSES = ["benign", "dos", "exploits", "worms"]


def _probs(d: dict[str, float]) -> np.ndarray:
    return np.array([d.get(c, 0.0) for c in CLASSES], dtype=np.float64)


def test_flow_confident_attack_wins():
    v = combine_verdict(_probs({"exploits": 0.9, "benign": 0.1}), CLASSES,
                        ae_score=0.0, ae_threshold=1.0)
    assert v.cls == "exploits"
    assert v.severity == "high"
    assert v.sources == ("flow",)
    assert v.confidence == 0.9


def test_corroboration_raises_confidence():
    # flow уверен в атаке И AE флагует → confidence выше, источник flow+ae
    base = combine_verdict(_probs({"dos": 0.8, "benign": 0.2}), CLASSES, 0.0, 1.0)
    corr = combine_verdict(_probs({"dos": 0.8, "benign": 0.2}), CLASSES,
                           ae_score=5.0, ae_threshold=1.0)
    assert corr.cls == "dos"
    assert corr.sources == ("flow", "ae")
    assert corr.confidence > base.confidence


def test_open_set_unknown_when_flow_unsure_and_ae_high():
    # flow ни в чём не уверен (max-prob < τ), AE флагует → UNKNOWN (zero-day)
    v = combine_verdict(_probs({"benign": 0.4, "dos": 0.35, "exploits": 0.25}),
                        CLASSES, ae_score=3.0, ae_threshold=1.0, tau=0.5)
    assert v.cls == UNKNOWN
    assert v.severity == "medium"
    assert v.sources == ("ae",)


def test_ae_rescues_flow_blind_spot():
    # flow уверенно говорит benign, но AE видит аномалию → UNKNOWN (страховка)
    v = combine_verdict(_probs({"benign": 0.95, "dos": 0.05}), CLASSES,
                        ae_score=10.0, ae_threshold=1.0)
    assert v.cls == UNKNOWN
    assert v.sources == ("ae",)


def test_benign_when_nobody_flags():
    v = combine_verdict(_probs({"benign": 0.97, "dos": 0.03}), CLASSES,
                        ae_score=0.2, ae_threshold=1.0)
    assert v.cls == BENIGN
    assert v.severity == "none"
    assert v.sources == ("flow",)


def test_low_confidence_attack_not_taken_without_ae():
    # flow склоняется к атаке, но max-prob < τ и AE молчит → benign (не флагуем)
    v = combine_verdict(_probs({"dos": 0.45, "benign": 0.4, "exploits": 0.15}),
                        CLASSES, ae_score=0.1, ae_threshold=1.0, tau=0.5)
    assert v.cls == BENIGN


def test_anomaly_confidence_grows_with_magnitude():
    weak = combine_verdict(_probs({"benign": 0.9}), CLASSES, 1.1, 1.0)
    strong = combine_verdict(_probs({"benign": 0.9}), CLASSES, 100.0, 1.0)
    assert 0.5 <= weak.confidence < strong.confidence <= 1.0


def test_severity_map():
    assert severity_of("worms") == "critical"
    assert severity_of("reconnaissance") == "low"
    assert severity_of("benign") == "none"
    assert severity_of("totally_new_class") == "medium"  # дефолт


def test_verdict_is_dataclass():
    v = combine_verdict(_probs({"exploits": 0.9}), CLASSES, 0.0, 1.0)
    assert isinstance(v, Verdict)
