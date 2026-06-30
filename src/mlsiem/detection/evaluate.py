"""Оценка MVP-детекции: мета (Flow+AE) vs одиночные эксперты (PLAN.md, фаза 4).

Прогоняем оба эксперта на held-out, калибруем AE-порог на отдельной части
(не на тесте), применяем мета-правила и сравниваем способность детектировать
атаку (attack vs benign) у меты против Flow-only и AE-only.

⚠️ Числа на UNSW несут оговорку фазы 3 (benign/attack сгенерированы разными
инструментами → AE/мета выглядят лучше, чем будут на реальном полигоне). Здесь
проверяется МЕХАНИЗМ объединения, не финальное качество (оно — фаза 7/21).
"""

import argparse
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from mlsiem.detection.ensemble import ExpertEnsemble
from mlsiem.detection.meta import BENIGN, combine_verdicts_batch
from mlsiem.experts.data_builder import build_datasets


def _collect(ensemble: ExpertEnsemble, loader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flow_p, ae_s, ys = [], [], []
    for x, mask, y in loader:
        out = ensemble.infer(x, mask)
        flow_p.append(out.flow_probs)
        ae_s.append(out.ae_scores)
        ys.append(y.numpy())
    return np.concatenate(flow_p), np.concatenate(ae_s), np.concatenate(ys)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--flow-dir", default="mlartifacts/flow_cnn_lstm")
    p.add_argument("--ae-dir", default="mlartifacts/autoencoder")
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument("--tau", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_datasets(args.data, window=16, val_frac=0.2)
    ensemble = ExpertEnsemble.load(args.flow_dir, args.ae_dir, device=device)
    benign_id = data.encoder.encode(BENIGN)

    loader = DataLoader(data.val, batch_size=256, shuffle=False)
    flow_probs, ae_scores, labels = _collect(ensemble, loader)

    # калибровка AE-порога на КАЛИБ-части (первая половина), тест — вторая
    n = len(labels)
    half = n // 2
    calib_benign = ae_scores[:half][labels[:half] == benign_id]
    thr = float(np.quantile(calib_benign, 1 - args.target_fpr))

    te = slice(half, n)
    fp, ae_te, y_te = flow_probs[te], ae_scores[te], labels[te]
    is_attack = y_te != benign_id

    # три детектора «атака да/нет»
    flow_pred = np.argmax(fp, axis=1) != benign_id            # Flow-only
    ae_pred = ae_te > thr                                      # AE-only
    verdicts = combine_verdicts_batch(fp, ensemble.classes, ae_te, thr, tau=args.tau)
    meta_pred = np.array([v.cls != BENIGN for v in verdicts])  # мета

    def stats(pred):
        tp = int((pred & is_attack).sum())
        fp_ = int((pred & ~is_attack).sum())
        fn = int((~pred & is_attack).sum())
        tn = int((~pred & ~is_attack).sum())
        recall = tp / (tp + fn) if tp + fn else 0.0
        fpr = fp_ / (fp_ + tn) if fp_ + tn else 0.0
        prec = tp / (tp + fp_) if tp + fp_ else 0.0
        return recall, fpr, prec

    print(f"тестовых окон: {len(y_te)} (атак {int(is_attack.sum())}), порог AE={thr:.4f}\n")
    print(f"{'детектор':<12}{'recall':>9}{'FPR':>9}{'precision':>11}")
    for name, pred in [("Flow-only", flow_pred), ("AE-only", ae_pred), ("МЕТА", meta_pred)]:
        r, f, pr = stats(pred)
        print(f"{name:<12}{r:>9.3f}{f:>9.3f}{pr:>11.3f}")

    # что мета спасла там, где Flow промолчал (open-set вклад AE)
    rescued = int((meta_pred & ~flow_pred & is_attack).sum())
    print(f"\nмета поймала атак, которые Flow пропустил: {rescued}")
    sev = Counter(v.severity for v in verdicts if v.cls != BENIGN)
    print("распределение severity по тревогам:", dict(sev))


if __name__ == "__main__":
    main()
