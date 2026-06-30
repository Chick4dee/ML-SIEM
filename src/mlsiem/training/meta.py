"""Обучение stacking-меты (фаза 7) и честное сравнение с наивным OR из фазы 4.

Идея прогона:
  1. Берём УЖЕ обученные эксперты (Flow + AE) и гоняем их по hold-out части UNSW
     (val) — данным, которых эксперты не видели при обучении. Так выходы экспертов
     «честные», без переуверенности на своих тренировочных примерах (контракт #10.5).
  2. Эту hold-out выборку делим ещё раз: на чём учить стэкинг (meta-train) и на чём
     его потом мерить (meta-test). Стэкинг не должен видеть свой тест.
  3. Учим маленький классификатор поверх выходов экспертов и сравниваем три
     «детектора атаки» на одном и том же тесте: только Flow, наивный OR, стэкинг.
     Ждём, что стэкинг поднимет precision/снизит FPR относительно OR, не растеряв
     recall — ровно то, что наивный OR не умел.

Запуск:
    python -m mlsiem.training.meta --data "data/processed/unsw/*.parquet"
"""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from mlsiem.detection.ensemble import ExpertEnsemble
from mlsiem.detection.meta import BENIGN, combine_verdicts_batch
from mlsiem.detection.stacking import StackingMeta, build_meta_features
from mlsiem.experts.data_builder import build_datasets
from mlsiem.mlops.tracking import setup_tracking


def _run_experts(ensemble: ExpertEnsemble, loader):
    """Прогоняет оба эксперта по всем окнам и собирает их выходы в один массив."""
    flow_p, ae_s, ys = [], [], []
    for x, mask, y in loader:
        out = ensemble.infer(x, mask)
        flow_p.append(out.flow_probs)
        ae_s.append(out.ae_scores)
        ys.append(y.numpy())
    return np.concatenate(flow_p), np.concatenate(ae_s), np.concatenate(ys)


def _detection_stats(pred_is_attack: np.ndarray, true_is_attack: np.ndarray) -> dict:
    """recall / FPR / precision для бинарной задачи «атака или нет»."""
    tp = int((pred_is_attack & true_is_attack).sum())
    fp = int((pred_is_attack & ~true_is_attack).sum())
    fn = int((~pred_is_attack & true_is_attack).sum())
    tn = int((~pred_is_attack & ~true_is_attack).sum())
    return {
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
    }


def _macro_pr_auc(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    """Среднее PR-AUC по классам (one-vs-rest). Классы, которых нет в тесте,
    пропускаем — по ним метрику не посчитать."""
    aucs = []
    for c in range(n_classes):
        present = labels == c
        if present.any() and (~present).any():
            aucs.append(average_precision_score(present.astype(int), probs[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--flow-dir", default="mlartifacts/flow_cnn_lstm")
    p.add_argument("--ae-dir", default="mlartifacts/autoencoder")
    p.add_argument("--out", default="mlartifacts/stacking")
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument("--meta-train-frac", type=float, default=0.6,
                   help="доля hold-out под обучение стэкинга (остальное — тест)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_datasets(args.data, window=16, val_frac=0.2)
    ensemble = ExpertEnsemble.load(args.flow_dir, args.ae_dir, device=device)
    benign_id = data.encoder.encode(BENIGN)
    n_classes = len(data.encoder)

    # Шаг 1: выходы экспертов на hold-out (val — эксперты его не видели).
    loader = DataLoader(data.val, batch_size=256, shuffle=False)
    flow_probs, ae_scores, labels = _run_experts(ensemble, loader)

    # Шаг 2: делим hold-out на «учить стэкинг» и «мерить стэкинг».
    n = len(labels)
    cut = int(n * args.meta_train_frac)
    tr = slice(0, cut)
    te = slice(cut, n)

    # Порог AE калибруем на benign из meta-train (а не из теста!) под целевой FPR.
    tr_benign = ae_scores[tr][labels[tr] == benign_id]
    ae_threshold = float(np.quantile(tr_benign, 1 - args.target_fpr))

    # Шаг 3: признаки для стэкинга из выходов экспертов.
    feats_tr = build_meta_features(flow_probs[tr], ae_scores[tr], ae_threshold)
    feats_te = build_meta_features(flow_probs[te], ae_scores[te], ae_threshold)

    stacking = StackingMeta.fit(feats_tr, labels[tr], data.encoder.classes, ae_threshold)

    # --- Оценка на meta-test: три детектора на одних и тех же данных ---
    true_is_attack = labels[te] != benign_id

    # (а) только Flow: берём его argmax
    flow_pred = flow_probs[te].argmax(axis=1) != benign_id
    # (б) наивный OR из фазы 4
    verdicts = combine_verdicts_batch(flow_probs[te], ensemble.classes,
                                      ae_scores[te], ae_threshold)
    or_pred = np.array([v.cls != BENIGN for v in verdicts])
    # (в) стэкинг: его предсказанный класс
    stack_proba = stacking.predict_proba(feats_te)
    stack_pred = stack_proba.argmax(axis=1) != benign_id

    setup_tracking()
    out_dir = Path(args.out)
    with mlflow.start_run(run_name="stacking_meta"):
        mlflow.log_params({"n_classes": n_classes, "ae_threshold": ae_threshold,
                           "meta_train": int(cut), "meta_test": int(n - cut)})
        stacking.save(out_dir)

        print(f"meta-test окон: {len(true_is_attack)} "
              f"(атак {int(true_is_attack.sum())}), порог AE={ae_threshold:.4f}\n")
        print(f"{'детектор':<12}{'recall':>9}{'FPR':>9}{'precision':>11}")
        for name, pred in [("Flow-only", flow_pred), ("наивный OR", or_pred),
                           ("STACKING", stack_pred)]:
            s = _detection_stats(pred, true_is_attack)
            mlflow.log_metrics({f"{name}_recall": s["recall"], f"{name}_fpr": s["fpr"],
                                f"{name}_precision": s["precision"]})
            print(f"{name:<12}{s['recall']:>9.3f}{s['fpr']:>9.3f}{s['precision']:>11.3f}")

        # многоклассовое качество: стэкинг против одного Flow
        flow_macro = _macro_pr_auc(flow_probs[te], labels[te], n_classes)
        stack_macro = _macro_pr_auc(stack_proba, labels[te], n_classes)
        mlflow.log_metrics({"flow_macro_pr_auc": flow_macro,
                            "stacking_macro_pr_auc": stack_macro})
        print(f"\nmulticlass macro PR-AUC:  Flow={flow_macro:.4f}  "
              f"STACKING={stack_macro:.4f}")


if __name__ == "__main__":
    main()
