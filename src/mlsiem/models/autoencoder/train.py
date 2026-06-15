"""Обучение Эксперта 4 (Conv1D AE) на benign + калибровка порога по FPR.

Запуск:
    python -m mlsiem.models.autoencoder.train --data "data/processed/unsw/*.parquet"

AE учится ТОЛЬКО на benign-окнах. Порог берём по целевому FPR на benign-val
(НЕ «70-й перцентиль» — это давало ~30% ложных в v1). Оценка: насколько
ошибка реконструкции отделяет атаки от нормы (ROC-AUC/PR-AUC) и какой recall по
классам при целевом FPR — ожидаем сильный сигнал по dos/волюметрике.

AE считаем в fp32: под AMP большая ошибка реконструкции переполняет fp16 в inf,
а inf×0 на паддинг-позициях даёт nan. Модель крошечная — выигрыш AMP ничтожен.
"""

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

from mlsiem.mlops.tracking import setup_tracking
from mlsiem.models.autoencoder.model import AEConfig, Conv1DAutoencoder, reconstruction_error
from mlsiem.models.data_builder import build_datasets, save_artifacts


def _recon_errors(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (ошибки реконструкции, метки) по всему загрузчику."""
    model.eval()
    errs, labels = [], []
    with torch.no_grad():
        for x, mask, y in loader:
            x, mask = x.to(device), mask.to(device)
            e = reconstruction_error(model(x), x, mask)
            errs.append(e.float().cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(errs), np.concatenate(labels)


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_datasets(args.data, window=args.window, val_frac=args.val_frac)
    benign_id = data.encoder.encode("benign")
    n_features = len(data.feature_names)

    # train ТОЛЬКО на benign; для скорости подвыборка (benign почти идентичны)
    benign_idx = np.flatnonzero(data.train.window_labels == benign_id)
    rng = np.random.default_rng(0)
    if args.max_train and len(benign_idx) > args.max_train:
        benign_idx = rng.choice(benign_idx, args.max_train, replace=False)
    train_ds = Subset(data.train, benign_idx.tolist())

    model = Conv1DAutoencoder(AEConfig(n_features=n_features, window=args.window)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(data.val, batch_size=args.batch, shuffle=False)

    setup_tracking()
    out_dir = Path(args.out)
    with mlflow.start_run(run_name="autoencoder"):
        mlflow.log_params({
            "n_features": n_features, "window": args.window, "batch": args.batch,
            "lr": args.lr, "epochs": args.epochs, "target_fpr": args.target_fpr,
            "latent_size": model.cfg.latent_size, "train_benign_windows": len(train_ds),
        })
        save_artifacts(data, out_dir)

        best_recon, patience = float("inf"), args.patience
        for epoch in range(args.epochs):
            model.train()
            running = 0.0
            for x, mask, _ in train_loader:
                x, mask = x.to(device), mask.to(device)
                optimizer.zero_grad()
                loss = reconstruction_error(model(x), x, mask).mean()
                loss.backward()
                optimizer.step()
                running += loss.item()
            train_recon = running / max(len(train_loader), 1)

            # val_recon ТОЛЬКО по benign (на чём учились) — критерий early stop
            errs, labels = _recon_errors(model, val_loader, device)
            benign_mask = labels == benign_id
            val_recon = float(errs[benign_mask].mean())
            mlflow.log_metric("train_recon", train_recon, step=epoch)
            mlflow.log_metric("val_recon_benign", val_recon, step=epoch)
            print(f"epoch {epoch}: train_recon={train_recon:.5f} "
                  f"val_recon(benign)={val_recon:.5f}", flush=True)

            if val_recon < best_recon - 1e-6:
                best_recon = val_recon
                torch.save(model.state_dict(), out_dir / "autoencoder.pt")
                patience = args.patience
            else:
                patience -= 1
                if patience <= 0:
                    print(f"early stop на эпохе {epoch}", flush=True)
                    break

        # Финальная оценка лучшей модели
        model.load_state_dict(torch.load(out_dir / "autoencoder.pt"))
        errs, labels = _recon_errors(model, val_loader, device)
        benign_mask = labels == benign_id
        is_attack = (~benign_mask).astype(int)

        roc = roc_auc_score(is_attack, errs)
        pr = average_precision_score(is_attack, errs)
        thr = float(np.quantile(errs[benign_mask], 1 - args.target_fpr))
        actual_fpr = float((errs[benign_mask] > thr).mean())
        # порог — часть serve-артефакта эксперта (нужен мета-слою)
        (out_dir / "ae_threshold.json").write_text(
            json.dumps({"threshold": thr, "target_fpr": args.target_fpr,
                        "roc_auc": roc, "pr_auc": pr}, ensure_ascii=False),
            encoding="utf-8")
        mlflow.log_metrics({"anomaly_roc_auc": roc, "anomaly_pr_auc": pr,
                            "threshold": thr, "actual_fpr": actual_fpr})
        print(f"\nАномалия-детекция: ROC-AUC={roc:.3f} PR-AUC={pr:.3f} "
              f"| порог@FPR={args.target_fpr}: {thr:.5f} (факт FPR {actual_fpr:.3f})")

        print("recall по классам при целевом FPR:")
        for cls in data.encoder.classes:
            if cls == "benign":
                continue
            cid = data.encoder.encode(cls)
            m = labels == cid
            if m.sum() == 0:
                continue
            recall = float((errs[m] > thr).mean())
            mlflow.log_metric(f"recall_{cls}", recall)
            print(f"  {cls:<16} {recall:.3f}  (n={int(m.sum())})", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="mlartifacts/autoencoder")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument("--max-train", type=int, default=200_000,
                   help="подвыборка benign-окон для скорости (0 = все)")
    train(p.parse_args())


if __name__ == "__main__":
    main()
