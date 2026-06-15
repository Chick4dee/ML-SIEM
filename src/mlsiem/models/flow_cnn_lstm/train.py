"""Обучение Эксперта 1 (Flow CNN+LSTM) на UNSW-NB15 с трекингом в MLflow.

Запуск (полное обучение):
    python -m mlsiem.models.flow_cnn_lstm.train --data "data/processed/unsw/*.parquet"

Быстрый smoke (несколько файлов, 2 эпохи):
    python -m mlsiem.models.flow_cnn_lstm.train --data "data/processed/unsw/17-2-2015__1.parquet" --epochs 2
"""  # noqa: E501

import argparse
import math
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics.classification import MulticlassAveragePrecision

from mlsiem.mlops.tracking import setup_tracking
from mlsiem.models.data_builder import build_datasets, save_artifacts
from mlsiem.models.flow_cnn_lstm.model import FlowCNNLSTM, FlowModelConfig
from mlsiem.models.losses import FocalLoss, inverse_frequency_alpha


def _emb_dim(card: int) -> int:
    return min(16, max(2, round(math.sqrt(card))))


def _evaluate(model, loader, n_classes, device) -> tuple[float, list[float]]:
    metric = MulticlassAveragePrecision(num_classes=n_classes, average=None).to(device)
    model.eval()
    with torch.no_grad():
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x, mask)
            metric.update(logits.float().softmax(dim=1), y)
    per_class = metric.compute().cpu().tolist()
    # классы без позитивов в val дают nan AP — усредняем только по присутствующим
    present = [ap for ap in per_class if not math.isnan(ap)]
    macro = sum(present) / len(present) if present else float("nan")
    return macro, per_class


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_datasets(args.data, window=args.window, val_frac=args.val_frac)
    n_cat = len(data.cat_positions)
    n_numeric = len(data.feature_names) - n_cat
    cfg = FlowModelConfig(
        n_numeric=n_numeric,
        n_classes=len(data.encoder),
        cat_cardinalities=data.cat_cardinalities,
        cat_emb_dims=[_emb_dim(c) for c in data.cat_cardinalities],
        cat_positions=data.cat_positions,
    )
    model = FlowCNNLSTM(cfg).to(device)

    # Балансировка дисбаланса: либо сэмплер (мягкий, ∝1/√частота на КЛАСС), либо
    # веса в focal (alpha). Вместе пересолят редкие классы, поэтому при balanced
    # alpha выключаем — балансом занимается сэмплер.
    if args.balanced:
        labels = data.train.window_labels
        class_w = 1.0 / np.sqrt(np.bincount(labels, minlength=len(data.encoder)).clip(min=1))
        sample_w = class_w[labels]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(labels), replacement=True)
        criterion = FocalLoss(gamma=args.gamma, alpha=None)
        train_loader = DataLoader(data.train, batch_size=args.batch, sampler=sampler,
                                  num_workers=args.workers, pin_memory=True,
                                  persistent_workers=args.workers > 0, drop_last=True)
    else:
        alpha = inverse_frequency_alpha(data.class_counts).to(device)
        criterion = FocalLoss(gamma=args.gamma, alpha=alpha)
        train_loader = DataLoader(data.train, batch_size=args.batch, shuffle=True,
                                  num_workers=args.workers, pin_memory=True,
                                  persistent_workers=args.workers > 0, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    val_loader = DataLoader(data.val, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0)

    setup_tracking()
    out_dir = Path(args.out)
    with mlflow.start_run(run_name="flow_cnn_lstm"):
        mlflow.log_params({
            "n_numeric": n_numeric, "n_classes": len(data.encoder),
            "window": args.window, "batch": args.batch, "lr": args.lr,
            "gamma": args.gamma, "epochs": args.epochs, "balanced": args.balanced,
            "train_windows": len(data.train), "val_windows": len(data.val),
        })
        save_artifacts(data, out_dir)

        best_macro, best_epoch, patience = -1.0, -1, args.patience
        for epoch in range(args.epochs):
            model.train()
            running = 0.0
            for x, mask, y in train_loader:
                x, mask, y = x.to(device), mask.to(device), y.to(device)
                optimizer.zero_grad()
                with torch.autocast("cuda", enabled=device.type == "cuda"):
                    loss = criterion(model(x, mask), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running += loss.item()
            train_loss = running / max(len(train_loader), 1)

            macro, per_class = _evaluate(model, val_loader, len(data.encoder), device)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_pr_auc_macro", macro, step=epoch)
            for cls, ap in zip(data.encoder.classes, per_class, strict=True):
                if not math.isnan(ap):
                    mlflow.log_metric(f"val_pr_auc_{cls}", ap, step=epoch)
            print(f"epoch {epoch}: train_loss={train_loss:.4f} "
                  f"val_PR-AUC(macro)={macro:.4f}", flush=True)

            if macro > best_macro:
                best_macro, best_epoch = macro, epoch
                torch.save(model.state_dict(), out_dir / "flow_cnn_lstm.pt")
                patience = args.patience
            else:
                patience -= 1
                if patience <= 0:
                    print(f"early stop на эпохе {epoch}", flush=True)
                    break

        mlflow.log_metric("best_val_pr_auc_macro", best_macro)
        mlflow.log_metric("best_epoch", best_epoch)
        print(f"\nЛучший val PR-AUC (macro): {best_macro:.4f} (эпоха {best_epoch})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="glob по parquet")
    p.add_argument("--out", default="mlartifacts/flow_cnn_lstm")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--workers", type=int, default=4, help="воркеры DataLoader")
    p.add_argument("--balanced", action="store_true", default=True,
                   help="class-balanced sampling (по умолчанию вкл)")
    p.add_argument("--no-balanced", dest="balanced", action="store_false")
    train(p.parse_args())


if __name__ == "__main__":
    main()
