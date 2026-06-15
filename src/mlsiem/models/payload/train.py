"""Обучение Эксперта 3 (char-CNN) на CSIC 2010.

Запуск:
    python -m mlsiem.models.payload.train --data "A:/datasets/CSIC-2010"

CSIC сбалансирован (normal 72k / anomalous 25k) — обычная бинарная классификация
содержимого запроса. Метрики: PR-AUC/recall/precision по классу атаки.
"""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_score, recall_score
from torch.utils.data import DataLoader

from mlsiem.features.csic import load_csic
from mlsiem.mlops.tracking import setup_tracking
from mlsiem.models.dataset import LabelEncoder
from mlsiem.models.payload.dataset import PayloadDataset
from mlsiem.models.payload.model import CharCNN, CharCNNConfig, CharTokenizer


def _split(df, val_frac, seed):
    # стратифицированный по метке случайный сплit (CSIC — не временной ряд)
    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - val_frac))
    return df[idx[:cut].tolist()], df[idx[cut:].tolist()]


@torch.no_grad()
def _evaluate(model, loader, device, attack_id):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        logits = model(x.to(device))
        probs.append(logits.softmax(1)[:, attack_id].cpu().numpy())
        ys.append(y.numpy())
    p = np.concatenate(probs)
    y = np.concatenate(ys)
    is_attack = (y == attack_id).astype(int)
    pred = (p >= 0.5).astype(int)
    return {
        "pr_auc": average_precision_score(is_attack, p),
        "recall": recall_score(is_attack, pred, zero_division=0),
        "precision": precision_score(is_attack, pred, zero_division=0),
    }


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_csic(args.data)
    train_df, val_df = _split(df, args.val_frac, args.seed)

    tokenizer = CharTokenizer.fit(train_df["text"].to_list(), max_len=args.max_len)
    encoder = LabelEncoder.fit(df["label"])
    attack_id = encoder.encode("anomalous")

    train_ds = PayloadDataset(train_df, tokenizer, encoder)
    val_ds = PayloadDataset(val_df, tokenizer, encoder)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    model = CharCNN(CharCNNConfig(vocab_size=tokenizer.vocab_size,
                                  n_classes=len(encoder))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    setup_tracking()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(run_name="payload_charcnn"):
        mlflow.log_params({"vocab": tokenizer.vocab_size, "max_len": args.max_len,
                           "batch": args.batch, "lr": args.lr, "epochs": args.epochs,
                           "train": len(train_ds), "val": len(val_ds)})
        tokenizer.save(out_dir / "tokenizer.json")
        encoder.save(out_dir / "label_encoder.json")

        best_pr, patience = -1.0, args.patience
        for epoch in range(args.epochs):
            model.train()
            running = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                running += loss.item()
            m = _evaluate(model, val_loader, device, attack_id)
            mlflow.log_metric("train_loss", running / max(len(train_loader), 1), step=epoch)
            for k, v in m.items():
                mlflow.log_metric(f"val_{k}", v, step=epoch)
            print(f"epoch {epoch}: loss={running / max(len(train_loader), 1):.4f} "
                  f"PR-AUC={m['pr_auc']:.4f} recall={m['recall']:.4f} "
                  f"precision={m['precision']:.4f}", flush=True)

            if m["pr_auc"] > best_pr:
                best_pr = m["pr_auc"]
                torch.save(model.state_dict(), out_dir / "payload_charcnn.pt")
                patience = args.patience
            else:
                patience -= 1
                if patience <= 0:
                    print(f"early stop на эпохе {epoch}", flush=True)
                    break
        mlflow.log_metric("best_val_pr_auc", best_pr)
        print(f"\nЛучший val PR-AUC: {best_pr:.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="папка CSIC-2010")
    p.add_argument("--out", default="mlartifacts/payload")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    train(p.parse_args())


if __name__ == "__main__":
    main()
