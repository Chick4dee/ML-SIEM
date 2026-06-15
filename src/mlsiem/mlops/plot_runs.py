"""Построение графиков обучения из истории MLflow → PNG в docs/figures/.

Кривые (loss/PR-AUC/recon по эпохам) и финальные per-class метрики сохраняются
как статические картинки для отчёта/портфолио (MLflow-данные локальны и в git не
идут). Запуск: python -m mlsiem.mlops.plot_runs
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402

from mlsiem.mlops.tracking import _repo_root, setup_tracking  # noqa: E402

OUT = _repo_root() / "docs" / "figures"


def _history(client, run_id, key):
    pts = sorted(client.get_metric_history(run_id, key), key=lambda m: m.step)
    return [m.step for m in pts], [m.value for m in pts]


def _run_id(runs, name):
    sub = runs[runs["tags.mlflow.runName"] == name]
    return sub.iloc[0]["run_id"] if len(sub) else None


def main() -> None:
    setup_tracking()
    OUT.mkdir(parents=True, exist_ok=True)
    client = mlflow.tracking.MlflowClient()
    runs = mlflow.search_runs(experiment_names=["mlsiem"], order_by=["start_time"])

    # 1. Flow (Э1): loss + val PR-AUC macro по эпохам
    rid = _run_id(runs, "flow_cnn_lstm")
    if rid:
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ep, loss = _history(client, rid, "train_loss")
        ax1.plot(ep, loss, "C3-o", label="train loss")
        ax1.set_xlabel("эпоха")
        ax1.set_ylabel("train loss", color="C3")
        ax2 = ax1.twinx()
        ep2, pr = _history(client, rid, "val_pr_auc_macro")
        ax2.plot(ep2, pr, "C0-s", label="val PR-AUC (macro)")
        ax2.set_ylabel("val PR-AUC (macro)", color="C0")
        plt.title("Эксперт 1 (Flow CNN+LSTM): обучение на UNSW")
        fig.tight_layout()
        fig.savefig(OUT / "flow_training.png", dpi=120)
        plt.close(fig)

        # per-class PR-AUC (лучшая = эпоха с макс macro)
        classes = [c.replace("metrics.val_pr_auc_", "") for c in runs.columns
                   if c.startswith("metrics.val_pr_auc_") and "macro" not in c]
        vals = []
        for c in classes:
            _, v = _history(client, rid, f"val_pr_auc_{c}")
            vals.append((c, max(v) if v else 0.0))
        vals.sort(key=lambda x: -x[1])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh([c for c, _ in vals][::-1], [v for _, v in vals][::-1], color="C0")
        ax.set_xlabel("PR-AUC")
        ax.set_xlim(0, 1)
        plt.title("Эксперт 1: per-class PR-AUC (UNSW)")
        fig.tight_layout()
        fig.savefig(OUT / "flow_per_class.png", dpi=120)
        plt.close(fig)

    # 2. AE (Э4): recon train/val по эпохам
    rid = _run_id(runs, "autoencoder")
    if rid:
        fig, ax = plt.subplots(figsize=(7, 4))
        ep, tr = _history(client, rid, "train_recon")
        _, vl = _history(client, rid, "val_recon_benign")
        ax.plot(ep, tr, "C3-o", label="train recon")
        ax.plot(ep, vl, "C0-s", label="val recon (benign)")
        ax.set_xlabel("эпоха")
        ax.set_ylabel("ошибка реконструкции")
        ax.legend()
        plt.title("Эксперт 4 (Autoencoder): сходимость на benign")
        fig.tight_layout()
        fig.savefig(OUT / "ae_training.png", dpi=120)
        plt.close(fig)

    # 3. Payload (Э3): PR-AUC/recall/precision по эпохам
    rid = _run_id(runs, "payload_charcnn")
    if rid:
        fig, ax = plt.subplots(figsize=(7, 4))
        for key, style in [("val_pr_auc", "C0-s"), ("val_recall", "C2-^"),
                           ("val_precision", "C1-v")]:
            ep, v = _history(client, rid, key)
            ax.plot(ep, v, style, label=key.replace("val_", ""))
        ax.set_xlabel("эпоха")
        ax.set_ylabel("метрика")
        ax.legend()
        ax.set_ylim(0.9, 1.0)
        plt.title("Эксперт 3 (Payload char-CNN): обучение на CSIC")
        fig.tight_layout()
        fig.savefig(OUT / "payload_training.png", dpi=120)
        plt.close(fig)

    print("графики сохранены в", OUT)
    for p in sorted(OUT.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
