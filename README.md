# ML-SIEM

Гибридный SIEM/SOAR с детекцией сетевых и хостовых атак на ансамбле самостоятельно
обученных нейросетей (Mixture-of-Experts) и детерминированным движком реагирования.

> Учебный portfolio-проект, строится «production-grade»: ML + Backend + Frontend + Data + DevOps + Security.

## Идея в двух строках

- **Нейросети** (PyTorch: CNN+LSTM, Autoencoder, char-CNN) отвечают на «*что* происходит» — вероятностно.
- **SOAR** (правила, playbooks, MITRE ATT&CK) отвечает на «*что делать*» — детерминированно, объяснимо, обратимо,
  с подтверждением аналитика (human-in-the-loop) через SOC-консоль на React.

## Архитектура

```
Kafka (ingestion) → nfstream-фичи + windowing → 5 нейро-экспертов → мета-слой (stacking, open-set)
    → корреляция сеть↔хост → SOAR (симуляция действий, Approve/Reject) →
    → PostgreSQL + OpenSearch → FastAPI + WebSocket → React SOC-консоль (8 экранов)
```

Сквозное: MLflow, DVC, ONNX (опц.), Prometheus + Grafana, Docker Compose, GitHub Actions.
Демо-стенд: изолированный киберполигон на Hyper-V (Kali + Caldera против одноразовой
Windows-жертвы, GHOSTS как benign-фон), SIEM — наблюдатель на хосте с GPU.

## Статус

🚧 **Фазы 0–6 пройдены** (Э2/Temporal отложен): данные, 3 эксперта (Flow + AE +
Payload), мета-слой, MVP-детект. Подробности — [docs/JOURNAL.md](docs/JOURNAL.md).

Полный план проекта (архитектура, данные, риски, дорожная карта, журнал решений):
[docs/PLAN.md](docs/PLAN.md); карта покрытия «класс → эксперт» — [docs/COVERAGE.md](docs/COVERAGE.md).

## Структура `src/mlsiem/`

```
data/            пайплайн данных: PCAP → фичи → окна
  extractor.py     PCAP → потоки (nfstream)
  windowing.py     потоки → маскированные окна
  preprocessing.py нормализация/кодирование фич
  labeling.py      перенос ground-truth меток
  taxonomy.py      единый словарь классов
  sources/         разбор датасетов: ctu13.py, unsw.py, csic.py
experts/         архитектуры моделей + общие части
  flow.py          Эксперт 1 (Flow CNN+LSTM)
  autoencoder.py   Эксперт 4 (Conv1D AE)
  payload.py       Эксперт 3 (char-CNN)
  windows_dataset.py / payload_dataset.py  torch-датасеты
  losses.py        focal loss
  data_builder.py  сборка train/val
detection/       объединение экспертов в вердикт
  meta.py          правила {class, severity, confidence}
  ensemble.py      загрузка + инференс экспертов
  evaluate.py      оценка MVP-детекта
training/        ⮕ ВСЁ, что запускается для обучения
  flow.py  autoencoder.py  payload.py
mlops/           tracking.py (MLflow), plot_runs.py (графики)
ingestion/ storage/ api/ response/   — под будущие фазы
```

## Установка

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev,ml,capture]
# torch под CUDA отдельно (Windows):
pip install torch --index-url https://download.pytorch.org/whl/cu126
pytest
```

## Запуск обучения

```powershell
# Эксперт 1 (Flow) на UNSW
python -m mlsiem.training.flow --data "data/processed/unsw/*.parquet" --epochs 25

# Эксперт 4 (Autoencoder) на benign UNSW
python -m mlsiem.training.autoencoder --data "data/processed/unsw/*.parquet"

# Эксперт 3 (Payload) на CSIC
python -m mlsiem.training.payload --data "A:/datasets/CSIC-2010"

# Оценка MVP-детекта (мета Flow+AE)
python -m mlsiem.detection.evaluate --data "data/processed/unsw/*.parquet"

# Графики обучения из MLflow → docs/figures/
python -m mlsiem.mlops.plot_runs
```

Метрики прогонов — в MLflow: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Подготовка данных (из сырых PCAP)

```powershell
# CTU-13: все сценарии → размеченные parquet
python -m mlsiem.data.sources.ctu13 --dataset "A:/datasets/CTU-13/CTU-13-Dataset" --output data/processed/ctu13
# UNSW-NB15: все pcap → размеченные parquet
python -m mlsiem.data.sources.unsw --dataset "A:/datasets/UNSW-NB15/OneDrive_2_12.06.2026" --output data/processed/unsw
```

## Данные (DVC)

Сырые датасеты и обработанные parquet версионируются через **DVC**, в git хранятся
только `.dvc`-указатели. Remote — **локальный** (объёмы большие: сырьё ~100 ГБ),
поэтому данные не лежат в публичном репозитории.

```powershell
# обработанные данные (если remote доступен на этой машине)
dvc pull
```

Remote по умолчанию настроен на локальный путь (`A:\dvc-remote`). На другой машине
переопредели его под свой носитель:

```powershell
dvc remote modify --local localremote url <путь-к-твоему-remote>
```

Сырьё (PCAP UNSW-NB15 / CTU-13 / CSIC-2010) скачивается отдельно — источники см.
в [docs/JOURNAL.md](docs/JOURNAL.md).
