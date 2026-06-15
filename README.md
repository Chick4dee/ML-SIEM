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

🚧 **Фаза 0 из 22** — каркас, окружение, проверка GPU/CUDA, smoke-тесты ключевых зависимостей.

Полный план проекта (архитектура, данные, риски, дорожная карта, журнал решений):
[docs/PLAN.md](docs/PLAN.md).

## Структура

```
src/mlsiem/      ingestion, features, models (5 экспертов + meta), detection, response, api, storage
ui/              React + TypeScript SOC-консоль
mlops/           MLflow, ONNX-экспорт
infra/           docker-compose, Prometheus, Grafana, OpenSearch
tests/           pytest
docs/            PLAN.md
data/            датасеты (DVC, в git не входят)
```

## Запуск (фаза 0)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
pytest
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
