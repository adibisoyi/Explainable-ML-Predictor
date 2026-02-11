# Explainable ML Predictor

A production-minded end-to-end machine learning project that trains a tabular classifier, serves predictions with FastAPI, and explains individual predictions using SHAP.

## WHY this project exists

- Show the full ML lifecycle in a small codebase: train -> save artifacts -> serve -> explain.
- Demonstrate explainability in a practical API (`/explain`) rather than theory-only notebooks.
- Provide a clean baseline that already includes production launch guardrails.

## Production-ready additions in this branch

- API key authentication + role-based authorization (`predictor` and `admin`).
- Structured JSON logs with request tracing (`x-request-id` propagation).
- Online feature-drift monitor and `/monitoring/drift` status endpoint.
- CI quality gates for linting, typing, tests, and security scans.
- Reproducible deployment assets (`Dockerfile`, `docker-compose`, `infra/k8s`).
- Load-test scaffold and SLO definitions in `docs/SLO.md`.

## Quick install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Train a model

```bash
python -m exml.cli train --model logistic --out artifacts/
```

Training prints validation metrics and saves:

- `artifacts/pipeline.joblib`
- `artifacts/metadata.json` (includes baseline feature stats for drift checks)
- `artifacts/background.joblib`

## Run the API

```bash
python -m exml.cli serve
```

Default local keys:

- predictor key: `dev-predict-key`
- admin key: `dev-admin-key`

> Override keys with `EXML_API_KEYS` JSON env var. In production, set `EXML_ENV=prod` to disable built-in dev keys and require explicit key configuration.

Endpoints:

- `GET /health` (public)
- `POST /predict` (`predictor` or `admin`)
- `POST /explain` (`admin`)
- `GET /monitoring/drift` (`admin`)

## Build a valid payload quickly

```bash
python -m exml.cli sample-json > sample.json
```

## Call the API with curl

```bash
curl -s http://127.0.0.1:8000/health
```

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: dev-predict-key' \
  -H 'x-request-id: demo-123' \
  -d @sample.json
```

```bash
curl -s http://127.0.0.1:8000/monitoring/drift \
  -H 'x-api-key: dev-admin-key'
```

## CI quality gates (GitHub Actions)

Workflow: `.github/workflows/ci.yml`

- `ruff check src tests`
- `mypy src`
- `pytest -q`
- `bandit -r src`
- `pip-audit`

## Reproducible deployment

### Docker

```bash
docker build -t exml-api .
docker run --rm -p 8000:8000 exml-api
```

### Docker Compose

```bash
docker compose up --build
```

### Kubernetes manifests

- `infra/k8s/deployment.yaml`
- `infra/k8s/secret.example.yaml`

## Performance & SLOs

See `docs/SLO.md` for explicit objectives and load-test execution.
