# Explainable ML Predictor

A minimal end-to-end machine learning project that trains a tabular classifier, serves predictions with FastAPI, and explains individual predictions using SHAP.

## WHY this project exists

- Show the full ML lifecycle in a small codebase: train -> save artifacts -> serve -> explain.
- Demonstrate explainability in a practical API (`/explain`) rather than theory-only notebooks.
- Provide a clean baseline you can extend to your own CSV data later.

## Quick install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Train a model

```bash
python -m exml.cli train --model logistic --out artifacts/
python -m exml.cli train --model rf --out artifacts/
```

Training prints validation metrics (accuracy + ROC-AUC) and saves:

- `artifacts/pipeline.joblib`
- `artifacts/metadata.json`
- `artifacts/background.joblib`

## Run the API

```bash
python -m exml.cli serve
```

Endpoints:

- `GET /health`
- `POST /predict`
- `POST /explain`

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
  -d @sample.json
```

```bash
curl -s -X POST http://127.0.0.1:8000/explain \
  -H 'Content-Type: application/json' \
  -d @sample.json
```

## Example responses

`/predict`

```json
{
  "predicted_class": 0,
  "predicted_probability": 0.9996
}
```

`/explain` (truncated)

```json
{
  "base_value": 0.36,
  "predicted_probability": 0.9996,
  "top_contributions": [
    {"feature": "worst perimeter", "value": 184.6, "contribution": 1.12},
    {"feature": "mean concave points", "value": 0.1471, "contribution": 0.79}
  ]
}
```

## CLI commands

```bash
python -m exml.cli train --model logistic --out artifacts/
python -m exml.cli serve
python -m exml.cli sample-json
python -m exml.cli predict --json '{"mean radius": 17.99, ...}'
python -m exml.cli explain --json '{"mean radius": 17.99, ...}'
```

## Train from CSV (basic extension)

Your CSV must contain a target column (default name: `target`) and numeric feature columns.

```bash
python -m exml.cli train --model logistic --csv your_data.csv --target target --out artifacts/
```

## How this helps in real-world ML

- Lets teams inspect *why* a prediction happened, not only *what* happened.
- Makes model behavior easier to debug when performance drifts.
- Provides an API contract (schemas + metadata) that is production-friendly from day one.
