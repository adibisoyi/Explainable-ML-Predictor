# Learn Notes

## Core ML concepts (high signal)

- **Train/validation split**: We hold out validation data so we can estimate model quality on unseen examples.
- **Pipelines**: A pipeline bundles preprocessing + model so inference uses the exact same transformations as training.
- **Why save artifacts**: Saving trained objects (`pipeline.joblib`) avoids retraining every time we serve predictions.
- **Overfitting basics**: A model that memorizes training data often performs worse on new data; validation metrics help detect this.
- **What probabilities mean**: `predict_proba` is the model's confidence estimate for class membership, not a guarantee.
- **SHAP basics (local explanations)**: SHAP attributes a prediction to feature contributions for a single row.

## SHAP in this project

- **What SHAP is**: A method grounded in Shapley values that distributes prediction impact across features.
- **Why we use it**: It gives per-prediction reasoning, useful for debugging and stakeholder trust.
- **Base value**: The model's reference output before feature effects are added.
- **Contribution**: How much each feature pushes the prediction away from the base value.
- **Limitations**:
  - Local explanations do not replace global model understanding.
  - Correlated features can share or swap attribution, so interpretation requires care.

## Build log (what was built and why)

### Step 1 — Repo skeleton + config
- Added a src-layout package to keep runtime code separate from tests/docs.
- Added a small dependency set (`sklearn`, `fastapi`, `shap`, `joblib`) for a minimal but complete MVP.
- Added artifact naming constants so every module uses one source of truth.

### Step 2 — Training pipeline + artifacts
- Implemented dataset loading from built-in breast cancer data (plus basic CSV option).
- Built two baseline models (`logistic`, `rf`) inside sklearn pipelines.
- Saved model artifact, metadata, and SHAP background sample so serving is immediate.

### Step 3 — SHAP explanation logic
- Added single-row SHAP explanation function that works with linear and tree models.
- Returned compact top-k contributions to keep API responses easy to inspect.
- Included predicted probability with explanation output for practical decision support.

### Step 4 — FastAPI + schemas
- Added explicit request schema matching the breast cancer feature names.
- Added `/health`, `/predict`, and `/explain` endpoints with startup-time artifact loading.
- Added clear 503 error messages when artifacts are missing.

### Step 5 — CLI wrappers
- Added `train`, `serve`, `predict`, `explain`, and `sample-json` commands.
- Kept CLI local-first so users can demo the project without external services.
- Reused the same artifact format between CLI and API to avoid duplication.

### Step 6 — Smoke tests
- Added a train smoke test to verify artifacts and reasonable validation accuracy.
- Added API startup/health smoke test with FastAPI `TestClient`.
- Focused on fast checks so contributors can run tests quickly.

### Step 7 — Documentation
- Added README quickstart and practical examples with curl payloads.
- Added architecture notes to explain module boundaries and data flow.
- Kept docs focused on the "why" behind each major component.
