# Architecture

## System view

```text
[Dataset/CSV]
     |
     v
[Train Pipeline: preprocess + model] ---> [artifacts/pipeline.joblib + metadata.json + background.joblib]
                                                     |
                                                     v
                                         [FastAPI service (/predict, /explain)]
                                                     |
                                                     v
                                  [SHAP local explanation for one prediction]
```

## Module responsibilities

- `exml/data.py` — loads either the built-in breast cancer dataset or a basic CSV source.
- `exml/model.py` — defines small, readable sklearn pipelines for Logistic Regression and Random Forest.
- `exml/train.py` — trains model + preprocessing together, evaluates, and writes reproducible artifacts.
- `exml/explain.py` — computes local SHAP contributions for one input row.
- `exml/api.py` — serves health, prediction, and explanation endpoints with artifact loading once at startup.
- `exml/cli.py` — unifies train/serve/predict/explain commands so the project is runnable in a few commands.

## WHY this design exists

- Keeping preprocessing and model in one sklearn `Pipeline` prevents train/serve skew.
- Writing `metadata.json` keeps feature order explicit and avoids silent errors at inference time.
- Caching a small training background sample allows SHAP explanations without re-training.
- FastAPI + pydantic schemas make request validation explicit and beginner-friendly.
