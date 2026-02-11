import json

from exml.train import train_and_save


def test_train_smoke(tmp_path):
    metadata = train_and_save(model_name="logistic", out_dir=str(tmp_path))

    assert (tmp_path / "pipeline.joblib").exists()
    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "background.joblib").exists()

    saved_metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["model_name"] == "logistic"
    assert saved_metadata["metrics"]["accuracy"] > 0.8
