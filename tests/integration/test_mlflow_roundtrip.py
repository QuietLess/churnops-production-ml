"""Registry integration: log -> register -> alias -> load by alias -> predict.

Uses a throwaway SQLite-backed MLflow store in tmp_path (no server needed).
"""

import mlflow
import pytest
from mlflow import MlflowClient

from app.model_service import ModelService
from src.config import RAW_FEATURES
from src.training.serialization import UntrustedModelTypeError, trusted_types_for

pytestmark = pytest.mark.integration


@pytest.fixture()
def tracking(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.chdir(tmp_path)  # artifacts land in tmp_path/mlruns
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.set_tracking_uri(None)


def test_champion_alias_roundtrip(tracking, fitted_pipeline, clean_df, tmp_path):
    X = clean_df[RAW_FEATURES]
    mlflow.set_experiment("test")
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            fitted_pipeline,
            name="model",
            registered_model_name="churnops-test",
            metadata={"threshold": 0.33, "explain_background": []},
            skops_trusted_types=trusted_types_for(fitted_pipeline),
        )
    MlflowClient().set_registered_model_alias("churnops-test", "champion", info.registered_model_version)

    service = ModelService.load("models:/churnops-test@champion", tracking)
    assert service.model_alias == "champion"
    assert service.model_version == str(info.registered_model_version)
    assert service.threshold == 0.33
    out = service.score(X.head(5).to_dict(orient="records"))
    assert len(out) == 5 and all(0 <= o["churn_probability"] <= 1 for o in out)

    # Exported-directory path (used for deployment images) carries lineage in metadata.
    export = tmp_path / "export"
    mlflow.sklearn.save_model(
        fitted_pipeline,
        str(export),
        metadata={
            "threshold": 0.33,
            "model_name": "churnops-test",
            "model_alias": "champion",
            "model_version": service.model_version,
        },
        skops_trusted_types=trusted_types_for(fitted_pipeline),
    )
    exported = ModelService.load(str(export))
    assert exported.model_version == service.model_version
    assert exported.score(X.head(1).to_dict(orient="records")) == service.score(
        X.head(1).to_dict(orient="records")
    )


def test_missing_model_fails_fast(tracking):
    from app.model_service import ModelLoadError

    with pytest.raises(ModelLoadError):
        ModelService.load("models:/does-not-exist@champion", tracking)


def test_untrusted_types_are_refused():
    class Sneaky:  # defined outside the allowlisted packages
        pass

    with pytest.raises(UntrustedModelTypeError):
        trusted_types_for({"payload": Sneaky()})
