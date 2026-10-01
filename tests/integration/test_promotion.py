"""Champion/challenger promotion rules against a throwaway MLflow registry."""

import mlflow
import pytest
from mlflow import MlflowClient

from src.config import MODEL_ALIAS, REGISTERED_MODEL_NAME
from src.training.register import current_champion, promote, rollback
from src.training.serialization import trusted_types_for

pytestmark = pytest.mark.integration


@pytest.fixture()
def registry(tmp_path, monkeypatch, fitted_pipeline):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("promotion-test")
    client = MlflowClient()

    def register(score: float) -> str:
        with mlflow.start_run():
            info = mlflow.sklearn.log_model(
                fitted_pipeline,
                name="model",
                registered_model_name=REGISTERED_MODEL_NAME,
                metadata={"threshold": 0.5},
                skops_trusted_types=trusted_types_for(fitted_pipeline),
            )
        v = str(info.registered_model_version)
        client.set_model_version_tag(REGISTERED_MODEL_NAME, v, "valid_pr_auc", str(score))
        return v

    yield client, register
    mlflow.set_tracking_uri(None)


def test_first_version_becomes_champion(registry, tmp_path):
    client, register = registry
    v1 = register(0.60)
    version, promoted = promote(client, export_dir=tmp_path / "export")
    assert (version, promoted) == (v1, True)
    assert str(current_champion(client).version) == v1
    assert (tmp_path / "export" / "MLmodel").exists()


def test_worse_challenger_is_not_promoted(registry):
    client, register = registry
    v1 = register(0.65)
    promote(client, export_dir=None)
    register(0.60)  # worse
    version, promoted = promote(client, export_dir=None)
    assert (version, promoted) == (v1, False)
    assert str(client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS).version) == v1


def test_better_challenger_is_promoted_and_force_overrides(registry):
    client, register = registry
    register(0.60)
    promote(client, export_dir=None)
    v2 = register(0.66)
    assert promote(client, export_dir=None) == (v2, True)
    v3 = register(0.50)
    assert promote(client, v3, force=True, export_dir=None) == (v3, True)


def champion_version(client) -> str:
    return str(client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS).version)


def test_rollback_restores_previous_champion_and_walks_history(registry, tmp_path):
    client, register = registry
    v1 = register(0.60)
    promote(client, export_dir=None)
    v2 = register(0.62)
    promote(client, export_dir=None)
    v3 = register(0.64)
    promote(client, export_dir=None)
    assert client.get_model_version(REGISTERED_MODEL_NAME, v3).tags["previous_champion"] == v2

    assert rollback(client, export_dir=tmp_path / "export") == (v3, v2)
    assert champion_version(client) == v2
    assert (tmp_path / "export" / "MLmodel").exists()  # the restored model is what gets served
    assert "rolled_back" in client.get_model_version(REGISTERED_MODEL_NAME, v3).tags

    assert rollback(client, export_dir=None) == (v2, v1)  # second rollback goes one further back
    assert champion_version(client) == v1


def test_rolled_back_version_is_not_repromoted_without_force(registry):
    client, register = registry
    register(0.60)
    promote(client, export_dir=None)
    v2 = register(0.70)
    promote(client, export_dir=None)
    v1, _ = rollback(client, export_dir=None)[::-1]
    assert promote(client, export_dir=None) == (v1, False)  # latest (v2) is rolled back -> skipped
    assert promote(client, v2, force=True, export_dir=None) == (v2, True)


def test_rollback_without_history_fails(registry):
    client, register = registry
    register(0.60)
    with pytest.raises(SystemExit, match="No champion"):
        rollback(client, export_dir=None)
    promote(client, export_dir=None)
    with pytest.raises(SystemExit, match="previous_champion"):
        rollback(client, export_dir=None)
