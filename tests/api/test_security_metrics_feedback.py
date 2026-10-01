"""API key auth, rate limiting, Prometheus metrics and the ground-truth feedback loop."""

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings
from src.config import RAW_FEATURES, TARGET_COL
from src.monitoring.batch_generator import generate, send_to_api


def make_client(model_service, repository, **settings):
    app = create_app(settings=Settings(**settings), model_service=model_service, repository=repository)
    return TestClient(app)


def metric_value(text: str, name: str) -> float | None:
    for line in text.splitlines():
        if line.startswith(name + " ") or line.startswith(name + "{"):
            return float(line.rsplit(" ", 1)[1])
    return None


# --- auth -------------------------------------------------------------------
@pytest.fixture()
def secured(model_service, repository):
    with make_client(model_service, repository, api_keys=("old-key", "new-key")) as c:
        yield c


def test_scoring_requires_api_key_when_configured(secured, customer):
    r = secured.post("/predict", json=customer)
    assert r.status_code == 401 and r.headers["www-authenticate"] == "ApiKey"
    assert secured.post("/predict", json=customer, headers={"X-API-Key": "wrong"}).status_code == 401
    assert (
        secured.post(
            "/feedback", json={"outcomes": [{"prediction_id": "x", "actual_outcome": 1}]}
        ).status_code
        == 401
    )


def test_any_configured_key_is_accepted(secured, customer):
    for key in ("old-key", "new-key"):  # rotation: both keys valid during the switch
        assert secured.post("/predict", json=customer, headers={"X-API-Key": key}).status_code == 200


def test_ops_endpoints_stay_open(secured):
    for path in ("/health", "/model-info", "/metrics", "/docs"):
        assert secured.get(path).status_code == 200, path


def test_auth_disabled_by_default(client, customer):
    assert client.post("/predict", json=customer).status_code == 200


# --- rate limit -------------------------------------------------------------
def test_rate_limit_returns_429_with_retry_after(model_service, repository, customer):
    with make_client(model_service, repository, rate_limit_per_minute=2) as c:
        assert [c.post("/predict", json=customer).status_code for _ in range(2)] == [200, 200]
        r = c.post("/predict", json=customer)
        assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
        assert c.get("/health").status_code == 200  # health checks are never limited


def test_rate_limit_is_per_api_key(model_service, repository, customer):
    with make_client(model_service, repository, api_keys=("a", "b"), rate_limit_per_minute=1) as c:
        assert c.post("/predict", json=customer, headers={"X-API-Key": "a"}).status_code == 200
        assert c.post("/predict", json=customer, headers={"X-API-Key": "a"}).status_code == 429
        assert c.post("/predict", json=customer, headers={"X-API-Key": "b"}).status_code == 200


# --- metrics ----------------------------------------------------------------
def test_metrics_count_requests_and_predictions(client, customer):
    client.post("/predict", json=customer)
    client.post("/batch-predict", json={"records": [customer, customer]})
    client.post("/predict", json={**customer, "tenure": -1})  # 422
    text = client.get("/metrics").text
    assert (
        'churnops_model_info{model_alias="champion",model_name="churnops-model",model_version="test"} 1.0'
        in text
    )
    assert 'churnops_http_requests_total{method="POST",path="/predict",status="200"} 1.0' in text
    assert 'churnops_http_requests_total{method="POST",path="/predict",status="422"} 1.0' in text
    assert metric_value(text, "churnops_churn_probability_count") == 3
    served = sum(
        float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if line.startswith("churnops_predictions_total{")
    )
    assert served == 3
    assert "churnops_http_request_latency_seconds_bucket" in text


def test_unknown_paths_do_not_create_new_metric_labels(client):
    client.get("/does-not-exist-123")
    text = client.get("/metrics").text
    assert "does-not-exist" not in text and 'path="unmatched"' in text


# --- feedback ---------------------------------------------------------------
def test_feedback_attaches_outcome_to_logged_prediction(client, customer, repository):
    pid = client.post("/predict", json=customer).json()["prediction_id"]
    r = client.post(
        "/feedback",
        json={
            "outcomes": [
                {"prediction_id": pid, "actual_outcome": 1},
                {"prediction_id": "nope", "actual_outcome": 0},
            ]
        },
    )
    assert r.status_code == 200
    assert r.json() == {"updated": 1, "not_found": ["nope"]}
    (row,) = repository.labelled()
    assert row.prediction_id == pid and row.actual_outcome == 1
    assert 'churnops_feedback_total{actual_outcome="1"} 1.0' in client.get("/metrics").text


@pytest.mark.parametrize(
    "bad", [{"outcomes": []}, {"outcomes": [{"prediction_id": "a", "actual_outcome": 2}]}]
)
def test_feedback_validation(client, bad):
    assert client.post("/feedback", json=bad).status_code == 422


def test_feedback_unavailable_without_prediction_log(model_service):
    app = create_app(settings=Settings(log_predictions=False), model_service=model_service)
    with TestClient(app) as c:
        assert (
            c.post("/feedback", json={"outcomes": [{"prediction_id": "a", "actual_outcome": 1}]}).status_code
            == 503
        )


# --- end-to-end: simulated traffic + delayed labels -> live performance gauges
def test_simulated_traffic_with_feedback_produces_live_metrics(
    model_service, repository, clean_df, monkeypatch
):
    app = create_app(
        settings=Settings(api_keys=("sim-key",), performance_min_labelled=20),
        model_service=model_service,
        repository=repository,
    )
    # send_to_api opens its own httpx.Client; hand it a TestClient (an httpx.Client subclass) instead.
    monkeypatch.setattr(httpx, "Client", lambda base_url, timeout, headers: TestClient(app, headers=headers))
    batch = generate(clean_df, "baseline", n=60, seed=1, keep_target=True)
    assert list(batch.columns) == ["customerID", *RAW_FEATURES, TARGET_COL]

    assert send_to_api(batch, "http://test", chunk=25, feedback=True, api_key="sim-key") == 60

    rows = repository.labelled()
    assert len(rows) == 60
    assert sorted(r.actual_outcome for r in rows) == sorted(batch[TARGET_COL].tolist())
    with TestClient(app) as c:  # second lifespan run for the same app
        text = c.get("/metrics").text
    assert text.count("churnops_live_roc_auc{") == 1  # restarts must not duplicate the collector
    assert metric_value(text, "churnops_live_labelled_predictions") == 60
    roc_auc = metric_value(text, "churnops_live_roc_auc")
    assert roc_auc is not None and 0.5 < roc_auc <= 1


def test_send_to_api_feedback_needs_labels(clean_df):
    with pytest.raises(ValueError, match="keep_target"):
        send_to_api(generate(clean_df, "baseline", n=5), "http://unused", feedback=True)
