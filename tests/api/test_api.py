import io

import pandas as pd

from tests.conftest import make_raw_df


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True and body["status"] == "ok" and body["database"] == "ok"


def test_model_info_reports_lineage(client):
    body = client.get("/model-info").json()
    assert body["model_name"] == "churnops-model"
    assert body["model_alias"] == "champion"
    assert body["model_version"] == "test"
    assert 0 < body["threshold"] < 1


def test_docs_available(client):
    assert client.get("/docs").status_code == 200
    assert "/predict" in client.get("/openapi.json").json()["paths"]


def test_predict_valid(client, customer):
    r = client.post("/predict", json=customer)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["prediction"] == int(body["churn_probability"] >= body["threshold"])
    assert body["risk_level"] in {"low", "medium", "high"}
    assert body["model_version"] == "test" and body["customerID"] == "DEMO-0001"
    assert body["logged"] is True
    assert "x-process-time-ms" in r.headers


def test_predict_negative_tenure_returns_422(client, customer):
    customer["tenure"] = -1
    assert client.post("/predict", json=customer).status_code == 422


def test_predict_unknown_category_returns_422(client, customer):
    customer["PaymentMethod"] = "Crypto"
    assert client.post("/predict", json=customer).status_code == 422


def test_predict_missing_field_returns_422(client, customer):
    customer.pop("Contract")
    assert client.post("/predict", json=customer).status_code == 422


def test_prediction_is_logged_with_expected_fields(client, customer, repository):
    body = client.post("/predict", json=customer).json()
    rows = repository.recent()
    assert len(rows) == 1
    row = rows[0]
    assert row.prediction_id == body["prediction_id"]
    assert row.customer_id == "DEMO-0001"
    assert row.model_version == "test"
    assert row.probability == body["churn_probability"]
    assert row.actual_outcome is None
    assert "customerID" not in row.request_features and row.request_features["tenure"] == 8


def test_batch_preserves_record_count_and_order(client, customer):
    records = [dict(customer, customerID=f"B{i}", tenure=i) for i in range(7)]
    r = client.post("/batch-predict", json={"records": records})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 7
    assert [p["customerID"] for p in body["predictions"]] == [f"B{i}" for i in range(7)]


def test_batch_rejects_empty(client):
    assert client.post("/batch-predict", json={"records": []}).status_code == 422


def test_batch_csv_upload(client, repository):
    df = make_raw_df(25, seed=3)  # raw format: includes Churn column and blank TotalCharges
    buf = io.BytesIO(df.to_csv(index=False).encode())
    r = client.post("/batch-predict/csv", files={"file": ("batch.csv", buf, "text/csv")})
    assert r.status_code == 200, r.text
    assert r.json()["count"] == 25
    assert repository.count() == 25


def test_batch_csv_reports_invalid_rows(client):
    df = make_raw_df(5, seed=4)
    df.loc[2, "tenure"] = -5
    buf = io.BytesIO(df.to_csv(index=False).encode())
    r = client.post("/batch-predict/csv", files={"file": ("batch.csv", buf, "text/csv")})
    assert r.status_code == 422
    assert r.json()["detail"]["rows"][0]["row"] == 2


def test_log_failure_does_not_fail_prediction(client, customer, repository, monkeypatch):
    def boom(rows):
        raise RuntimeError("db down")

    monkeypatch.setattr(repository, "log_many", boom)
    r = client.post("/predict", json=customer)
    assert r.status_code == 200 and r.json()["logged"] is False


def test_explain_returns_contributions(client, customer):
    r = client.post("/explain", json=customer)
    assert r.status_code == 200
    body = r.json()
    total = body["base_value"] + sum(
        c["contribution"] for c in body["top_increasing"] + body["top_decreasing"]
    )
    assert 0 <= body["churn_probability"] <= 1
    assert abs(total - body["churn_probability"]) < 0.05  # top-k only, so approximately
    assert "not causal" in body["note"]
    pred = client.post("/predict", json=customer).json()["churn_probability"]
    assert abs(pred - body["churn_probability"]) < 1e-4
    assert isinstance(pd.Series([c["feature"] for c in body["top_increasing"]]).tolist(), list)
