# ChurnOps developer commands. Run `make help` for a list.
PY ?= python
API_URL ?= http://localhost:8000

.PHONY: help install data baseline train promote test test-fast lint format api dashboard \
        simulate simulate-traffic drift drift-db mlflow-ui docker docker-run up down train-remote all

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## install pinned dev dependencies
	$(PY) -m pip install -r requirements.txt

data:  ## download raw IBM Telco CSV into data/raw/
	$(PY) -m src.data.download

baseline:  ## M1 logistic-regression baseline (no MLflow)
	$(PY) -m src.training.baseline

train:  ## train candidates, log to MLflow, register a new version
	$(PY) -m src.training.train

promote:  ## set the champion alias (challenger check) and export the champion
	$(PY) -m src.training.register

test:  ## full test suite with coverage
	$(PY) -m pytest --cov --cov-report=term

test-fast:  ## unit + API tests only
	$(PY) -m pytest -m "not integration" -q

lint:  ## ruff lint + format check
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:  ## auto-format
	$(PY) -m ruff check . --fix
	$(PY) -m ruff format .

api:  ## run the API locally, serving the exported champion
	MODEL_URI=artifacts/champion_model $(PY) -m uvicorn app.main:app --reload --port 8000

dashboard:  ## run the Streamlit dashboard locally
	API_URL=$(API_URL) $(PY) -m streamlit run dashboard/app.py

simulate:  ## write SIMULATED production batches
	$(PY) -m src.monitoring.batch_generator

simulate-traffic:  ## write batches AND send them to the running API
	$(PY) -m src.monitoring.batch_generator --send $(API_URL)

drift:  ## Evidently drift reports for every simulated scenario
	$(PY) -m src.monitoring.drift --all-scenarios

drift-db:  ## drift report on the last 500 logged API requests
	$(PY) -m src.monitoring.drift --from-db --last 500

mlflow-ui:  ## MLflow UI for the local SQLite store
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

docker:  ## build the API image
	docker build -t churnops-api:local .

docker-run:  ## run the API image on :8000
	docker run --rm -p 8000:8000 churnops-api:local

up:  ## full local stack (postgres, mlflow, api, dashboard)
	docker compose up --build -d

down:
	docker compose down

train-remote:  ## train + promote against the compose MLflow server
	MLFLOW_TRACKING_URI=http://localhost:5000 $(PY) -m src.training.train
	MLFLOW_TRACKING_URI=http://localhost:5000 $(PY) -m src.training.register

all: data baseline train promote simulate drift test  ## reproduce everything from scratch
