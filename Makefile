.RECIPEPREFIX = >
SHELL := /bin/bash
PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.DEFAULT_GOAL := help

.PHONY: help setup data simulate features train evaluate experiments api dashboard test lint typecheck security sbom docker-up docker-down clean

help:  ## Show available targets
> @grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Create venv and install the project with all extras
> $(PY) -m venv $(VENV)
> $(BIN)/pip install --upgrade pip
> $(BIN)/pip install -e ".[all]"
> $(BIN)/pre-commit install || true

data: simulate  ## Alias for simulate

simulate:  ## Generate synthetic cloud telemetry (normal + attack scenarios)
> $(BIN)/cloudsentinel simulate --output data/synthetic/events.parquet

features:  ## Build the feature store from raw/synthetic events
> $(BIN)/cloudsentinel features build --input data/synthetic/events.parquet

train:  ## Train the configured detection models
> $(BIN)/cloudsentinel train --input data/features/store.parquet

evaluate:  ## Evaluate models and write reports/metrics.csv
> $(BIN)/cloudsentinel evaluate --input data/features/store.parquet

experiments:  ## Run the 7 scientific experiments end to end
> $(BIN)/cloudsentinel experiments --seeds 11,22,33 --deep

api:  ## Run the FastAPI backend locally
> $(BIN)/uvicorn cloudsentinel.api.main:app --reload --port 8000

dashboard:  ## Run the Next.js dashboard locally
> cd dashboard && npm install && npm run dev

test:  ## Run the test suite with coverage
> $(BIN)/pytest --cov=cloudsentinel --cov-report=term-missing

lint:  ## Lint and format check
> $(BIN)/ruff check .
> $(BIN)/ruff format --check .

typecheck:  ## Static type checking
> $(BIN)/mypy

security:  ## SAST, dependency audit and secret scanning
> $(BIN)/bandit -c pyproject.toml -r src -q
> $(BIN)/pip-audit || true
> $(BIN)/detect-secrets scan > .secrets.baseline.tmp && diff -q .secrets.baseline .secrets.baseline.tmp || true

sbom:  ## Generate a CycloneDX SBOM
> $(BIN)/pip install cyclonedx-bom >/dev/null
> $(BIN)/cyclonedx-py environment -o reports/sbom.json

docker-up:  ## Start the full local stack
> docker compose up --build

docker-down:  ## Stop the stack and remove volumes
> docker compose down -v

clean:  ## Remove caches and build artifacts
> rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build
> find . -name '__pycache__' -type d -prune -exec rm -rf {} +
