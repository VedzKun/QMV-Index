.PHONY: data features train backtest serve dashboard test lint all

PYTHON   := python
UVICORN  := uvicorn
STREAMLIT:= streamlit

# ── Pipeline ──────────────────────────────────────────────────────────────
data:
	$(PYTHON) -m src.collect.historical --days 90

features:
	$(PYTHON) -m src.features.build_features --out-dir data/processed

train:
	$(PYTHON) - <<'EOF'
	import pickle, numpy as np, pandas as pd
	from src.api.model_store import load_model
	for T in [1, 3, 6]:
	    load_model(T)        # triggers fresh training + pickle save
	    print(f"Trained T={T}")
'EOF'

backtest:
	$(PYTHON) -m pytest tests/ -v

# ── Serving ───────────────────────────────────────────────────────────────
serve:
	$(UVICORN) src.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	$(STREAMLIT) run src/dashboard/app.py --server.port 8501

# ── Testing ───────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	python -m flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503 || true

# ── Docker helpers ────────────────────────────────────────────────────────
docker-build:
	docker build -t qmv-index:latest .

docker-serve:
	docker run -p 8000:8000 qmv-index:latest

all: data features train test serve
