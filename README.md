# QMV Index — Quantitative Market Value for Content

> *Predict the realized attention value of a Hacker News story within hours of posting.*

[![CI](https://github.com/VedzKun/QMV-Index/actions/workflows/ci.yml/badge.svg)](https://github.com/VedzKun/QMV-Index/actions)

---

## What is this?

The **QMV Index** is a transparent, interpretable model that assigns a **0-100 score** to a piece of content based on its **early engagement signals** (observed at T = 1, 3, or 6 hours after posting). The score represents the story's predicted percentile of realized attention (points + comments) at 72 hours, relative to the trailing 30-day baseline.

Built as a portfolio project for a data scientist role at **ZZAZZ** — a company that prices and trades content using real-time AI valuations.

---

## Project Structure

```
QMV-Index/
├── src/
│   ├── collect/          # Live snapshot collector + Algolia bootstrap
│   ├── eda/              # EDA helpers and data simulation
│   ├── features/         # Feature engineering (7 families, leakage-tested)
│   ├── models/           # CV splitter, metrics
│   ├── backtest/         # Rolling-origin backtest, QMV index builder
│   └── api/              # FastAPI service + Streamlit dashboard
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_modeling.ipynb     # Model comparison table
│   ├── 03_backtest_qmv.ipynb # Rolling backtest + index validation
│   └── 03_interpretability.ipynb # SHAP / PDP / explain_story()
├── reports/
│   ├── model_card.md
│   ├── feature_dictionary.md
│   ├── interpretability.md
│   └── failure_analysis.md
├── tests/
│   ├── test_collect.py       # 38 unit tests for data collection
│   └── test_leakage.py       # Leakage tests for feature engineering
├── Dockerfile
├── Makefile
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
pip install uvicorn streamlit
```

### 2. Bootstrap data (or use synthetic data)
```bash
# Option A: real data (requires 2-3 weeks of collection)
python -m src.collect.snapshot --run

# Option B: synthetic data (immediate, used for all notebooks)
python -m src.features.build_features --out-dir data/processed
```

### 3. Run the API
```bash
make serve
# → http://localhost:8000/docs
```

### 4. Run the demo dashboard
```bash
make dashboard
# → http://localhost:8501
```

### 5. Run all tests
```bash
make test
```

---

## API Usage

```bash
curl -X POST http://localhost:8000/valuate \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Show HN: I built a QMV index for HN",
    "author": "pg",
    "posted_at": 1728000000,
    "T": 3,
    "snapshots": [
      {"hours_since_post": 1, "score": 12,  "descendants": 4},
      {"hours_since_post": 3, "score": 45,  "descendants": 18}
    ]
  }'
```

**Response:**
```json
{
  "qmv_index": 78.4,
  "predicted_value": 5.21,
  "interval_low": 4.11,
  "interval_high": 6.31,
  "top_drivers": [
    {"feature": "v3", "contribution": 0.812, "direction": "positive"},
    {"feature": "author_hist_median", "contribution": 0.34, "direction": "positive"}
  ],
  "model_version": "a3f2c1b0"
}
```

---

## Methodology

| Step | Detail |
|---|---|
| **Target** | `log1p(points_72h + 2 × comments_72h)` |
| **Leakage prevention** | 72-hour gap in time-series CV; expanding-window author/domain features shifted by 1 story |
| **Model** | HistGradientBoostingRegressor with monotonic constraints on velocity features |
| **Interpretability** | Ridge coefficients for signed attributions; partial dependence plots |
| **Index** | Trailing 30-day percentile of predicted values |

See `reports/model_card.md` for full metrics, limitations, and failure modes.

---

## Reproducing Results

```bash
make data      # bootstrap historical data
make features  # build T=1,3,6 feature matrices
make train     # train and pickle models
make backtest  # rolling-origin evaluation
make test      # unit + leakage tests
```

---

## License

MIT
