# QMV Index

**Quantitative Market Value Index** — an interpretable model that assigns a numeric attention value to Hacker News content using early engagement signals, built as a portfolio project for ZZAZZ.

---

## What it does

Given a Hacker News story at T hours after posting (T ∈ {1, 3, 6}), the model predicts its **realized attention value at 72 hours**:

`
target = log1p(points_72h + 2 * comments_72h)
`

This prediction is then rescaled to a **[0–100] percentile QMV Index** relative to a trailing 30-day window of content — making it directly comparable across time.

---

## Quick Start

`ash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch raw data (requires internet; ~5 min for 1 year)
python -m src.ingest

# 3. Run the full pipeline
python -m src.train

# 4. Run tests
pytest tests/ -v --cov=src
`

---

## Repo Structure

`
QMV-Index/
├── config.yaml          # Single source of truth — all params here
├── requirements.txt
├── data/
│   ├── raw/             # Immutable Parquet snapshots (gitignored)
│   ├── interim/         # Cleaned tables (gitignored)
│   └── features/        # Feature matrices per horizon (gitignored)
├── src/
│   ├── ingest.py        # HN Algolia API fetcher
│   ├── clean.py         # Dedup, type-cast, quality flags
│   ├── features.py      # Pure feature engineering functions
│   ├── split.py         # Chronological splitter + leakage checker
│   ├── train.py         # Ridge / GAM / LightGBM training
│   ├── evaluate.py      # Metrics + rolling back-test
│   ├── index.py         # QMV [0-100] percentile scaler
│   └── shap_explain.py  # SHAP summary + waterfall helpers
├── notebooks/           # EDA and model showcases (read-only analysis)
├── tests/               # pytest suite (run before every commit)
└── reports/figures/     # Generated plots
`

---

## Key Design Decisions

| Decision | Why |
|---|---|
| Chronological splits only | Prevents temporal leakage; simulates real deployment |
| config.yaml as single truth | No magic numbers in code; reproducible runs |
| T=0 metadata features as v1 | HN API doesn't archive hourly snapshots; honest about this |
| GAM + Ridge as baselines | Quantifies LightGBM's non-linear value |
| SHAP for interpretability | Explains individual story valuations for ZZAZZ use case |

---

## Known Limitations (v0.1)

- **No true hourly snapshots**: Early engagement features (T=1/3/6h) are filled with zeros until live collection is implemented. Current model uses T=0 metadata only.
- **Target approximation**: points and 
um_comments at fetch time proxy the 72h values; stories fetched at exactly 72h may still be accruing attention.
- **Author karma**: Not currently pulled from Algolia (requires per-user API calls). Architecture supports it when added.

---

## Reproducibility

All random seeds are pinned via config.yaml:project.random_seed.
Data pipeline is deterministic given the same etch_start_date / etch_end_date.

---

*Built for ZZAZZ — content pricing and trading via real-time AI market valuations.*

---

## Running the Snapshot Collector (2–3 week data collection)

### One-time setup
```bash
pip install -r requirements.txt
python -m src.collect.historical --config config.yaml   # bootstrap with Algolia data (~30 min)
```

### Start the live collector
```bash
# Foreground (development)
python -m src.collect.snapshot --run --db data/raw/hn_snapshots.db

# Background via nohup (Linux / WSL)
nohup python -m src.collect.snapshot --run > data/raw/collector.log 2>&1 &

# Windows Task Scheduler: run this command every minute as a recurring task.
# The collector sleeps internally, so it is safe to call more often than needed.
```

### systemd timer (recommended for a VPS / Linux box)

Create `/etc/systemd/system/qmv-collector.service`:
```ini
[Unit]
Description=QMV HN Snapshot Collector

[Service]
WorkingDirectory=/path/to/QMV-Index
ExecStart=/usr/bin/python -m src.collect.snapshot --run
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl enable --now qmv-collector
sudo journalctl -u qmv-collector -f   # live logs
```

### What you need for a useful dataset
| Duration | Stories tracked | Usable 72h targets | Notes |
|---|---|---|---|
| 3 days | ~2 000 | ~400 | Minimum for EDA only |
| 1 week | ~5 000 | ~2 500 | Rough model training |
| **2–3 weeks** | **~15 000** | **~12 000** | **Recommended minimum** |
| 6 weeks | ~40 000 | ~35 000 | Solid back-test |

### Monitoring during collection
```bash
# How many stories tracked so far?
sqlite3 data/raw/hn_snapshots.db "SELECT COUNT(*) FROM stories;"

# Snapshot completion rate
sqlite3 data/raw/hn_snapshots.db "
SELECT target_hours, SUM(done=1) as ok, SUM(done=2) as failed, SUM(done=0) as pending
FROM scheduled_snapshots GROUP BY target_hours ORDER BY target_hours;"

# Data quality events (grep structured log)
Select-String "QUALITY|" data/raw/collector.log | Select-Object -Last 20
```

### Stopping safely
`Ctrl+C` — the collector handles `KeyboardInterrupt` cleanly and closes the DB
before exiting. All pending scheduled_snapshots rows survive in the DB and will
resume on next start.
