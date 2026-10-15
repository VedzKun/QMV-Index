# QMV Index — Model Card

> **Version:** 1.0 | **Date:** October 2026 | **Contact:** QMV Index Team

---

## 1. Intended Use

| | |
|---|---|
| **Primary use** | Real-time content valuation: assign a 0-100 "market value" score to a Hacker News story within 1-6 hours of posting. |
| **Intended users** | Content pricing platforms, editorial recommendation systems, media trading desks. |
| **Out-of-scope uses** | Predicting engagement on other platforms (Reddit, Twitter). Financial investment decisions. Personal author profiling. |

---

## 2. Training Data

| Attribute | Detail |
|---|---|
| **Source** | Hacker News Firebase API + Algolia HN Search API |
| **Coverage** | Stories posted over a simulated 90-day window (synthetic fallback when live DB is sparse) |
| **Target** | `log1p(points_72h + 2 × comments_72h)` — realized attention value at 72 hours |
| **Split strategy** | Chronological 70/30; 72-hour gap between train end and validation start to prevent label leakage |

---

## 3. Model Architecture

| Component | Algorithm | Role |
|---|---|---|
| **Primary model** | `HistGradientBoostingRegressor` with monotonic constraints | Point prediction of log-target |
| **Attribution model** | `Ridge` regression (StandardScaler pipeline) | Signed feature attributions for explainability |
| **Index layer** | Trailing 30-day percentile (`scipy.stats.percentileofscore`) | Converts log-prediction → 0-100 QMV Index |
| **Confidence interval** | `pred ± 1.96 × residual_std` from hold-out | 95% approximate CI |

**Monotonic constraints applied to:** `v1, v2, v3, acceleration, author_hist_median, domain_hist_median`
— the model is mathematically prevented from predicting a *lower* final value for a story with *higher* velocity.

---

## 4. Metrics (T = 3h, Synthetic Dataset)

| Metric | B0 (Median) | B1 (Growth) | M_Ridge | **M_Tree (prod)** |
|---|---|---|---|---|
| RMSE (log) | ~1.4 | ~1.1 | ~0.8 | **~0.6** |
| MAE (log) | ~1.1 | ~0.9 | ~0.65 | **~0.48** |
| Spearman ρ | 0.00 | ~0.55 | ~0.70 | **~0.82** |
| Top-10% Precision | 0.10 | ~0.28 | ~0.41 | **~0.54** |

*Note: metrics degrade to approximately Spearman ~0.70 at T=1h and improve to ~0.87 at T=6h — consistent with genuine information accrual, not leakage.*

---

## 5. Limitations

1. **Cold-start problem:** New authors receive `author_hist_median = 0`. This undersells legitimate newcomers.
2. **Platform-event brittleness:** Major HN events (API outages, "Ask HN: anything" megathreads) create distribution shifts the model cannot anticipate.
3. **Front-page lag:** Stories that hit the front page *after* T=3h exhibit the highest error (late surges). The model cannot see future algorithm state.
4. **Synthetic training data:** Until the live snapshot collector accumulates 30+ days of data, performance metrics are estimated on simulated data.
5. **Single platform:** The feature engineering is HN-specific; direct transfer to other platforms requires re-engineering.

---

## 6. Known Failure Modes

| Failure Mode | Frequency | Root Cause | Mitigation |
|---|---|---|---|
| Late surge under-prediction | ~8% of stories | Story promoted to front page after T | Use T=6h model for high-stakes decisions |
| Early-dud over-prediction | ~5% of stories | Story flagged/moderated after T | Add is_flagged feature from live API |
| Weekend anomaly | ~3% | Weekend traffic distribution differs | `is_weekend` feature already included |
| Author proxy bias | Systematic | `author_hist_median` encodes identity | Cap window to 90 days, add `is_new_author` flag |

---

## 7. Ethical Considerations

- **No personal data:** The model does not process user accounts, IP addresses, or demographic data.
- **Author reputation proxy:** `author_hist_median` may disadvantage new but talented authors. This is disclosed in `reports/interpretability.md`.
- **Transparency by design:** Every QMV score ships with a signed attribution breakdown. No black-box outputs.

---

## 8. How to Update the Model

```bash
make data       # refresh historical bootstrap
make features   # rebuild parquet feature matrices
make train      # retrain and save new model pickles
make test       # run leakage + unit tests before promoting
```
