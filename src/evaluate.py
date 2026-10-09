"""
evaluate.py — Metric computation and time-aware back-testing.

Public API
----------
    compute_metrics(y_true, y_pred) -> dict
    backtest(df, cfg, horizon_h, model_fn) -> pd.DataFrame
    plot_backtest(results_df, cfg) -> matplotlib.figure.Figure
"""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute regression metrics.

    Returns
    -------
    dict with keys: rmse, mae, r2, spearman_rho, spearman_p
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    rho, pval = spearmanr(y_true, y_pred)
    return {
        "rmse":        rmse,
        "mae":         mae,
        "r2":          float(r2),
        "spearman_rho": float(rho),
        "spearman_p":   float(pval),
        "n":           len(y_true),
    }


# ---------------------------------------------------------------------------
# Back-test
# ---------------------------------------------------------------------------

def backtest(
    df: pd.DataFrame,
    cfg: dict,
    horizon_h: int,
    model_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
) -> pd.DataFrame:
    """
    Rolling-window back-test (time-aware, no leakage).

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with 'created_at' and 'target'.
    cfg : dict
        Master config.
    horizon_h : int
        Prediction horizon (for logging only; model_fn handles features).
    model_fn : Callable[[train_df, test_df], np.ndarray]
        Function that trains on train_df, predicts on test_df.
        Must return array of predictions aligned to test_df index.

    Returns
    -------
    pd.DataFrame
        One row per back-test window with columns:
        window_start, window_end, n_train, n_test, rmse, mae, r2, spearman_rho.
    """
    bcfg = cfg["evaluation"]["backtest"]
    window  = timedelta(days=bcfg["window_days"])
    min_tr  = timedelta(days=bcfg["min_train_days"])
    step    = timedelta(days=bcfg["step_days"])

    df = df.sort_values("created_at").reset_index(drop=True)
    t_start = df["created_at"].min()
    t_end   = df["created_at"].max()

    current = t_start + min_tr + window
    results = []

    while current <= t_end:
        test_start  = current - window
        train_end   = test_start - timedelta(days=1)
        train_start = t_start

        train_df = df[df["created_at"] <= train_end]
        test_df  = df[(df["created_at"] >= test_start) & (df["created_at"] < current)]

        if len(train_df) < 50 or len(test_df) < 10:
            current += step
            continue

        preds = model_fn(train_df, test_df)
        m = compute_metrics(test_df["target"].values, preds)
        m["window_start"] = test_start.date()
        m["window_end"]   = current.date()
        m["n_train"]      = len(train_df)
        results.append(m)

        logger.debug("Backtest window %s→%s  RMSE=%.3f  ρ=%.3f",
                     test_start.date(), current.date(), m["rmse"], m["spearman_rho"])
        current += step

    if not results:
        logger.warning("No back-test windows produced — check min_train_days and data range.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    logger.info("Backtest complete: %d windows, mean RMSE=%.3f, mean ρ=%.3f",
                len(results_df), results_df["rmse"].mean(), results_df["spearman_rho"].mean())
    return results_df
