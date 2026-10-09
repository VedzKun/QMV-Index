"""
shap_explain.py — SHAP-based interpretability layer.

Public API
----------
    build_explainer(booster, X_background) -> shap.TreeExplainer
    shap_summary_plot(explainer, X, feature_names, cfg) -> Figure
    shap_waterfall_plot(explainer, X_row, feature_names, cfg) -> Figure
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def build_explainer(booster, X_background: np.ndarray) -> shap.TreeExplainer:
    """Create a SHAP TreeExplainer with a sampled background dataset."""
    explainer = shap.TreeExplainer(booster, data=X_background,
                                   feature_perturbation="interventional")
    logger.info("SHAP TreeExplainer built on %d background samples.", len(X_background))
    return explainer


def shap_summary_plot(
    explainer: shap.TreeExplainer,
    X: np.ndarray,
    feature_names: list[str],
    cfg: dict,
    title: str = "SHAP Feature Importance",
) -> plt.Figure:
    """
    Beeswarm summary plot of SHAP values.

    Returns
    -------
    matplotlib Figure (caller is responsible for saving/closing).
    """
    shap_values = explainer(X)
    top_k = cfg["shap"]["top_k_features"]

    fig, ax = plt.subplots(figsize=(10, 0.4 * top_k + 2))
    shap.plots.beeswarm(shap_values, max_display=top_k, show=False)
    plt.title(title, fontsize=13, pad=10)
    plt.tight_layout()
    return fig


def shap_waterfall_plot(
    explainer: shap.TreeExplainer,
    X_row: np.ndarray,
    feature_names: list[str],
    cfg: dict,
    title: str = "SHAP Waterfall — Single Story",
) -> plt.Figure:
    """
    Waterfall plot for a single story's SHAP decomposition.

    Parameters
    ----------
    X_row : np.ndarray
        1-D array of feature values for one story.
    """
    sv = explainer(X_row.reshape(1, -1))
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(sv[0], show=False)
    plt.title(title, fontsize=13, pad=10)
    plt.tight_layout()
    return fig


def save_figure(fig: plt.Figure, name: str, cfg: dict) -> Path:
    """Save a matplotlib figure to reports/figures/."""
    fig_dir = Path(cfg["reports"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"{name}.png"
    fig.savefig(out, dpi=cfg["reports"]["dpi"], bbox_inches="tight")
    logger.info("Figure saved → %s", out)
    return out
