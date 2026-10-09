"""
index.py — QMV Index: rescale model predictions to [0, 100] percentile.

The QMV Index is a RELATIVE score: a story's predicted attention value
is compared to the distribution of all stories in a trailing 30-day window.

Public API
----------
    QMVScaler            — stateful scaler (fit on window, transform new stories)
    compute_qmv_index(predictions, window_predictions) -> np.ndarray
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

logger = logging.getLogger(__name__)


class QMVScaler:
    """
    Percentile-based scaler for the QMV Index.

    Usage
    -----
    >>> scaler = QMVScaler(cfg)
    >>> scaler.fit(window_predictions)      # trailing 30-day predictions
    >>> qmv = scaler.transform(new_preds)   # returns float array in [0, 100]
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg["qmv_index"]
        self._window: np.ndarray | None = None

    def fit(self, window_predictions: np.ndarray) -> "QMVScaler":
        """Store trailing-window distribution."""
        min_n = self.cfg["min_window_stories"]
        if len(window_predictions) < min_n:
            raise ValueError(
                f"Window has only {len(window_predictions)} stories; "
                f"minimum required is {min_n}. Widen the date range."
            )
        self._window = np.asarray(window_predictions, dtype=float)
        logger.info("QMVScaler fitted on %d window stories.", len(self._window))
        return self

    def transform(self, predictions: np.ndarray) -> np.ndarray:
        """Map predictions to [0, 100] percentile index."""
        if self._window is None:
            raise RuntimeError("QMVScaler must be fitted before transform.")
        preds = np.asarray(predictions, dtype=float)
        scores = np.array([
            percentileofscore(self._window, p, kind="rank")
            for p in preds
        ])
        lo, hi = self.cfg["output_scale"]
        return np.clip(scores, lo, hi)

    def fit_transform(self, predictions: np.ndarray) -> np.ndarray:
        """Fit on the same array and transform (useful for historical scoring)."""
        return self.fit(predictions).transform(predictions)


def compute_qmv_index(
    predictions: np.ndarray,
    window_predictions: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """
    One-shot QMV scoring.

    Parameters
    ----------
    predictions : np.ndarray
        Raw model predictions for stories to score.
    window_predictions : np.ndarray
        Raw predictions for trailing window (baseline distribution).
    cfg : dict
        Master config.

    Returns
    -------
    np.ndarray
        QMV scores in [0, 100].
    """
    scaler = QMVScaler(cfg)
    scaler.fit(window_predictions)
    return scaler.transform(predictions)
