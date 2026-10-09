"""
test_index.py — Unit tests for src/index.py (QMVScaler).

Key assertions
--------------
* Output is always in [0, 100].
* A higher raw prediction always yields a higher or equal QMV score (monotone).
* Raises if window is too small.
* fit_transform on uniform distribution returns spread across [0,100].
"""
import numpy as np
import pytest

from src.index import QMVScaler, compute_qmv_index


def _cfg(min_window=5):
    return {"qmv_index": {"min_window_stories": min_window, "output_scale": [0, 100]}}


class TestQMVScaler:
    def test_output_in_bounds(self):
        cfg = _cfg()
        scaler = QMVScaler(cfg)
        window = np.random.default_rng(0).uniform(0, 10, 200)
        scaler.fit(window)
        scores = scaler.transform(np.array([0.0, 5.0, 10.0, 15.0]))
        assert (scores >= 0).all() and (scores <= 100).all()

    def test_monotone(self):
        cfg = _cfg()
        scaler = QMVScaler(cfg).fit(np.arange(100, dtype=float))
        preds = np.array([10.0, 20.0, 30.0, 40.0])
        scores = scaler.transform(preds)
        assert list(scores) == sorted(scores), "QMV scores not monotonically increasing"

    def test_raises_small_window(self):
        cfg = _cfg(min_window=100)
        scaler = QMVScaler(cfg)
        with pytest.raises(ValueError, match="minimum required"):
            scaler.fit(np.arange(50, dtype=float))

    def test_raises_before_fit(self):
        cfg = _cfg()
        scaler = QMVScaler(cfg)
        with pytest.raises(RuntimeError, match="fitted"):
            scaler.transform(np.array([1.0, 2.0]))

    def test_fit_transform_spread(self):
        """Uniform distribution should yield well-spread scores."""
        cfg = _cfg(min_window=10)
        preds = np.linspace(0, 10, 100)
        scores = QMVScaler(cfg).fit_transform(preds)
        assert scores.min() < 10 and scores.max() > 90

    def test_compute_qmv_index_shape(self):
        cfg = _cfg(min_window=10)
        window = np.arange(200, dtype=float)
        preds  = np.array([5.0, 100.0, 150.0])
        out = compute_qmv_index(preds, window, cfg)
        assert out.shape == (3,)
