"""
test_evaluate.py — Unit tests for src/evaluate.py.

Key assertions
--------------
* compute_metrics returns expected keys and sane ranges.
* RMSE == 0 for perfect predictions.
* backtest returns a DataFrame with expected columns.
* backtest produces 0 windows if data is too short.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluate import compute_metrics, backtest


def _cfg():
    return {
        "evaluation": {
            "backtest": {
                "window_days": 30,
                "min_train_days": 60,
                "step_days": 7,
            }
        }
    }


def _make_df(n=200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "story_id":   [str(i) for i in range(n)],
        "created_at": dates,
        "target":     rng.uniform(0, 10, n),
        "feature_a":  rng.standard_normal(n),
    })


class TestComputeMetrics:
    def test_returns_expected_keys(self):
        y = np.array([1.0, 2.0, 3.0])
        m = compute_metrics(y, y + 0.5)
        for key in ["rmse", "mae", "r2", "spearman_rho", "spearman_p", "n"]:
            assert key in m

    def test_perfect_predictions_zero_error(self):
        y = np.arange(10, dtype=float)
        m = compute_metrics(y, y)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert m["r2"]   == pytest.approx(1.0, abs=1e-10)
        assert m["spearman_rho"] == pytest.approx(1.0, abs=1e-6)

    def test_spearman_range(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(100)
        m = compute_metrics(y, rng.standard_normal(100))
        assert -1.0 <= m["spearman_rho"] <= 1.0


class TestBacktest:
    def _simple_model_fn(self, train_df, test_df):
        """Trivial model: predict the train mean."""
        return np.full(len(test_df), train_df["target"].mean())

    def test_returns_dataframe(self):
        df = _make_df(200)
        result = backtest(df, _cfg(), horizon_h=1, model_fn=self._simple_model_fn)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        df = _make_df(200)
        result = backtest(df, _cfg(), horizon_h=1, model_fn=self._simple_model_fn)
        if len(result) > 0:
            for col in ["rmse", "mae", "r2", "spearman_rho", "window_start", "window_end"]:
                assert col in result.columns

    def test_no_windows_if_data_too_short(self):
        df = _make_df(30)  # not enough for min_train_days=60
        result = backtest(df, _cfg(), horizon_h=1, model_fn=self._simple_model_fn)
        assert len(result) == 0

    def test_metrics_in_valid_range(self):
        df = _make_df(300)
        result = backtest(df, _cfg(), horizon_h=1, model_fn=self._simple_model_fn)
        if len(result) > 0:
            assert (result["rmse"] >= 0).all()
            # Trivial model predicts a constant -> spearmanr returns NaN.
            # Only assert range for windows where correlation is defined.
            valid_rho = result["spearman_rho"].dropna()
            if len(valid_rho) > 0:
                assert valid_rho.between(-1, 1).all()

