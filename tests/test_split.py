"""
test_split.py — Unit tests for src/split.py.

Key assertions
--------------
* No story_id overlap across any two splits.
* train max timestamp < val min timestamp.
* val max timestamp < test min timestamp.
* gap_days buffer is respected.
"""
import pandas as pd
import pytest

from src.split import chronological_split, leakage_check


def _make_cfg(gap_days=1, train_frac=0.70, val_frac=0.15):
    return {
        "split": {
            "strategy": "chronological",
            "train_frac": train_frac,
            "val_frac":   val_frac,
            "test_frac":  1 - train_frac - val_frac,
            "gap_days":   gap_days,
        }
    }


def _make_stories(n=500) -> pd.DataFrame:
    """Create n synthetic stories spanning ~500 days."""
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "story_id":   [str(i) for i in range(n)],
        "created_at": dates,
        "target":     [float(i % 10) for i in range(n)],
    })


class TestChronologicalSplit:
    def test_no_id_overlap(self):
        df = _make_stories()
        train, val, test = chronological_split(df, _make_cfg())
        leakage_check(train, val, test)  # raises if overlap

    def test_temporal_order(self):
        df = _make_stories()
        train, val, test = chronological_split(df, _make_cfg())
        assert train["created_at"].max() < val["created_at"].min()
        assert val["created_at"].max() < test["created_at"].min()

    def test_gap_respected(self):
        df = _make_stories()
        cfg = _make_cfg(gap_days=7)
        train, val, test = chronological_split(df, cfg)
        gap_tv = (val["created_at"].min() - train["created_at"].max()).days
        assert gap_tv >= 7, f"Gap train->val is {gap_tv} days, expected >=7"

    def test_coverage(self):
        """Stories should be fully assigned across splits (gap eats <=gap_days rows)."""
        df = _make_stories()
        gap = 1
        cfg = _make_cfg(gap_days=gap)
        train, val, test = chronological_split(df, cfg)
        total = len(train) + len(val) + len(test)
        # The gap buffer can drop at most 2*gap_days rows (one per boundary).
        assert total <= len(df)
        assert total >= len(df) - 2 * gap
        assert total > 0

    def test_sorted_output(self):
        df = _make_stories().sample(frac=1, random_state=42)  # shuffle input
        train, val, test = chronological_split(df, _make_cfg())
        # Each split must be internally sorted
        assert (train["created_at"].diff().dropna() >= pd.Timedelta(0)).all()
        assert (val["created_at"].diff().dropna()   >= pd.Timedelta(0)).all()
        assert (test["created_at"].diff().dropna()  >= pd.Timedelta(0)).all()


class TestLeakageCheck:
    def test_passes_clean_split(self):
        df = _make_stories()
        train, val, test = chronological_split(df, _make_cfg())
        leakage_check(train, val, test)  # should not raise

    def test_raises_on_id_overlap(self):
        df = _make_stories(300)
        train = df.iloc[:200]
        val   = df.iloc[150:250]   # overlap with train
        test  = df.iloc[250:]
        with pytest.raises(AssertionError, match="LEAKAGE"):
            leakage_check(train, val, test)

    def test_raises_on_temporal_overlap(self):
        df = _make_stories(300)
        train = df.iloc[:200]
        val   = df.iloc[180:250]   # timestamps overlap with train
        test  = df.iloc[250:]
        with pytest.raises(AssertionError, match="LEAKAGE"):
            leakage_check(train, val, test)

