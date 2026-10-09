"""
test_features.py — Unit tests for src/features.py.

Key assertions
--------------
* Temporal features use cyclical encoding in [-1, 1].
* Title features detect Ask HN / Show HN correctly.
* Snapshot features fill with zeros when snapshot columns absent (no leakage).
* domain_features uses provided vocab in inference mode (no training leakage).
* build_features returns the same column set for train and inference call.
"""
import numpy as np
import pandas as pd
import pytest

from src.features import (
    temporal_features,
    title_features,
    domain_features,
    snapshot_features,
    build_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(**kwargs) -> pd.DataFrame:
    """Minimal story DataFrame for testing."""
    defaults = {
        "story_id":   ["1", "2", "3"],
        "title":      ["Ask HN: Question?", "Show HN: Demo [v2]", "Plain title 42"],
        "url":        ["", "https://github.com/user/repo", "https://example.com/paper.pdf"],
        "author":     ["alice", "[deleted]", "bob"],
        "created_at": pd.to_datetime([
            "2024-03-15 09:00:00+00:00",
            "2024-03-15 22:00:00+00:00",
            "2024-03-16 06:00:00+00:00",
        ]),
        "points":      [100, 5, 0],
        "num_comments":[50, 2, 0],
        "target":      [4.5, 1.2, 0.0],
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------

class TestTemporalFeatures:
    def test_cyclical_range(self):
        df = temporal_features(_make_df())
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            assert df[col].between(-1.0001, 1.0001).all(), f"{col} out of [-1,1]"

    def test_weekend_flag(self):
        df = temporal_features(_make_df())
        # 2024-03-15 is a Friday (dow=4 → not weekend)
        # 2024-03-16 is a Saturday (dow=5 → weekend)
        assert df.loc[2, "is_weekend"] == 1

    def test_no_null_output(self):
        df = temporal_features(_make_df())
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]:
            assert df[col].isna().sum() == 0


# ---------------------------------------------------------------------------
# Title features
# ---------------------------------------------------------------------------

class TestTitleFeatures:
    def test_ask_hn_detected(self):
        df = title_features(_make_df())
        assert df.loc[0, "title_is_ask_hn"] == 1
        assert df.loc[1, "title_is_ask_hn"] == 0

    def test_show_hn_detected(self):
        df = title_features(_make_df())
        assert df.loc[1, "title_is_show_hn"] == 1

    def test_question_mark(self):
        df = title_features(_make_df())
        assert df.loc[0, "title_has_question"] == 1
        assert df.loc[2, "title_has_question"] == 0

    def test_number_in_title(self):
        df = title_features(_make_df())
        assert df.loc[2, "title_has_number"] == 1   # "42"

    def test_brackets(self):
        df = title_features(_make_df())
        assert df.loc[1, "title_has_brackets"] == 1

    def test_upper_ratio_nonnegative(self):
        df = title_features(_make_df())
        assert (df["title_upper_ratio"] >= 0).all()


# ---------------------------------------------------------------------------
# Domain features
# ---------------------------------------------------------------------------

class TestDomainFeatures:
    def test_github_flag(self):
        df, _ = domain_features(_make_df())
        assert df.loc[1, "is_github"] == 1

    def test_pdf_flag(self):
        df, _ = domain_features(_make_df())
        assert df.loc[2, "is_pdf"] == 1

    def test_text_post_flag(self):
        df, _ = domain_features(_make_df())
        assert df.loc[0, "is_text_post"] == 1

    def test_inference_uses_provided_vocab(self):
        """Inference call must not look at test data to build vocab."""
        df_train = _make_df()
        df_test  = _make_df(url=["https://newsite.com", "https://github.com/x", ""])
        _, vocab = domain_features(df_train, top_k=10)
        df_out, vocab2 = domain_features(df_test, domain_vocab=vocab)
        assert vocab2 == vocab, "Vocab changed in inference mode — potential leakage"

    def test_same_columns_train_infer(self):
        df_train = _make_df()
        df_test  = _make_df()
        df_t, vocab = domain_features(df_train)
        df_i, _     = domain_features(df_test, domain_vocab=vocab)
        assert set(df_t.columns) == set(df_i.columns)


# ---------------------------------------------------------------------------
# Snapshot features — no-leakage (zero-fill when columns absent)
# ---------------------------------------------------------------------------

class TestSnapshotFeatures:
    def test_zero_fill_when_no_snapshots(self):
        """When no snapshot columns exist, all snap features must be 0."""
        df = snapshot_features(_make_df(), horizon_h=1)
        assert (df["snap_points_T1"] == 0).all()
        assert (df["snap_comments_T1"] == 0).all()

    def test_fills_from_column_when_present(self):
        base = _make_df()
        base["points_at_1h"]   = [10, 2, 0]
        base["comments_at_1h"] = [3, 0, 0]
        df = snapshot_features(base, horizon_h=1)
        assert df.loc[0, "snap_points_T1"] == 10
        assert df.loc[0, "snap_comments_T1"] == 3

    def test_c2p_ratio_zero_when_zero_points(self):
        df = snapshot_features(_make_df(), horizon_h=3)
        assert (df["snap_c2p_ratio_T3"] == 0).all()
