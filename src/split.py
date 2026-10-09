"""
split.py — Chronological train/val/test splitter with leakage checker.

Rules
-----
* Splits are ALWAYS by date — no shuffling.
* A gap_days buffer between train end and val start prevents temporal
  autocorrelation leakage.
* The leakage_check() function asserts no story from the future appears
  in the training window (story_id level).

Public API
----------
    chronological_split(df, cfg) -> (train, val, test)
    leakage_check(train, val, test) -> None   # raises AssertionError on fail
"""
from __future__ import annotations

import logging
from datetime import timedelta

import pandas as pd

logger = logging.getLogger(__name__)


def chronological_split(
    df: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df into train / val / test by created_at timestamp.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'created_at' (tz-aware datetime).
    cfg : dict
        Master config.

    Returns
    -------
    (train, val, test) DataFrames sorted by created_at.
    """
    scfg = cfg["split"]
    gap  = timedelta(days=scfg["gap_days"])

    df = df.sort_values("created_at").reset_index(drop=True)

    n = len(df)
    train_end_idx = int(n * scfg["train_frac"])
    val_end_idx   = int(n * (scfg["train_frac"] + scfg["val_frac"]))

    train_end_ts = df.iloc[train_end_idx - 1]["created_at"]
    val_start_ts = train_end_ts + gap

    val_end_ts   = df.iloc[val_end_idx - 1]["created_at"]
    test_start_ts = val_end_ts + gap

    train = df[df["created_at"] <= train_end_ts].copy()
    val   = df[df["created_at"] >= val_start_ts].copy()
    val   = val[val["created_at"] <= val_end_ts].copy()
    test  = df[df["created_at"] >= test_start_ts].copy()

    logger.info(
        "Split: train=%d [%s → %s]  val=%d [%s → %s]  test=%d [%s → %s]",
        len(train), train["created_at"].min().date(), train["created_at"].max().date(),
        len(val),   val["created_at"].min().date(),   val["created_at"].max().date(),
        len(test),  test["created_at"].min().date(),  test["created_at"].max().date(),
    )
    return train, val, test


def leakage_check(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Assert temporal integrity of the split.

    Checks
    ------
    1. No story_id overlap between train/val/test.
    2. All train timestamps precede all val timestamps.
    3. All val timestamps precede all test timestamps.

    Raises
    ------
    AssertionError if any check fails.
    """
    train_ids = set(train["story_id"])
    val_ids   = set(val["story_id"])
    test_ids  = set(test["story_id"])

    overlap_tv = train_ids & val_ids
    overlap_tt = train_ids & test_ids
    overlap_vt = val_ids   & test_ids

    assert not overlap_tv, f"LEAKAGE: {len(overlap_tv)} story IDs in both train and val"
    assert not overlap_tt, f"LEAKAGE: {len(overlap_tt)} story IDs in both train and test"
    assert not overlap_vt, f"LEAKAGE: {len(overlap_vt)} story IDs in both val and test"

    train_max_ts = train["created_at"].max()
    val_min_ts   = val["created_at"].min()
    val_max_ts   = val["created_at"].max()
    test_min_ts  = test["created_at"].min()

    assert train_max_ts < val_min_ts, (
        f"LEAKAGE: train max ts {train_max_ts} >= val min ts {val_min_ts}"
    )
    assert val_max_ts < test_min_ts, (
        f"LEAKAGE: val max ts {val_max_ts} >= test min ts {test_min_ts}"
    )

    logger.info("Leakage check PASSED — no temporal or ID overlap detected.")
