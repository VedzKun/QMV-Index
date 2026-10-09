"""
clean.py — Load raw Parquet, deduplicate, type-cast, add basic quality flags.

Public API
----------
    load_raw(cfg) -> pd.DataFrame
    clean(df, cfg) -> pd.DataFrame
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_raw(cfg: dict) -> pd.DataFrame:
    """Read all Parquet files from data/raw/ and concatenate."""
    raw_dir = Path(cfg["data"]["raw_dir"])
    files = sorted(raw_dir.glob("hn_stories_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw Parquet files found in {raw_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    logger.info("Loaded %d rows from %d file(s)", len(df), len(files))
    return df


def clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Clean raw HN story DataFrame.

    Steps
    -----
    1. Deduplicate on story_id (keep first).
    2. Drop rows with null story_id or created_at.
    3. Cast types.
    4. Add quality flag columns (is_text_only, is_suspicious).
    5. Compute target column.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_raw().
    cfg : dict
        Master config.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    n_raw = len(df)

    # 1. Deduplicate
    df = df.drop_duplicates(subset="story_id", keep="first").copy()
    logger.info("Deduplication: %d -> %d rows", n_raw, len(df))

    # 2. Drop nulls in required columns
    df = df.dropna(subset=["story_id", "created_at"])

    # 3. Type coercion
    df["story_id"]    = df["story_id"].astype(str)
    df["points"]      = pd.to_numeric(df["points"], errors="coerce").fillna(0).astype(int)
    df["num_comments"]= pd.to_numeric(df["num_comments"], errors="coerce").fillna(0).astype(int)
    df["created_at"]  = pd.to_datetime(df["created_at"], utc=True)
    df["title"]       = df["title"].fillna("").astype(str)
    df["url"]         = df["url"].fillna("").astype(str)
    df["author"]      = df["author"].fillna("[deleted]").astype(str)

    # 4. Quality flags
    df["is_text_only"]   = df["url"].str.strip() == ""
    df["is_suspicious"]  = (df["points"] == 0) & (df["num_comments"] == 0)

    # 5. Target
    pw = cfg["data"]["points_weight"]
    cw = cfg["data"]["comments_weight"]
    df["target"] = np.log1p(pw * df["points"] + cw * df["num_comments"])

    logger.info("Clean complete: %d rows, target range [%.2f, %.2f]",
                len(df), df["target"].min(), df["target"].max())
    return df.reset_index(drop=True)
