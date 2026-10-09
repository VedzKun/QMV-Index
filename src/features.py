"""
features.py — Pure feature engineering functions.

Design rules
------------
* Every function is PURE — no side effects, no globals.
* Every feature must only use information available at prediction time T.
* Snapshot features (points_at_T, comments_at_T) default to 0 when unavailable.
* Functions are small and individually testable.

Feature groups
--------------
A. Temporal        — posting time, day-of-week, cyclical encoding
B. Title           — length, word count, sentiment signals, keyword flags
C. Domain          — TLD, is_github, is_pdf, top-K domain encoding
D. Author          — log(karma), account age (if available)
E. Snapshot (T h)  — early engagement counts (0 when not collected)

Public API
----------
    build_features(df, cfg, horizon_h) -> pd.DataFrame
"""
from __future__ import annotations

import re
import logging
from typing import Sequence
from urllib.parse import urlparse

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# A. Temporal features
# ---------------------------------------------------------------------------

def temporal_features(df: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
    """
    Add cyclical time encodings and calendar features.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'created_at' (tz-aware datetime).
    tz : str
        Target timezone (default UTC).

    Returns
    -------
    pd.DataFrame
        New columns: hour_sin, hour_cos, dow_sin, dow_cos,
                     is_weekend, month, quarter.
    """
    ts = df["created_at"].dt.tz_convert(tz)

    hour = ts.dt.hour
    dow  = ts.dt.dayofweek  # 0=Mon, 6=Sun

    out = df.copy()
    out["hour_sin"]  = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"]  = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"]   = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"]   = np.cos(2 * np.pi * dow / 7)
    out["is_weekend"] = (dow >= 5).astype(int)
    out["month"]     = ts.dt.month
    out["quarter"]   = ts.dt.quarter
    return out


# ---------------------------------------------------------------------------
# B. Title features
# ---------------------------------------------------------------------------

_QUESTION_RE  = re.compile(r"\?")
_ASK_HN_RE    = re.compile(r"^ask hn", re.IGNORECASE)
_SHOW_HN_RE   = re.compile(r"^show hn", re.IGNORECASE)
_TELL_HN_RE   = re.compile(r"^tell hn", re.IGNORECASE)
_NUMBER_RE    = re.compile(r"\b\d+\b")
_BRACKET_RE   = re.compile(r"\[.*?\]|\(.*?\)")


def title_features(df: pd.DataFrame, max_len: int = 200) -> pd.DataFrame:
    """
    Derive features from the story title string.

    Returns
    -------
    pd.DataFrame
        New columns: title_len, title_word_count, title_has_question,
                     title_is_ask_hn, title_is_show_hn, title_is_tell_hn,
                     title_has_number, title_has_brackets, title_upper_ratio.
    """
    t = df["title"].str[:max_len].fillna("")
    out = df.copy()
    out["title_len"]          = t.str.len()
    out["title_word_count"]   = t.str.split().str.len().fillna(0).astype(int)
    out["title_has_question"] = t.str.contains(_QUESTION_RE).astype(int)
    out["title_is_ask_hn"]    = t.str.contains(_ASK_HN_RE).astype(int)
    out["title_is_show_hn"]   = t.str.contains(_SHOW_HN_RE).astype(int)
    out["title_is_tell_hn"]   = t.str.contains(_TELL_HN_RE).astype(int)
    out["title_has_number"]   = t.str.contains(_NUMBER_RE).astype(int)
    out["title_has_brackets"] = t.str.contains(_BRACKET_RE).astype(int)
    upper = t.str.count(r"[A-Z]")
    total = t.str.len().replace(0, np.nan)
    out["title_upper_ratio"]  = (upper / total).fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# C. Domain features
# ---------------------------------------------------------------------------

def _extract_domain(url: str) -> str:
    """Extract registered domain (e.g. 'github.com') from URL."""
    if not url or not url.startswith("http"):
        return "__text__"
    try:
        return urlparse(url).netloc.lstrip("www.")
    except Exception:
        return "__unknown__"


def domain_features(df: pd.DataFrame, top_k: int = 200,
                    domain_vocab: Sequence[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """
    Add domain-based features.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'url'.
    top_k : int
        Number of top domains to one-hot encode (training mode).
    domain_vocab : list[str] | None
        Pre-computed vocab (inference mode — ensures same columns as train).

    Returns
    -------
    (pd.DataFrame, list[str])
        DataFrame with new columns; list of domain vocab used.
    """
    out = df.copy()
    out["domain"] = out["url"].apply(_extract_domain)

    out["is_github"]  = (out["domain"] == "github.com").astype(int)
    out["is_pdf"]     = out["url"].str.lower().str.endswith(".pdf").fillna(False).astype(int)
    out["is_youtube"] = out["domain"].isin({"youtube.com", "youtu.be"}).astype(int)
    out["is_text_post"] = (out["domain"] == "__text__").astype(int)

    if domain_vocab is None:
        counts = out["domain"].value_counts()
        domain_vocab = counts.head(top_k).index.tolist()

    for d in domain_vocab:
        col = f"domain__{d.replace('.', '_').replace('-', '_')}"
        out[col] = (out["domain"] == d).astype(int)

    return out, list(domain_vocab)


# ---------------------------------------------------------------------------
# D. Author features
# ---------------------------------------------------------------------------

def author_features(df: pd.DataFrame, log_transform: bool = True) -> pd.DataFrame:
    """
    Add author-level features.

    Currently only uses 'author' (string). If author karma is available
    in a separate lookup table, merge it before calling this function
    and pass a DataFrame that already has an 'author_karma' column.

    Returns
    -------
    pd.DataFrame
        New columns: is_deleted_author, author_karma_log (if karma available).
    """
    out = df.copy()
    out["is_deleted_author"] = (out["author"] == "[deleted]").astype(int)

    if "author_karma" in df.columns:
        karma = df["author_karma"].clip(lower=0).fillna(0)
        if log_transform:
            out["author_karma_log"] = np.log1p(karma)
        else:
            out["author_karma_log"] = karma

    return out


# ---------------------------------------------------------------------------
# E. Snapshot features (early engagement at horizon T)
# ---------------------------------------------------------------------------

def snapshot_features(df: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    """
    Add engagement snapshot features for prediction horizon T=horizon_h.

    If columns 'points_at_{horizon_h}h' and 'comments_at_{horizon_h}h'
    exist in df, derive ratio and log features from them.
    Otherwise fill with zeros (T=0 / metadata-only mode).

    Parameters
    ----------
    df : pd.DataFrame
    horizon_h : int
        Prediction horizon in hours (1, 3, or 6).

    Returns
    -------
    pd.DataFrame
        New columns: snap_points, snap_comments, snap_engagement,
                     snap_points_log, snap_comments_log,
                     snap_comment_to_point_ratio.
    """
    out = df.copy()
    pt_col  = f"points_at_{horizon_h}h"
    cm_col  = f"comments_at_{horizon_h}h"

    snap_points   = df[pt_col].clip(lower=0).fillna(0)  if pt_col  in df.columns else pd.Series(0, index=df.index)
    snap_comments = df[cm_col].clip(lower=0).fillna(0)  if cm_col  in df.columns else pd.Series(0, index=df.index)

    out[f"snap_points_T{horizon_h}"]   = snap_points
    out[f"snap_comments_T{horizon_h}"] = snap_comments
    out[f"snap_engagement_T{horizon_h}"] = snap_points + 2 * snap_comments
    out[f"snap_points_log_T{horizon_h}"]   = np.log1p(snap_points)
    out[f"snap_comments_log_T{horizon_h}"] = np.log1p(snap_comments)

    denom = snap_points.replace(0, np.nan)
    out[f"snap_c2p_ratio_T{horizon_h}"] = (snap_comments / denom).fillna(0.0)

    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, cfg: dict, horizon_h: int,
                   domain_vocab: list[str] | None = None
                   ) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply all feature groups and return the final feature DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned story data (output of clean.clean()).
    cfg : dict
        Master config.
    horizon_h : int
        Prediction horizon (1, 3, or 6 hours).
    domain_vocab : list[str] | None
        Pre-computed domain vocab (pass for val/test to prevent leakage).

    Returns
    -------
    (pd.DataFrame, list[str])
        Feature DataFrame + domain vocab used.
    """
    fcfg = cfg["features"]

    df = temporal_features(df, tz=fcfg["time"]["tz"])
    df = title_features(df, max_len=fcfg["title"]["max_len"])
    df, vocab = domain_features(df, top_k=fcfg["domain"]["top_k"],
                                domain_vocab=domain_vocab)
    df = author_features(df, log_transform=fcfg["author"]["karma_log_transform"])
    df = snapshot_features(df, horizon_h=horizon_h)

    logger.info("Features built: %d columns for T=%dh horizon", len(df.columns), horizon_h)
    return df, vocab
