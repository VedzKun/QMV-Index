"""
ingest.py — Fetch Hacker News stories from the Algolia API.

Strategy
--------
* Fetch stories that are already >=72 h old (min_age_hours in config).
* At fetch time, points and 
um_comments are treated as the 72 h proxy
  values (target).  This is a known approximation — see model card.
* Raw JSON is persisted as Parquet in data/raw/ for reproducibility.

Public API
----------
    fetch_stories(cfg) -> pd.DataFrame
    save_raw(df, cfg) -> Path
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def _get(url: str, params: dict) -> dict[str, Any]:
    """GET with automatic retry / back-off."""
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _parse_story(hit: dict) -> dict:
    """Extract and normalise fields from a single Algolia hit."""
    created_at = hit.get("created_at")
    return {
        "story_id":       hit.get("objectID"),
        "title":          hit.get("title", ""),
        "url":            hit.get("url", ""),
        "author":         hit.get("author", ""),
        "created_at":     pd.to_datetime(created_at, utc=True) if created_at else pd.NaT,
        "points":         int(hit.get("points") or 0),
        "num_comments":   int(hit.get("num_comments") or 0),
        "story_text":     hit.get("story_text", ""),
    }


# ---------------------------------------------------------------------------
# Main fetcher
# ---------------------------------------------------------------------------

def fetch_stories(cfg: dict) -> pd.DataFrame:
    """
    Pull HN stories from Algolia Search API.

    Parameters
    ----------
    cfg : dict
        Full config loaded from config.yaml.

    Returns
    -------
    pd.DataFrame
        One row per story, with raw fields preserved.
    """
    dcfg = cfg["data"]
    base  = dcfg["hn_api_base"]
    tags  = dcfg["hn_tags"]
    batch = dcfg["fetch_batch_size"]
    max_p = dcfg["max_pages"]

    # Only stories old enough that the 72 h window has closed
    cutoff_ts = int(
        (datetime.now(timezone.utc) - timedelta(hours=dcfg["min_age_hours"])).timestamp()
    )

    start_ts = int(datetime.fromisoformat(dcfg["fetch_start_date"]).replace(tzinfo=timezone.utc).timestamp())
    end_ts   = int(datetime.fromisoformat(dcfg["fetch_end_date"]).replace(tzinfo=timezone.utc).timestamp())
    end_ts   = min(end_ts, cutoff_ts)

    records: list[dict] = []
    page = 0

    logger.info("Fetching HN stories %s → %s", dcfg["fetch_start_date"], dcfg["fetch_end_date"])

    while page < max_p:
        params = {
            "tags":           tags,
            "numericFilters": f"created_at_i>{start_ts},created_at_i<{end_ts}",
            "hitsPerPage":    batch,
            "page":           page,
        }
        data = _get(f"{base}/search_by_date", params)
        hits = data.get("hits", [])
        if not hits:
            break

        records.extend(_parse_story(h) for h in hits)
        logger.info("  Page %d — %d stories fetched so far", page, len(records))

        if page >= data.get("nbPages", 0) - 1:
            break
        page += 1
        time.sleep(0.25)  # polite rate limiting

    df = pd.DataFrame(records)
    logger.info("Total fetched: %d stories", len(df))
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_raw(df: pd.DataFrame, cfg: dict) -> Path:
    """
    Save raw DataFrame to data/raw/ as Parquet.

    Returns
    -------
    Path
        Path to the saved file.
    """
    raw_dir = Path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = raw_dir / f"hn_stories_{stamp}.parquet"
    df.to_parquet(out, index=False)
    logger.info("Raw data saved → %s (%d rows)", out, len(df))
    return out
