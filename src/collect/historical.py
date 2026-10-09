"""
historical.py — Algolia HN Search API bootstrap for historical data.

Purpose
-------
Provides a large corpus of stories with FINAL engagement values
(points + comments as returned by Algolia at fetch time) to enable
pre-training and validation before live snapshot data accumulates.

Data provenance
---------------
All records written here carry data_source='algolia' in the
historical_stories table.  This table is intentionally SEPARATE from
the 'stories' + 'snapshots' tables so that downstream code always knows:

  historical_stories  →  final value only, no trajectory
  stories + snapshots →  full engagement trajectory

Public API
----------
    parse_algolia_hit(hit)                -> dict | None
    fetch_historical(cfg, conn, dry_run)  -> int
    save_historical(conn, records)        -> int
"""
from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.collect.db import get_connection, init_schema

logger = logging.getLogger(__name__)

ALGOLIA_BASE   = "https://hn.algolia.com/api/v1"
REQUEST_DELAY_S = 0.12   # ~8 req/s — well within Algolia free tier
MAX_PAGES       = 100    # safety cap per date-range chunk


# ---------------------------------------------------------------------------
# Parsing  (pure — no side effects)
# ---------------------------------------------------------------------------

def parse_algolia_hit(hit: dict) -> Optional[dict]:
    """
    Convert an Algolia search result hit into a historical_stories row dict.

    Returns None if the hit is missing a valid objectID or post time.
    Logs a WARNING for each skipped hit so analysts can audit data loss.
    """
    oid = hit.get("objectID")
    if not oid:
        logger.warning("Skipping hit with no objectID: %s", hit)
        return None

    # created_at_i is Unix epoch; fall back to parsing created_at string
    post_ts = hit.get("created_at_i")
    if post_ts is None:
        created_str = hit.get("created_at")
        if created_str:
            try:
                post_ts = int(
                    datetime.fromisoformat(created_str.rstrip("Z"))
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                )
            except ValueError:
                logger.warning("Unparseable created_at for %s: %r", oid, created_str)
                post_ts = None

    points      = hit.get("points")
    num_comments = hit.get("num_comments")

    return {
        "id":           int(oid),
        "title":        hit.get("title")  or "",
        "url":          hit.get("url")    or "",
        "by":           hit.get("author") or "",
        "time":         post_ts,
        "type":         "story",
        "points":       int(points)       if points      is not None else 0,
        "num_comments": int(num_comments) if num_comments is not None else 0,
        "fetched_at":   time.time(),
        "data_source":  "algolia",
    }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type((requests.HTTPError, requests.ConnectionError,
                                   requests.Timeout)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    reraise=True,
)
def _fetch_page(session: requests.Session, params: dict) -> dict:
    resp = session.get(f"{ALGOLIA_BASE}/search_by_date", params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------

def save_historical(conn: sqlite3.Connection, records: list[dict]) -> int:
    """
    Upsert historical story records.

    Uses INSERT OR IGNORE so repeated runs are idempotent.

    Returns
    -------
    int  Number of rows newly inserted (ignores already-present rows).
    """
    inserted = 0
    for rec in records:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO historical_stories
                (id, title, url, by, time, type, points, num_comments,
                 fetched_at, data_source)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rec["id"],   rec["title"],  rec["url"],   rec["by"],
                rec["time"], rec["type"],   rec["points"], rec["num_comments"],
                rec["fetched_at"], rec["data_source"],
            ),
        )
        inserted += cursor.rowcount
    conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def fetch_historical(
    cfg: dict,
    conn: sqlite3.Connection,
    dry_run: bool = False,
) -> int:
    """
    Pull historical HN stories from Algolia and store them.

    Iterates over the date range in config.yaml in monthly chunks to avoid
    hitting Algolia pagination limits (max 1000 results per query window).

    Parameters
    ----------
    cfg      : master config dict
    conn     : open SQLite connection with schema already applied
    dry_run  : if True, fetch and parse but do not write to DB

    Returns
    -------
    int  Total stories inserted.
    """
    dcfg     = cfg["data"]
    session  = requests.Session()
    session.headers.update({"User-Agent": "QMV-Index-Historical/0.1"})

    start_dt = datetime.fromisoformat(dcfg["fetch_start_date"]).replace(tzinfo=timezone.utc)
    end_dt   = datetime.fromisoformat(dcfg["fetch_end_date"]).replace(tzinfo=timezone.utc)

    total_inserted = 0

    # Chunk by month to stay within Algolia pagination limits
    chunk_start = start_dt
    while chunk_start < end_dt:
        # End of this chunk = next month (approx 31 days)
        from dateutil.relativedelta import relativedelta  # type: ignore
        chunk_end = min(chunk_start + relativedelta(months=1), end_dt)

        start_ts = int(chunk_start.timestamp())
        end_ts   = int(chunk_end.timestamp())

        logger.info("Fetching %s → %s", chunk_start.date(), chunk_end.date())

        page = 0
        while page < MAX_PAGES:
            params = {
                "tags":           dcfg.get("hn_tags", "story"),
                "numericFilters": f"created_at_i>{start_ts},created_at_i<={end_ts}",
                "hitsPerPage":    dcfg.get("fetch_batch_size", 1000),
                "page":           page,
            }
            try:
                data = _fetch_page(session, params)
            except Exception as exc:
                logger.error("Page fetch failed for chunk %s p%d: %s",
                             chunk_start.date(), page, exc)
                break

            hits    = data.get("hits", [])
            nb_pages = data.get("nbPages", 0)

            records = [r for h in hits if (r := parse_algolia_hit(h)) is not None]
            logger.info("  page %d/%d — %d/%d hits parsed",
                        page + 1, nb_pages, len(records), len(hits))

            if not dry_run and records:
                inserted = save_historical(conn, records)
                total_inserted += inserted
                logger.debug("  inserted %d new rows (of %d parsed)", inserted, len(records))

            if page >= nb_pages - 1:
                break
            page += 1
            time.sleep(REQUEST_DELAY_S)

        chunk_start = chunk_end

    logger.info("Historical fetch complete — %d total rows inserted", total_inserted)
    return total_inserted


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    from src import load_config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="QMV Historical Bootstrap — pull HN stories from Algolia."
    )
    parser.add_argument("--db",      default=None, help="SQLite DB path")
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and log but do not write to DB.")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    from src.collect.db import DB_DEFAULT
    db_path = Path(args.db) if args.db else DB_DEFAULT
    conn = get_connection(db_path)
    init_schema(conn)

    fetch_historical(cfg, conn, dry_run=args.dry_run)
    conn.close()


if __name__ == "__main__":
    main()
