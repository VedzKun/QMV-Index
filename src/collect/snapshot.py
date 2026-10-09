"""
snapshot.py — Live HN Firebase poller and snapshot scheduler.

Architecture
------------
1. poll_new_stories()  — calls /v0/newstories.json, registers unseen stories,
                         inserts scheduled_snapshot rows for each target horizon.
2. process_due_snapshots() — reads scheduled_snapshots WHERE done=0 AND due_at<=now,
                             fetches /v0/item/{id}.json, writes to snapshots table.
3. run_loop()          — alternates between 1 and 2, sleeping LOOP_SLEEP_S between ticks.

Resumability
------------
On restart the scheduler simply re-reads scheduled_snapshots WHERE done=0.
All inserts use INSERT OR IGNORE, so re-discovering the same story is safe.

Entry point
-----------
    python -m src.collect.snapshot --run [--db PATH] [--poll-interval 300]

Public (testable) API
---------------------
    parse_item(raw)                                  -> dict | None
    fetch_new_story_ids(session)                     -> list[int]
    fetch_item(session, item_id)                     -> dict | None
    register_story(conn, item, source)               -> bool
    create_schedule(conn, story_id, post_time, ...)  -> None
    get_due_snapshots(conn, now_ts, limit)           -> list[dict]
    record_snapshot(conn, story_id, snap_time, item) -> None
    mark_schedule_done(conn, story_id, target_h, done) -> None
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from src.collect.db import DB_DEFAULT, get_connection, init_schema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIREBASE_BASE   = "https://hacker-news.firebaseio.com/v0"
TARGET_HOURS    = [0.25, 0.5, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0, 72.0]
POLL_INTERVAL_S = 300   # seconds between /newstories polls
LOOP_SLEEP_S    = 60    # seconds to sleep at end of each main-loop tick
MAX_ATTEMPTS    = 5     # give up on a snapshot after this many failures
ITEM_DELAY_S    = 0.10  # polite delay between consecutive item fetches


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "QMV-Index-Collector/0.1"})
    return session


@retry(
    retry=retry_if_exception_type((requests.HTTPError, requests.ConnectionError,
                                   requests.Timeout)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _get_json(session: requests.Session, url: str) -> Optional[dict]:
    """GET url and return parsed JSON, or None for 404 / null body."""
    resp = session.get(url, timeout=15)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()  # Firebase returns literal null for deleted items → None


# ---------------------------------------------------------------------------
# Item fetching
# ---------------------------------------------------------------------------

def fetch_new_story_ids(session: requests.Session) -> list[int]:
    """
    Return list of story IDs from /v0/newstories.json (newest 500).
    Returns empty list on failure.
    """
    try:
        data = _get_json(session, f"{FIREBASE_BASE}/newstories.json")
        return data if isinstance(data, list) else []
    except Exception as exc:
        logger.error("fetch_new_story_ids failed: %s", exc)
        return []


def fetch_item(session: requests.Session, item_id: int) -> Optional[dict]:
    """
    Return raw Firebase item dict for item_id, or None if deleted/not found.
    """
    try:
        return _get_json(session, f"{FIREBASE_BASE}/item/{item_id}.json")
    except Exception as exc:
        logger.warning("fetch_item(%d) failed: %s", item_id, exc)
        return None


# ---------------------------------------------------------------------------
# Parsing  (pure functions — no side effects, fully testable)
# ---------------------------------------------------------------------------

def parse_item(raw: Optional[dict]) -> Optional[dict]:
    """
    Normalise a raw Firebase item dict.

    Returns
    -------
    dict  with keys: id, title, url, by, time, type, score, descendants,
                     is_dead, is_deleted
    None  if the item was deleted (raw is None or missing 'id').
    """
    if not raw or "id" not in raw:
        return None
    return {
        "id":          int(raw["id"]),
        "title":       raw.get("title") or "",
        "url":         raw.get("url")   or "",
        "by":          raw.get("by")    or "",
        "time":        raw.get("time"),           # may be None for very old items
        "type":        raw.get("type")  or "story",
        "score":       int(raw.get("score") or 0),
        "descendants": int(raw.get("descendants") or 0),
        "is_dead":     int(bool(raw.get("dead",    False))),
        "is_deleted":  int(bool(raw.get("deleted", False))),
    }


# ---------------------------------------------------------------------------
# DB operations
# ---------------------------------------------------------------------------

def register_story(
    conn: sqlite3.Connection,
    item: dict,
    source: str = "newstories",
) -> bool:
    """
    Insert a story into the stories table if not already present.

    Returns
    -------
    True  if this is a new story (inserted).
    False if the story was already known (INSERT OR IGNORE was a no-op).
    """
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO stories
            (id, title, url, by, time, type, source, first_seen_at,
             is_dead, is_deleted, score_at_discovery, descendants_at_discovery)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            item["id"], item["title"], item["url"], item["by"],
            item["time"], item["type"], source, time.time(),
            item["is_dead"], item["is_deleted"],
            item["score"], item["descendants"],
        ),
    )
    conn.commit()
    return cursor.rowcount == 1


def create_schedule(
    conn: sqlite3.Connection,
    story_id: int,
    post_time: float,
    target_hours: list[float] = TARGET_HOURS,
) -> None:
    """
    Insert pending snapshot jobs for story_id at each target horizon.

    Uses INSERT OR IGNORE so calling this twice for the same story is safe
    (restart-safe: existing pending rows are untouched).

    Parameters
    ----------
    post_time : float
        Unix epoch of the story's original post time.
    """
    rows = [
        (story_id, h, post_time + h * 3600.0)
        for h in target_hours
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO scheduled_snapshots(story_id, target_hours, due_at) VALUES (?,?,?)",
        rows,
    )
    conn.commit()


def get_due_snapshots(
    conn: sqlite3.Connection,
    now_ts: float,
    limit: int = 200,
) -> list[dict]:
    """
    Return up to `limit` pending snapshot jobs whose due_at <= now_ts,
    ordered by due_at ascending (oldest overdue first).

    Returns list of plain dicts (not sqlite3.Row) for easy testing.
    """
    rows = conn.execute(
        """
        SELECT ss.story_id, ss.target_hours, ss.due_at, ss.attempts,
               s.time AS post_time
        FROM   scheduled_snapshots ss
        JOIN   stories s ON s.id = ss.story_id
        WHERE  ss.done = 0
          AND  ss.due_at <= ?
          AND  ss.attempts < ?
        ORDER  BY ss.due_at ASC
        LIMIT  ?
        """,
        (now_ts, MAX_ATTEMPTS, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def record_snapshot(
    conn: sqlite3.Connection,
    story_id: int,
    snapshot_time: float,
    hours_since_post: float,
    item: Optional[dict],
) -> None:
    """
    Write one snapshot row.  item=None means the story was deleted at poll time.
    """
    if item is None:
        conn.execute(
            """
            INSERT OR IGNORE INTO snapshots
                (story_id, snapshot_time, hours_since_post,
                 score, descendants, is_dead, is_deleted)
            VALUES (?,?,?, NULL,NULL, 0,1)
            """,
            (story_id, snapshot_time, hours_since_post),
        )
    else:
        conn.execute(
            """
            INSERT OR IGNORE INTO snapshots
                (story_id, snapshot_time, hours_since_post,
                 score, descendants, is_dead, is_deleted)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                story_id, snapshot_time, hours_since_post,
                item["score"], item["descendants"],
                item["is_dead"], item["is_deleted"],
            ),
        )
    conn.commit()


def mark_schedule_done(
    conn: sqlite3.Connection,
    story_id: int,
    target_hours: float,
    done: int = 1,
) -> None:
    """Set done=1 (success) or done=2 (permanent failure)."""
    conn.execute(
        "UPDATE scheduled_snapshots SET done=? WHERE story_id=? AND target_hours=?",
        (done, story_id, target_hours),
    )
    conn.commit()


def increment_attempts(
    conn: sqlite3.Connection,
    story_id: int,
    target_hours: float,
) -> None:
    """Increment attempt counter and record timestamp."""
    conn.execute(
        """
        UPDATE scheduled_snapshots
        SET attempts = attempts + 1, last_attempt_at = ?
        WHERE story_id = ? AND target_hours = ?
        """,
        (time.time(), story_id, target_hours),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# High-level operations
# ---------------------------------------------------------------------------

def poll_new_stories(
    conn: sqlite3.Connection,
    session: requests.Session,
) -> int:
    """
    Fetch /newstories, register any new stories, and create their schedules.

    Returns
    -------
    int  Number of newly registered stories.
    """
    story_ids = fetch_new_story_ids(session)
    logger.info("poll_new_stories: %d IDs from /newstories", len(story_ids))
    new_count = 0

    for sid in story_ids:
        # Skip if already tracked
        existing = conn.execute(
            "SELECT 1 FROM stories WHERE id = ?", (sid,)
        ).fetchone()
        if existing:
            continue

        time.sleep(ITEM_DELAY_S)
        raw = fetch_item(session, sid)
        item = parse_item(raw)

        if item is None:
            logger.debug("Story %d deleted/null at discovery — skipping", sid)
            continue
        if item["type"] != "story":
            continue  # jobs, polls, comments — not relevant

        is_new = register_story(conn, item, source="newstories")
        if is_new and item["time"]:
            create_schedule(conn, item["id"], float(item["time"]))
            new_count += 1
            logger.info("New story registered: %d  '%s'", item["id"], item["title"][:60])

    logger.info("poll_new_stories: %d new stories registered", new_count)
    return new_count


def process_due_snapshots(
    conn: sqlite3.Connection,
    session: requests.Session,
) -> int:
    """
    Take all overdue snapshots and write results to the snapshots table.

    Returns
    -------
    int  Number of snapshots successfully taken.
    """
    now_ts = time.time()
    due = get_due_snapshots(conn, now_ts)
    logger.info("process_due_snapshots: %d jobs due", len(due))
    success = 0

    for job in due:
        sid          = job["story_id"]
        target_h     = job["target_hours"]
        post_time    = job["post_time"]
        lateness_min = (now_ts - job["due_at"]) / 60

        if lateness_min > 120:
            logger.warning(
                "Story %d T=%.2fh snapshot is %.0f min overdue — recording late",
                sid, target_h, lateness_min,
            )

        increment_attempts(conn, sid, target_h)
        time.sleep(ITEM_DELAY_S)

        raw  = fetch_item(session, sid)
        item = parse_item(raw)
        snap_time       = time.time()
        hours_since     = (snap_time - post_time) / 3600.0 if post_time else target_h

        try:
            record_snapshot(conn, sid, snap_time, hours_since, item)
            mark_schedule_done(conn, sid, target_h, done=1)
            success += 1
            logger.debug("Snapshot OK: story=%d T=%.2fh actual=%.2fh", sid, target_h, hours_since)
        except Exception as exc:
            logger.error("record_snapshot failed story=%d T=%.2fh: %s", sid, target_h, exc)
            attempts = job["attempts"] + 1
            if attempts >= MAX_ATTEMPTS:
                mark_schedule_done(conn, sid, target_h, done=2)
                logger.error("Story %d T=%.2fh permanently failed after %d attempts",
                             sid, target_h, attempts)

    return success


def run_loop(
    db_path: str | Path = DB_DEFAULT,
    poll_interval_s: int = POLL_INTERVAL_S,
    loop_sleep_s: int = LOOP_SLEEP_S,
) -> None:
    """
    Main collection loop.  Runs until interrupted (KeyboardInterrupt / SIGTERM).

    Polling cadence
    ---------------
    - Every poll_interval_s seconds: fetch /newstories and register new items.
    - Every loop_sleep_s seconds: process any overdue snapshot jobs.
    """
    conn    = get_connection(db_path)
    init_schema(conn)
    session = _make_session()

    logger.info("Snapshot collector starting. DB: %s", db_path)
    last_poll_ts = 0.0

    try:
        while True:
            now = time.time()

            # Poll for new stories on interval
            if now - last_poll_ts >= poll_interval_s:
                poll_new_stories(conn, session)
                last_poll_ts = time.time()

            # Always try to process due snapshots
            process_due_snapshots(conn, session)

            logger.debug("Sleeping %ds …", loop_sleep_s)
            time.sleep(loop_sleep_s)

    except KeyboardInterrupt:
        logger.info("Collector stopped by user.")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="QMV HN Snapshot Collector — polls Firebase and stores engagement data."
    )
    parser.add_argument("--run", action="store_true", required=True,
                        help="Start the collection loop.")
    parser.add_argument("--db", default=str(DB_DEFAULT),
                        help=f"Path to SQLite DB (default: {DB_DEFAULT})")
    parser.add_argument("--poll-interval", type=int, default=POLL_INTERVAL_S,
                        help="Seconds between /newstories polls (default: 300)")
    args = parser.parse_args()

    run_loop(
        db_path=args.db,
        poll_interval_s=args.poll_interval,
    )


if __name__ == "__main__":
    main()
