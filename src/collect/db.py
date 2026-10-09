"""
db.py — Shared SQLite schema, connection factory, and path helpers.

All tables are defined here.  Every other module in src/collect imports
this module to get a connection — do not define schema elsewhere.

Schema overview
---------------
stories            — stories discovered via live Firebase polling
snapshots          — engagement snapshots taken at scheduled intervals
scheduled_snapshots — queue of pending snapshot jobs (resumable scheduler)
historical_stories — final-value records from Algolia (no trajectory data)

Data-source taxonomy
--------------------
  "newstories"   stories seen in /v0/newstories, tracked with snapshots
  "topstories"   seen only in /v0/topstories (already ranked; less useful)
  "algolia"      historical_stories pulled from Algolia Search API

Public API
----------
    get_connection(db_path) -> sqlite3.Connection
    init_schema(conn)       -> None
"""
import sqlite3
from pathlib import Path

DB_DEFAULT = Path("data") / "raw" / "hn_snapshots.db"

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ------------------------------------------------------------------ stories
CREATE TABLE IF NOT EXISTS stories (
    id                       INTEGER PRIMARY KEY,
    title                    TEXT    NOT NULL DEFAULT '',
    url                      TEXT    NOT NULL DEFAULT '',
    by                       TEXT    NOT NULL DEFAULT '',
    time                     INTEGER,           -- Unix epoch of original post
    type                     TEXT    NOT NULL DEFAULT 'story',
    source                   TEXT    NOT NULL,  -- 'newstories' | 'topstories'
    first_seen_at            REAL    NOT NULL,  -- Unix epoch when we first polled it
    is_dead                  INTEGER NOT NULL DEFAULT 0,
    is_deleted               INTEGER NOT NULL DEFAULT 0,
    score_at_discovery       INTEGER,
    descendants_at_discovery INTEGER
);

-- --------------------------------------------------------------- snapshots
CREATE TABLE IF NOT EXISTS snapshots (
    story_id         INTEGER NOT NULL,
    snapshot_time    REAL    NOT NULL,  -- Unix epoch
    hours_since_post REAL    NOT NULL,  -- actual elapsed time (may differ from target)
    score            INTEGER,           -- NULL when item was deleted at snapshot time
    descendants      INTEGER,
    is_dead          INTEGER NOT NULL DEFAULT 0,
    is_deleted       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (story_id, snapshot_time),
    FOREIGN KEY (story_id) REFERENCES stories(id)
);

-- -------------------------------------------------- scheduled_snapshots
-- The scheduler reads this table on startup to resume interrupted runs.
-- done: 0 = pending, 1 = success, 2 = permanently failed (>MAX_ATTEMPTS)
CREATE TABLE IF NOT EXISTS scheduled_snapshots (
    story_id        INTEGER NOT NULL,
    target_hours    REAL    NOT NULL,
    due_at          REAL    NOT NULL,   -- Unix epoch when snapshot should fire
    done            INTEGER NOT NULL DEFAULT 0,
    attempts        INTEGER NOT NULL DEFAULT 0,
    last_attempt_at REAL,
    PRIMARY KEY (story_id, target_hours),
    FOREIGN KEY (story_id) REFERENCES stories(id)
);

-- -------------------------------------------------- historical_stories
-- Algolia pull: final engagement values only (no trajectory).
-- Kept separate from 'stories' so analysts always know the provenance.
CREATE TABLE IF NOT EXISTS historical_stories (
    id           INTEGER PRIMARY KEY,   -- HN story ID
    title        TEXT    NOT NULL DEFAULT '',
    url          TEXT    NOT NULL DEFAULT '',
    by           TEXT    NOT NULL DEFAULT '',
    time         INTEGER,               -- Unix epoch of original post
    type         TEXT    NOT NULL DEFAULT 'story',
    points       INTEGER NOT NULL DEFAULT 0,
    num_comments INTEGER NOT NULL DEFAULT 0,
    fetched_at   REAL    NOT NULL,      -- Unix epoch when we pulled from Algolia
    data_source  TEXT    NOT NULL DEFAULT 'algolia'
);

-- ----------------------------------------------------------------- indices
CREATE INDEX IF NOT EXISTS idx_sched_due
    ON scheduled_snapshots(done, due_at);
CREATE INDEX IF NOT EXISTS idx_snapshots_story
    ON snapshots(story_id);
CREATE INDEX IF NOT EXISTS idx_stories_time
    ON stories(time);
CREATE INDEX IF NOT EXISTS idx_hist_time
    ON historical_stories(time);
"""


def get_connection(db_path: str | Path = DB_DEFAULT) -> sqlite3.Connection:
    """
    Open (or create) the SQLite database and return a connection.

    WAL mode is set so reads do not block the writer and vice-versa,
    which matters when inspecting the DB while the collector is running.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Apply the full schema DDL (idempotent — uses CREATE IF NOT EXISTS)."""
    conn.executescript(_SCHEMA)
    conn.commit()
