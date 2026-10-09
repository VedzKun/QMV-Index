"""
test_collect.py — Tests for src/collect/ modules.

Strategy
--------
* parse_item and parse_algolia_hit: pure-function tests, no mocking needed.
* DB operations (register_story, create_schedule, get_due_snapshots):
  use in-memory SQLite (:memory:) — fast, isolated, no disk I/O.
* fetch_new_story_ids / fetch_item: mock requests.Session with MagicMock.
* run_loop is NOT tested here (integration concern; would need threading).
"""
from __future__ import annotations

import sqlite3
import time
from unittest.mock import MagicMock, call, patch

import pytest

from src.collect.db import get_connection, init_schema
from src.collect.snapshot import (
    MAX_ATTEMPTS,
    TARGET_HOURS,
    create_schedule,
    fetch_item,
    fetch_new_story_ids,
    get_due_snapshots,
    increment_attempts,
    mark_schedule_done,
    parse_item,
    record_snapshot,
    register_story,
)
from src.collect.historical import parse_algolia_hit, save_historical


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mem_conn():
    """Fresh in-memory SQLite DB with full schema applied."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    return conn


def _make_raw(**kwargs) -> dict:
    """Minimal valid Firebase item dict."""
    defaults = {
        "id":          42,
        "title":       "Ask HN: How does LightGBM handle missing values?",
        "url":         "",
        "by":          "alice",
        "time":        1_700_000_000,
        "type":        "story",
        "score":       17,
        "descendants": 5,
    }
    defaults.update(kwargs)
    return defaults


def _make_session(responses: dict) -> MagicMock:
    """
    Build a mock requests.Session where .get(url) returns json from responses dict.
    """
    session = MagicMock()

    def _get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = responses.get(url)
        resp.raise_for_status.return_value = None
        return resp

    session.get.side_effect = _get
    return session


# ===========================================================================
# A. parse_item — pure function tests
# ===========================================================================

class TestParseItem:
    def test_normal_item(self):
        raw = _make_raw()
        item = parse_item(raw)
        assert item is not None
        assert item["id"] == 42
        assert item["title"] == raw["title"]
        assert item["score"] == 17
        assert item["descendants"] == 5
        assert item["is_dead"] == 0
        assert item["is_deleted"] == 0

    def test_deleted_item_returns_none(self):
        """Firebase returns null (None in Python) for deleted items."""
        assert parse_item(None) is None

    def test_missing_id_returns_none(self):
        assert parse_item({"title": "no id here"}) is None

    def test_dead_item_flagged(self):
        item = parse_item(_make_raw(dead=True))
        assert item is not None
        assert item["is_dead"] == 1

    def test_explicitly_deleted_flagged(self):
        item = parse_item(_make_raw(deleted=True))
        assert item is not None
        assert item["is_deleted"] == 1

    def test_missing_score_defaults_to_zero(self):
        raw = _make_raw()
        del raw["score"]
        item = parse_item(raw)
        assert item["score"] == 0

    def test_missing_descendants_defaults_to_zero(self):
        raw = _make_raw()
        del raw["descendants"]
        item = parse_item(raw)
        assert item["descendants"] == 0

    def test_none_score_defaults_to_zero(self):
        """Algolia sometimes returns null for score."""
        item = parse_item(_make_raw(score=None))
        assert item["score"] == 0

    def test_url_normalised_to_empty_string(self):
        """Text posts have no url; None should become empty string."""
        item = parse_item(_make_raw(url=None))
        assert item["url"] == ""

    def test_non_story_type_preserved(self):
        item = parse_item(_make_raw(type="job"))
        assert item["type"] == "job"


# ===========================================================================
# B. parse_algolia_hit — pure function tests
# ===========================================================================

class TestParseAlgoliaHit:
    def _hit(self, **kwargs) -> dict:
        defaults = {
            "objectID":    "12345",
            "title":       "Show HN: My project",
            "url":         "https://example.com",
            "author":      "bob",
            "created_at_i": 1_700_000_000,
            "points":      42,
            "num_comments": 10,
        }
        defaults.update(kwargs)
        return defaults

    def test_normal_hit(self):
        rec = parse_algolia_hit(self._hit())
        assert rec is not None
        assert rec["id"] == 12345
        assert rec["points"] == 42
        assert rec["num_comments"] == 10
        assert rec["data_source"] == "algolia"

    def test_missing_object_id_returns_none(self):
        assert parse_algolia_hit({}) is None

    def test_none_points_defaults_to_zero(self):
        rec = parse_algolia_hit(self._hit(points=None))
        assert rec["points"] == 0

    def test_none_num_comments_defaults_to_zero(self):
        rec = parse_algolia_hit(self._hit(num_comments=None))
        assert rec["num_comments"] == 0

    def test_author_mapped_correctly(self):
        """Algolia uses 'author', not 'by'."""
        rec = parse_algolia_hit(self._hit(author="charlie"))
        assert rec["by"] == "charlie"

    def test_fallback_to_created_at_string(self):
        """When created_at_i is absent, parse created_at ISO string."""
        hit = self._hit()
        del hit["created_at_i"]
        hit["created_at"] = "2023-11-14T06:13:20.000Z"
        rec = parse_algolia_hit(hit)
        assert rec is not None
        assert rec["time"] == 1_699_942_400  # may vary slightly; just check non-None
        assert rec["time"] is not None

    def test_missing_url_is_empty_string(self):
        rec = parse_algolia_hit(self._hit(url=None))
        assert rec["url"] == ""


# ===========================================================================
# C. DB operations — schedule creation and due-snapshot queries
# ===========================================================================

class TestRegisterStory:
    def test_new_story_returns_true(self, mem_conn):
        item = parse_item(_make_raw())
        assert register_story(mem_conn, item) is True

    def test_duplicate_story_returns_false(self, mem_conn):
        item = parse_item(_make_raw())
        register_story(mem_conn, item)
        assert register_story(mem_conn, item) is False

    def test_row_stored_correctly(self, mem_conn):
        item = parse_item(_make_raw())
        register_story(mem_conn, item)
        row = mem_conn.execute("SELECT * FROM stories WHERE id=42").fetchone()
        assert row is not None
        assert row["title"] == item["title"]
        assert row["source"] == "newstories"


class TestCreateSchedule:
    def _register(self, conn):
        item = parse_item(_make_raw())
        register_story(conn, item)
        return item

    def test_creates_correct_number_of_rows(self, mem_conn):
        item = self._register(mem_conn)
        create_schedule(mem_conn, item["id"], float(item["time"]))
        count = mem_conn.execute(
            "SELECT COUNT(*) FROM scheduled_snapshots WHERE story_id=42"
        ).fetchone()[0]
        assert count == len(TARGET_HOURS)

    def test_due_at_correct(self, mem_conn):
        item = self._register(mem_conn)
        post_time = float(item["time"])
        create_schedule(mem_conn, item["id"], post_time, target_hours=[1.0, 3.0])
        rows = mem_conn.execute(
            "SELECT target_hours, due_at FROM scheduled_snapshots WHERE story_id=42"
            " ORDER BY target_hours"
        ).fetchall()
        assert rows[0]["due_at"] == pytest.approx(post_time + 1.0 * 3600)
        assert rows[1]["due_at"] == pytest.approx(post_time + 3.0 * 3600)

    def test_idempotent_on_duplicate_call(self, mem_conn):
        item = self._register(mem_conn)
        create_schedule(mem_conn, item["id"], float(item["time"]), target_hours=[1.0])
        create_schedule(mem_conn, item["id"], float(item["time"]), target_hours=[1.0])
        count = mem_conn.execute(
            "SELECT COUNT(*) FROM scheduled_snapshots WHERE story_id=42"
        ).fetchone()[0]
        assert count == 1  # second call was a no-op

    def test_initial_state_is_pending(self, mem_conn):
        item = self._register(mem_conn)
        create_schedule(mem_conn, item["id"], float(item["time"]), target_hours=[6.0])
        row = mem_conn.execute(
            "SELECT done, attempts FROM scheduled_snapshots WHERE story_id=42"
        ).fetchone()
        assert row["done"] == 0
        assert row["attempts"] == 0


class TestGetDueSnapshots:
    def _setup(self, conn, due_offset_s: float) -> dict:
        """Register a story and create a 1h schedule entry due in `due_offset_s` seconds."""
        item = parse_item(_make_raw())
        register_story(conn, item)
        post_time = float(item["time"])
        # Manually insert a schedule row with a specific due_at
        due_at = time.time() + due_offset_s
        conn.execute(
            "INSERT OR IGNORE INTO scheduled_snapshots(story_id, target_hours, due_at)"
            " VALUES (?,?,?)",
            (item["id"], 1.0, due_at),
        )
        conn.commit()
        return {"story_id": item["id"], "due_at": due_at, "post_time": post_time}

    def test_overdue_job_is_returned(self, mem_conn):
        self._setup(mem_conn, due_offset_s=-10)  # 10s in the past
        due = get_due_snapshots(mem_conn, now_ts=time.time())
        assert len(due) == 1
        assert due[0]["story_id"] == 42

    def test_future_job_not_returned(self, mem_conn):
        self._setup(mem_conn, due_offset_s=3600)  # 1h in the future
        due = get_due_snapshots(mem_conn, now_ts=time.time())
        assert len(due) == 0

    def test_done_job_excluded(self, mem_conn):
        info = self._setup(mem_conn, due_offset_s=-10)
        mem_conn.execute(
            "UPDATE scheduled_snapshots SET done=1 WHERE story_id=?",
            (info["story_id"],),
        )
        mem_conn.commit()
        due = get_due_snapshots(mem_conn, now_ts=time.time())
        assert len(due) == 0

    def test_max_attempts_exceeded_excluded(self, mem_conn):
        info = self._setup(mem_conn, due_offset_s=-10)
        mem_conn.execute(
            "UPDATE scheduled_snapshots SET attempts=? WHERE story_id=?",
            (MAX_ATTEMPTS, info["story_id"]),
        )
        mem_conn.commit()
        due = get_due_snapshots(mem_conn, now_ts=time.time())
        assert len(due) == 0

    def test_returns_dicts_not_rows(self, mem_conn):
        self._setup(mem_conn, due_offset_s=-1)
        due = get_due_snapshots(mem_conn, now_ts=time.time())
        assert isinstance(due[0], dict)


class TestRecordSnapshot:
    def _setup(self, conn):
        item = parse_item(_make_raw())
        register_story(conn, item)
        return item

    def test_writes_snapshot_row(self, mem_conn):
        item = self._setup(mem_conn)
        snap_time = time.time()
        record_snapshot(mem_conn, item["id"], snap_time, 1.0, item)
        row = mem_conn.execute(
            "SELECT * FROM snapshots WHERE story_id=?", (item["id"],)
        ).fetchone()
        assert row is not None
        assert row["score"] == item["score"]
        assert row["hours_since_post"] == pytest.approx(1.0)

    def test_deleted_item_writes_null_score(self, mem_conn):
        item = self._setup(mem_conn)
        snap_time = time.time()
        record_snapshot(mem_conn, item["id"], snap_time, 1.0, item=None)
        row = mem_conn.execute(
            "SELECT * FROM snapshots WHERE story_id=?", (item["id"],)
        ).fetchone()
        assert row["score"] is None
        assert row["is_deleted"] == 1


# ===========================================================================
# D. fetch_new_story_ids and fetch_item — mocked HTTP
# ===========================================================================

class TestFetchNewStoryIds:
    def test_returns_list(self):
        ids = [1, 2, 3, 4]
        url = "https://hacker-news.firebaseio.com/v0/newstories.json"
        session = _make_session({url: ids})
        result = fetch_new_story_ids(session)
        assert result == ids

    def test_returns_empty_on_none_response(self):
        url = "https://hacker-news.firebaseio.com/v0/newstories.json"
        session = _make_session({url: None})
        result = fetch_new_story_ids(session)
        assert result == []


class TestFetchItem:
    def test_returns_parsed_dict(self):
        raw = _make_raw()
        url = f"https://hacker-news.firebaseio.com/v0/item/{raw['id']}.json"
        session = _make_session({url: raw})
        result = fetch_item(session, raw["id"])
        assert result == raw

    def test_returns_none_for_deleted(self):
        url = "https://hacker-news.firebaseio.com/v0/item/99.json"
        session = _make_session({url: None})
        result = fetch_item(session, 99)
        assert result is None


# ===========================================================================
# E. save_historical — DB round-trip
# ===========================================================================

class TestSaveHistorical:
    def _rec(self, **kwargs) -> dict:
        defaults = {
            "id": 999, "title": "Test", "url": "https://t.co",
            "by": "user", "time": 1_700_000_000, "type": "story",
            "points": 50, "num_comments": 20,
            "fetched_at": time.time(), "data_source": "algolia",
        }
        defaults.update(kwargs)
        return defaults

    def test_inserts_new_record(self, mem_conn):
        inserted = save_historical(mem_conn, [self._rec()])
        assert inserted == 1

    def test_idempotent_on_duplicate(self, mem_conn):
        rec = self._rec()
        save_historical(mem_conn, [rec])
        inserted = save_historical(mem_conn, [rec])
        assert inserted == 0  # INSERT OR IGNORE was a no-op

    def test_data_persisted_correctly(self, mem_conn):
        rec = self._rec(points=77, num_comments=33)
        save_historical(mem_conn, [rec])
        row = mem_conn.execute(
            "SELECT * FROM historical_stories WHERE id=999"
        ).fetchone()
        assert row["points"] == 77
        assert row["num_comments"] == 33
        assert row["data_source"] == "algolia"
