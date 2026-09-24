"""Unit tests for the atomic, fail-closed session store."""

from __future__ import annotations

import os
import stat
from datetime import UTC, datetime, timedelta

import pytest

from bruv.browser.models import BrowserGoal, SessionState
from bruv.browser.sessions import (
    MAX_STATE_FILE_BYTES,
    SessionStore,
    SessionStoreError,
)


def make_state(session_id: str = "s-test1") -> SessionState:
    return SessionState(
        session_id=session_id,
        status="running",
        goal=BrowserGoal(goal_id="g-1", goal_ref="goals/g-1", initial_url="https://example.com"),
        backend="default",
    )


@pytest.fixture()
def store(tmp_path):
    return SessionStore(tmp_path / "sessions")


def test_save_load_round_trip_and_permissions(tmp_path, store):
    state = make_state()
    store.save(state, supplied_texts=("secret goal text",))
    path = tmp_path / "sessions" / "s-test1.json"

    assert store.load("s-test1") == state
    assert not list(path.parent.glob("*.tmp"))
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_supplied_text_rejected_and_not_persisted(store):
    with pytest.raises(SessionStoreError):
        store.save(make_state(), supplied_texts=("s-test1",))
    assert not list(store.root.iterdir())


def test_supplied_text_heuristic_matches_string_values_not_keys(store):
    state = make_state()
    # Short word equal to a JSON field name is not a collision: only string
    # values are inspected, and "goal" != any serialized value.
    store.save(state, supplied_texts=("goal",))
    assert store.load("s-test1") == state
    assert store.latest() == state
    # Exact short value (regardless of length) is rejected.
    with pytest.raises(SessionStoreError):
        store.save(state, supplied_texts=("s-test1",))
    # Embedded >=8-char supplied text is rejected.
    with pytest.raises(SessionStoreError):
        store.save(state, supplied_texts=("https://example.com",))
    # Short (<8) non-matching words pass.
    store.save(state, supplied_texts=("red",))


def _write_raw(store, session_id: str, raw: bytes) -> None:
    (store.root / f"{session_id}.json").write_bytes(raw)


@pytest.mark.parametrize(
    ("session_id", "build"),
    [
        ("corrupt", lambda store: _write_raw(store, "s-x", b"not json\n")),
        (
            "noncanonical",
            lambda store: _write_raw(store, "s-x", make_state().model_dump_json().encode()),
        ),
        ("oversize", lambda store: _write_raw(store, "s-x", b"x" * (MAX_STATE_FILE_BYTES + 1))),
        (
            "id-mismatch",
            lambda store: _write_raw(
                store, "s-x", (make_state().model_dump_json(indent=2) + "\n").encode()
            ),
        ),
        (
            "symlink",
            lambda store: (
                store.save(make_state("s-real")),
                os.symlink(str(store.root / "s-real.json"), str(store.root / "s-x.json")),
            ),
        ),
        (
            "nonregular-dir",
            lambda store: (store.root / "s-x.json").mkdir(),
        ),
    ],
)
def test_malformed_state_fails_closed(store, session_id, build):
    build(store)
    with pytest.raises(SessionStoreError):
        store.load("s-x")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO only")
def test_fifo_state_fails_closed_without_blocking(store):
    os.mkfifo(store.root / "s-x.json")
    with pytest.raises(SessionStoreError):
        store.load("s-x")


def test_latest_and_list_order_deterministic(store):
    base = datetime.now(UTC)
    for offset, sid in ((2, "s-a"), (3, "s-b"), (1, "s-c")):
        state = make_state(sid)
        state = state.model_copy(update={"updated_at": base + timedelta(minutes=offset)})
        store.save(state)
    states = store.list_states()
    assert [s.session_id for s in states] == ["s-b", "s-a", "s-c"]
    assert store.latest() is not None
    assert store.latest().session_id == "s-b"


def test_list_fails_closed_on_any_bad_entry(store):
    store.save(make_state("s-a"))
    _write_raw(store, "s-b", b"{ broken")
    with pytest.raises(SessionStoreError):
        store.list_states()
