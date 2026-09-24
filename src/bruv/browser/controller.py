"""Browser session controller: owns the run loop over driver and executor.

Security model:

- Raw goal text and user reply text live only in controller process memory,
  never in persisted ``SessionState`` models, callbacks, exceptions, or the
  transcript. Replies are discarded immediately after the typing call.
- Only trusted candidate kinds map to fixed actions. The model never supplies
  URLs, selectors, payloads, or text; navigate targets resolve from the
  runtime observation's trusted targets and unknown or missing metadata fails
  closed.
- Review-gated candidates pause for a human and never execute. Text requests
  block the run thread on a condition variable while keeping the executor and
  session open; only a matching, nonblank reply resumes the run.
- Any driver, executor, malformed-candidate, limit, or unexpected failure
  transitions to ``failed`` with a category-only ``failure_reason`` (never an
  exception message or provider payload). ``KeyboardInterrupt`` and
  ``SystemExit`` are never swallowed. No retry loops around failures.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from typing import Final

from pydantic import ValidationError

from bruv.browser.actions import (
    MAX_SCROLL_AMOUNT,
    ClickAction,
    NavigateAction,
    ScrollAction,
    WaitAction,
)
from bruv.browser.driver import BrowserBackendDriver, BrowserDriverError
from bruv.browser.executor import BrowserError, BrowserExecutor
from bruv.browser.models import (
    BrowserGoal,
    BrowserStatus,
    ExecutedAction,
    PendingRequest,
    SessionState,
    TrustedCandidate,
)
from bruv.browser.observation import (
    SCROLL_DOWN_CANDIDATE_ID,
    SCROLL_UP_CANDIDATE_ID,
    WAIT_CANDIDATE_ID,
    RuntimeObservation,
)

# Fixed bounded parameters for synthetic actions; the executor clamps again.
_SCROLL_AMOUNT: Final[int] = MAX_SCROLL_AMOUNT
_WAIT_SECONDS: Final[float] = 2.0

# Category-only failure reasons. Never include exception messages or payloads.
_FAILURE_DEADLINE: Final[str] = "deadline_exceeded"
_FAILURE_MAX_STEPS: Final[str] = "max_steps_exceeded"
_FAILURE_DRIVER: Final[str] = "driver_error"
_FAILURE_EXECUTOR: Final[str] = "executor_error"
_FAILURE_ACTION: Final[str] = "invalid_action"
_FAILURE_CALLBACK: Final[str] = "callback_error"
_FAILURE_REPLY_TIMEOUT: Final[str] = "reply_deadline_exceeded"
_FAILURE_UNEXPECTED: Final[str] = "unexpected_error"

# Interval for wakeable condition waits while blocked on a reply.
_WAIT_POLL_SECONDS: Final[float] = 0.5

# Conservative cap on stored reply text; over-limit replies fail generically.
_MAX_REPLY_CHARS: Final[int] = 20_000


class BrowserControllerError(Exception):
    """Fail-closed controller error. Messages are text-safe."""


class _LimitBreach(Exception):
    """Internal limit enforcement signal carrying a safe category only."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _utc_now() -> datetime:
    return datetime.now(UTC)


class BrowserController:
    """Thread-owned run loop coordinating driver decisions and executor actions.

    Constructed with an unstarted ``BrowserExecutor``. ``run()`` must be called
    from a single thread; it owns ``executor.start/run/close``. Other threads
    may call ``stop()`` and ``reply()`` and may read ``state``.
    """

    def __init__(
        self,
        *,
        session_id: str,
        initial_url: str,
        goal_text: str,
        goal_ref: str,
        driver: BrowserBackendDriver,
        executor: BrowserExecutor,
        max_steps: int,
        time_limit_seconds: float | None = None,
        origin_allowlist: tuple[str, ...] = (),
        on_state: Callable[[SessionState], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if max_steps < 1:
            raise BrowserControllerError("max_steps must be at least 1")
        if time_limit_seconds is not None and time_limit_seconds <= 0:
            raise BrowserControllerError("time_limit_seconds must be positive when provided")
        self._session_id = session_id
        self._initial_url = initial_url
        # Memory only: the raw goal text never enters any persisted model.
        self._goal_text = goal_text
        self._driver = driver
        self._executor = executor
        self._max_steps = max_steps
        self._time_limit_seconds = time_limit_seconds
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self._on_state = on_state
        self._cond = threading.Condition()
        self._state = SessionState(
            session_id=session_id,
            status="running",
            goal=BrowserGoal(
                goal_id=f"goal-{uuid.uuid4().hex[:24]}",
                goal_ref=goal_ref,
                initial_url=initial_url,
            ),
            backend=driver.backend_name,
            origin_allowlist=tuple(origin_allowlist),
        )
        self._pending_request: PendingRequest | None = None
        self._executed: list[ExecutedAction] = []
        self._steps = 0
        self._stop = False
        self._reply_text: str | None = None
        self._run_started = False
        self._start_time: float | None = None

    # -- public interface ------------------------------------------------------

    @property
    def state(self) -> SessionState:
        """Thread-safe immutable snapshot of the current session state."""
        with self._cond:
            return self._state

    def run(self) -> SessionState:
        """Own the executor lifecycle and run the decide-act loop to a terminal state."""
        if self._run_started:
            raise BrowserControllerError("run() may only be called once per controller")
        self._run_started = True
        try:
            self._start_time = self._clock()
            self._executor.start()
            self._loop()
        except (KeyboardInterrupt, SystemExit):
            with suppress(Exception):
                self._finish("stopped")
            raise
        except _LimitBreach as exc:
            self._finish("failed", exc.reason)
        except BrowserDriverError:
            self._finish("failed", _FAILURE_DRIVER)
        except BrowserError:
            self._finish("failed", _FAILURE_EXECUTOR)
        except BrowserControllerError:
            self._finish("failed", _FAILURE_ACTION)
        except Exception:
            self._finish("failed", _FAILURE_UNEXPECTED)
        finally:
            with suppress(Exception):
                self._executor.close()
            self._attach_artifacts()
            self._emit()
        return self.state

    def stop(self) -> None:
        """Request a stop; idempotent. The run loop stops before the next provider call."""
        with self._cond:
            self._stop = True
            self._cond.notify_all()

    def reply(self, request_id: str, text: str) -> None:
        """Supply the reply for the current pending text request. Memory only.

        Only nonblank text for the currently pending request is accepted; the
        text is stored in memory, the state returns to ``running``, and the
        blocked run thread is woken. The text is never persisted or logged.
        """
        with self._cond:
            pending = self._pending_request
            if (
                self._state.status != "needs_text"
                or pending is None
                or pending.kind != "text"
                or request_id != pending.request_id
            ):
                raise BrowserControllerError("no matching pending text request")
            if not isinstance(text, str) or not text.strip():
                raise BrowserControllerError("reply text must be a nonblank string")
            if len(text) > _MAX_REPLY_CHARS:
                raise BrowserControllerError("reply text exceeds the allowed length")
            self._reply_text = text
            self._pending_request = None
            self._transition_locked(status="running", pending_request=None)
        self._emit()
        with self._cond:
            self._cond.notify_all()

    # -- run loop ---------------------------------------------------------------

    def _loop(self) -> None:
        while True:
            self._enforce_limits()
            with self._cond:
                if self._stop:
                    self._finish_locked("stopped")
                    return
            observation = self._executor.observe()
            self._record_observation(observation)
            if self._driver.goal_complete(self._goal_text, observation, tuple(self._executed)):
                self._finish("done")
                return
            candidate = self._driver.select_candidate(
                self._goal_text, observation, tuple(self._executed), observation.candidates
            )
            kind = candidate.kind
            if kind == "request_review" or candidate.needs_review_kind is not None:
                self._pause_for_review(observation, candidate)
                return
            if kind == "request_text":
                self._pause_for_text(observation, candidate)
                # _pause_for_text returns only after the reply was typed (it
                # marks failed/stopped internally otherwise) or the run was
                # stopped; resume deciding from a fresh observation.
                with self._cond:
                    if self._state.status in ("failed", "stopped"):
                        return
                continue
            action = self._build_action(observation, candidate)
            self._enforce_limits()
            self._execute_action(candidate, action)

    def _enforce_limits(self) -> None:
        if self._time_limit_seconds is not None:
            elapsed = self._clock() - self._start_monotonic()
            if elapsed >= self._time_limit_seconds:
                raise _LimitBreach(_FAILURE_DEADLINE)
        if self._steps >= self._max_steps:
            raise _LimitBreach(_FAILURE_MAX_STEPS)

    def _start_monotonic(self) -> float:
        if self._start_time is None:
            self._start_time = self._clock()
        return self._start_time

    def _build_action(
        self, observation: RuntimeObservation, candidate: TrustedCandidate
    ) -> ClickAction | NavigateAction | ScrollAction | WaitAction:
        """Map a trusted candidate to its fixed action. Unknown metadata fails closed."""
        kind = candidate.kind
        if kind == "click":
            return ClickAction(candidate_id=candidate.candidate_id)
        if kind == "navigate":
            target = observation.target_for(candidate.candidate_id)
            if target is None:
                raise BrowserControllerError("navigate candidate has no trusted target")
            try:
                return NavigateAction(candidate_id=candidate.candidate_id, url=target)
            except ValidationError:
                raise BrowserControllerError(
                    "navigate candidate target is not a safe URL"
                ) from None
        if kind == "scroll":
            if candidate.candidate_id == SCROLL_DOWN_CANDIDATE_ID:
                return ScrollAction(direction="down", amount=_SCROLL_AMOUNT)
            if candidate.candidate_id == SCROLL_UP_CANDIDATE_ID:
                return ScrollAction(direction="up", amount=_SCROLL_AMOUNT)
            raise BrowserControllerError("unknown scroll candidate")
        if kind == "wait":
            if candidate.candidate_id != WAIT_CANDIDATE_ID:
                raise BrowserControllerError("unknown wait candidate")
            return WaitAction(seconds=_WAIT_SECONDS)
        raise BrowserControllerError(f"candidate kind is not executable: {kind}")

    def _execute_action(
        self,
        candidate: TrustedCandidate,
        action: ClickAction | NavigateAction | ScrollAction | WaitAction,
    ) -> None:
        self._steps += 1
        self._executor.execute(action)
        executed = ExecutedAction(
            step=self._steps,
            kind=action.kind,
            description=f"{candidate.kind} candidate {candidate.candidate_id}",
            outcome="ok",
        )
        self._executed.append(executed)
        with self._cond:
            self._transition_locked(executed_actions=tuple(self._executed))
        self._emit()

    def _pause_for_review(
        self, observation: RuntimeObservation, candidate: TrustedCandidate
    ) -> None:
        review_kind = candidate.needs_review_kind
        if review_kind is None:
            raise BrowserControllerError("review request is missing its review kind")
        request = PendingRequest(
            request_id=f"req-{uuid.uuid4().hex[:24]}",
            kind="review",
            prompt_ref=f"obs:{observation.summary.observation_id}",
            candidate_id=candidate.candidate_id,
            review_kind=review_kind,
        )
        with self._cond:
            self._pending_request = request
            self._transition_locked(status="needs_review", pending_request=request)
        self._emit()

    def _pause_for_text(self, observation: RuntimeObservation, candidate: TrustedCandidate) -> None:
        """Persist the text request, block for the reply, then type it once."""
        request = PendingRequest(
            request_id=f"req-{uuid.uuid4().hex[:24]}",
            kind="text",
            prompt_ref=f"obs:{observation.summary.observation_id}",
            candidate_id=candidate.candidate_id,
        )
        with self._cond:
            self._pending_request = request
            self._transition_locked(status="needs_text", pending_request=request)
        self._emit()
        text = self._await_reply(request)
        if text is None:
            return  # stopped or failed while waiting; already recorded.
        self._steps += 1
        try:
            self._executor.type_text(candidate.candidate_id, text)
        finally:
            del text  # discard the in-memory reply immediately after use
        self._executed.append(
            ExecutedAction(
                step=self._steps,
                kind="type",
                description=f"typed into candidate {candidate.candidate_id}",
                outcome="ok",
            )
        )
        with self._cond:
            self._transition_locked(executed_actions=tuple(self._executed))
        self._emit()

    def _await_reply(self, request: PendingRequest) -> str | None:
        """Block for a reply under the deadline. None means stopped or timed out."""
        with self._cond:
            while self._reply_text is None and not self._stop:
                if self._time_limit_seconds is not None:
                    remaining = self._time_limit_seconds - (self._clock() - self._start_monotonic())
                    if remaining <= 0:
                        self._pending_request = None
                        self._finish_locked("failed", failure_reason=_FAILURE_REPLY_TIMEOUT)
                        return None
                    self._cond.wait(min(remaining, _WAIT_POLL_SECONDS))
                else:
                    self._cond.wait(_WAIT_POLL_SECONDS)
            if self._reply_text is None:
                # Stop requested while blocked.
                self._pending_request = None
                self._finish_locked("stopped")
                return None
            text = self._reply_text
            self._reply_text = None
            return text

    # -- state bookkeeping ---------------------------------------------------------

    def _record_observation(self, observation: RuntimeObservation) -> None:
        """Attach the bounded summary only; runtime page text is never persisted."""
        with self._cond:
            self._transition_locked(observation=observation.summary)
        self._emit()

    def _attach_artifacts(self) -> None:
        artifacts = self._executor.session_artifacts
        if artifacts is None:
            return
        with self._cond:
            self._transition_locked(artifacts=artifacts)

    def _finish(self, status: BrowserStatus, failure_reason: str | None = None) -> None:
        with self._cond:
            self._finish_locked(status, failure_reason)
        self._emit()

    def _finish_locked(self, status: BrowserStatus, failure_reason: str | None = None) -> None:
        self._pending_request = None
        self._transition_locked(status=status, failure_reason=failure_reason, pending_request=None)

    def _transition_locked(self, **updates: object) -> None:
        self._state = self._state.model_copy(update={**updates, "updated_at": _utc_now()})

    def _emit(self) -> None:
        """Publish the current snapshot. Callback failures fail closed."""
        if self._on_state is None:
            return
        with self._cond:
            snapshot = self._state
        try:
            self._on_state(snapshot)
        except Exception:
            if snapshot.status == "failed":
                return  # never loop on a failing callback during failure publication
            self._finish("failed", _FAILURE_CALLBACK)


__all__ = ["BrowserController", "BrowserControllerError"]
