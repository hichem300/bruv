"""Browser driver: bounded canonical decisions over an existing bruv backend.

Fail-closed. One Noul question decides goal completion; canonical Choice calls
select action candidates through recursive tournament selection. Raw goal text
and page text live only in request state held in process memory; nothing here
persists, and no navigation targets, raw selectors, supplied replies, secrets,
or input values ever enter a request.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Final

from pydantic import ValidationError

from bruv.application import ApplicationError, DecisionFacade
from bruv.backends.factory import ClientFactory, create_backend
from bruv.backends.interface import DecisionBackend
from bruv.backends.registry import browser_choice_batch_sizes, browser_supports_noul_choice
from bruv.browser.models import ExecutedAction, TrustedCandidate
from bruv.browser.observation import RuntimeObservation
from bruv.browser.selection import SelectionError, select_candidate
from bruv.config import AppConfig
from bruv.domain.questions import ChoiceQuestion, NoulQuestion
from bruv.domain.requests import DecisionRequest
from bruv.domain.results import (
    ABSTAIN_ANSWER_ID,
    AbstainAnswer,
    Answer,
    ChoiceAnswer,
    DecisionResult,
    NoulAnswer,
)
from bruv.domain.validation import BackendCapabilities
from bruv.onboarding.credentials import Credentials

# Explicit conservative caps applied inside the driver even though the
# observation builder already bounds its own output.
_MAX_GOAL_CHARS: Final[int] = 2000
_MAX_PAGE_TEXT_CHARS: Final[int] = 6000
_MAX_TITLE_CHARS: Final[int] = 512
_MAX_URL_CHARS: Final[int] = 2048
_MAX_STATE_CANDIDATES: Final[int] = 40
_MAX_RECENT_ACTIONS: Final[int] = 10

_GOAL_QUESTION_ID: Final[str] = "goal_complete"
_FALLBACK_QUESTION_ID: Final[str] = "goal_complete_fallback"
_BATCH_QUESTION_ID: Final[str] = "choose_candidate"

_COMPLETE_ID: Final[str] = "complete"
_CONTINUE_ID: Final[str] = "continue"

_GOAL_NOUL_INSTRUCTIONS: Final[str] = (
    "Decide whether the user's goal is fully complete on the current page. "
    "Answer true only when nothing remains to do to reach the goal."
)
_GOAL_NOUL_TRUE: Final[str] = "The goal is fully complete on this page."
_GOAL_NOUL_FALSE: Final[str] = "The goal is not complete; more actions are needed."
_FALLBACK_INSTRUCTIONS: Final[str] = (
    "Decide whether the user's goal is complete on the current page."
)
_FALLBACK_COMPLETE: Final[str] = "The goal is complete; stop."
_FALLBACK_CONTINUE: Final[str] = "More actions are needed; continue."
_BATCH_INSTRUCTIONS: Final[str] = (
    "Choose the single best next action for the goal from the candidate options."
)


class BrowserDriverError(Exception):
    """Fail-closed browser driver error. Messages are text-safe."""


def _clamp(value: str, limit: int) -> str:
    collapsed = " ".join(value.split())
    return collapsed[:limit]


def _require_single_answer(result: DecisionResult, question_id: str) -> Answer:
    answers = result.answers
    if set(answers) != {question_id}:
        raise BrowserDriverError(
            f"backend returned a malformed result for question {question_id!r}"
        )
    return answers[question_id]


class BrowserBackendDriver:
    """Canonical Noul/Choice decisions for the browser over one bruv backend."""

    def __init__(self, *, backend: DecisionBackend, config: AppConfig) -> None:
        capabilities = backend.capabilities
        if capabilities.backend != config.backend:
            raise BrowserDriverError(
                "backend capabilities do not match configured backend: "
                f"{capabilities.backend!r} != {config.backend!r}"
            )
        if not browser_supports_noul_choice(capabilities):
            raise BrowserDriverError(
                f"backend {capabilities.backend!r} does not answer canonical "
                "noul and choice questions; browser driving is unsupported"
            )
        self._backend = backend
        self._config = config
        self._capabilities: BackendCapabilities = capabilities
        self._facade = DecisionFacade(backend)
        self._batch_sizes: tuple[int, ...] = browser_choice_batch_sizes(capabilities)

    @classmethod
    def create(
        cls,
        config: AppConfig,
        credentials: Credentials,
        *,
        client_factory: ClientFactory | None = None,
    ) -> BrowserBackendDriver:
        """Build the configured backend through the registry composition root.

        ``client_factory`` is the existing optional selected-dependency factory
        used by tests; there is no parallel provider abstraction here.
        """
        backend = create_backend(config, credentials, client_factory=client_factory)
        return cls(backend=backend, config=config)

    # -- safe metadata -------------------------------------------------------

    @property
    def backend_name(self) -> str:
        return self._capabilities.backend

    @property
    def calibrated(self) -> bool:
        return self._capabilities.calibrated

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    # -- request state -------------------------------------------------------

    def _build_state(
        self,
        goal_text: str,
        observation: RuntimeObservation,
        recent_actions: Sequence[ExecutedAction],
    ) -> dict[str, object]:
        summary = observation.summary
        bounded_descriptions = [candidate.description for candidate in observation.candidates][
            :_MAX_STATE_CANDIDATES
        ]
        # Most recent actions first in bounded order.
        recent = [
            {
                "step": action.step,
                "kind": action.kind,
                "description": _clamp(action.description, 200),
                "outcome": action.outcome,
            }
            for action in recent_actions[-_MAX_RECENT_ACTIONS:]
        ]
        state: dict[str, object] = {
            "goal": _clamp(goal_text, _MAX_GOAL_CHARS),
            "page": {
                "url": _clamp(summary.url, _MAX_URL_CHARS),
                "title": _clamp(summary.title, _MAX_TITLE_CHARS),
                "text": observation.page_text[:_MAX_PAGE_TEXT_CHARS],
            },
            "candidates": bounded_descriptions,
            "recent_actions": recent,
        }
        return state

    def _request(
        self,
        question_id: str,
        question: ChoiceQuestion | NoulQuestion,
        goal_text: str,
        observation: RuntimeObservation,
        recent_actions: Sequence[ExecutedAction],
    ) -> DecisionRequest:
        state: object = self._build_state(goal_text, observation, recent_actions)
        if not self._capabilities.allows_json_state:
            # Same bounded state, deterministically serialized for text-only backends.
            state = json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        try:
            request = DecisionRequest(state=state, questions={question_id: question})
        except ValidationError as exc:
            raise BrowserDriverError("canonical request construction failed") from exc
        return request

    def _evaluate(self, request: DecisionRequest) -> DecisionResult:
        try:
            result = self._facade.evaluate(request)
        except ApplicationError as exc:
            raise BrowserDriverError(f"backend evaluation failed: {exc.code}") from exc
        except ValidationError as exc:
            raise BrowserDriverError("backend returned a malformed result") from exc
        except Exception as exc:
            raise BrowserDriverError("backend evaluation failed") from exc
        if not isinstance(result, DecisionResult):
            raise BrowserDriverError("backend returned a non-canonical result")
        if result.backend != self._capabilities.backend:
            raise BrowserDriverError("backend result reports a mismatched backend name")
        if result.calibrated != self._capabilities.calibrated:
            raise BrowserDriverError(
                "backend calibration claim does not match its declared capabilities"
            )
        return result

    # -- goal completion -----------------------------------------------------

    def goal_complete(
        self,
        goal_text: str,
        observation: RuntimeObservation,
        recent_actions: Sequence[ExecutedAction],
    ) -> bool:
        question = NoulQuestion(
            instructions=_GOAL_NOUL_INSTRUCTIONS,
            criteria={"true": _GOAL_NOUL_TRUE, "false": _GOAL_NOUL_FALSE},
        )
        request = self._request(_GOAL_QUESTION_ID, question, goal_text, observation, recent_actions)
        result = self._evaluate(request)
        answer = _require_single_answer(result, _GOAL_QUESTION_ID)
        if isinstance(answer, AbstainAnswer):
            raise BrowserDriverError("backend abstained on goal completion")
        if not isinstance(answer, NoulAnswer):
            raise BrowserDriverError("backend returned a wrong answer type for goal completion")
        if answer.value is not None:
            return answer.value
        # Probability-only Noul: threshold only when both calibrations hold;
        # otherwise exactly one canonical two-option fallback, no retry.
        if answer.noul is not None:
            if result.calibrated and self._capabilities.calibrated:
                return answer.noul >= 0.5
            return self._goal_complete_fallback(goal_text, observation, recent_actions)
        raise BrowserDriverError("backend returned an empty noul answer")

    def _goal_complete_fallback(
        self,
        goal_text: str,
        observation: RuntimeObservation,
        recent_actions: Sequence[ExecutedAction],
    ) -> bool:
        question = ChoiceQuestion(
            instructions=_FALLBACK_INSTRUCTIONS,
            criteria={_COMPLETE_ID: _FALLBACK_COMPLETE, _CONTINUE_ID: _FALLBACK_CONTINUE},
        )
        request = self._request(
            _FALLBACK_QUESTION_ID, question, goal_text, observation, recent_actions
        )
        result = self._evaluate(request)
        answer = _require_single_answer(result, _FALLBACK_QUESTION_ID)
        if isinstance(answer, AbstainAnswer):
            raise BrowserDriverError("backend abstained on the goal-completion fallback")
        if not isinstance(answer, ChoiceAnswer):
            raise BrowserDriverError(
                "backend returned a wrong answer type for the goal-completion fallback"
            )
        if answer.choice == _COMPLETE_ID:
            return True
        if answer.choice == _CONTINUE_ID:
            return False
        raise BrowserDriverError("backend returned an unknown fallback choice")

    # -- candidate selection ---------------------------------------------------

    def choose_batch(
        self,
        goal_text: str,
        observation: RuntimeObservation,
        recent_actions: Sequence[ExecutedAction],
        batch: Sequence[TrustedCandidate],
    ) -> str:
        candidates = tuple(batch)
        ids = [candidate.candidate_id for candidate in candidates]
        if not candidates or len(set(ids)) != len(ids):
            raise BrowserDriverError("candidate batch must be non-empty with unique IDs")
        if len(candidates) not in self._batch_sizes:
            raise BrowserDriverError(
                f"candidate batch size {len(candidates)} is not supported by "
                f"backend {self._capabilities.backend!r}"
            )
        question = ChoiceQuestion(
            instructions=_BATCH_INSTRUCTIONS,
            criteria={
                candidate.candidate_id: _clamp(candidate.description, 200)
                for candidate in candidates
            },
        )
        request = self._request(
            _BATCH_QUESTION_ID, question, goal_text, observation, recent_actions
        )
        result = self._evaluate(request)
        answer = _require_single_answer(result, _BATCH_QUESTION_ID)
        if isinstance(answer, AbstainAnswer):
            raise BrowserDriverError("backend abstained on candidate selection")
        if not isinstance(answer, ChoiceAnswer):
            raise BrowserDriverError("backend returned a wrong answer type for candidate selection")
        if answer.choice == ABSTAIN_ANSWER_ID:
            raise BrowserDriverError("backend abstained on candidate selection")
        if answer.choice not in set(ids):
            raise BrowserDriverError("backend returned a candidate ID outside the presented batch")
        return answer.choice

    def select_candidate(
        self,
        goal_text: str,
        observation: RuntimeObservation,
        recent_actions: Sequence[ExecutedAction],
        candidates: Sequence[TrustedCandidate],
    ) -> TrustedCandidate:
        """Delegate the recursive tournament; exactly one final winner returns."""
        try:
            return select_candidate(
                tuple(candidates),
                self._capabilities,
                lambda batch: self.choose_batch(goal_text, observation, recent_actions, batch),
            )
        except SelectionError as exc:
            raise BrowserDriverError(f"candidate selection failed: {exc}") from exc


__all__ = ["BrowserBackendDriver", "BrowserDriverError"]
