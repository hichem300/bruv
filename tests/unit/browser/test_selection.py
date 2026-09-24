"""Tournament selection tests: exactly-one winner, fail-closed, RLCD totals."""

from __future__ import annotations

import pytest

from bruv.backends.registry import browser_choice_batch_sizes
from bruv.browser.models import ElementFingerprint, TrustedCandidate
from bruv.browser.selection import BatchChoice, SelectionError, select_candidate
from bruv.domain.results import ABSTAIN_ANSWER_ID
from bruv.domain.validation import BackendCapabilities

RLCD_CAPS = BackendCapabilities(
    backend="rlcd",
    question_types=frozenset({"noul", "choice"}),
    calibrated=True,
    allows_json_state=True,
    explicit_abstention=True,
    supported_total_candidates=frozenset({2, 3, 4, 5, 6, 7, 9, 11, 17, 25}),
)


def cand(n: int) -> TrustedCandidate:
    return TrustedCandidate(
        candidate_id=f"c{n}",
        kind="click",
        fingerprint=ElementFingerprint(
            fingerprint_id=f"f{n}",
            tag="button",
            text_length=1,
            attributes_hash="a",
        ),
        description=f"button {n}",
    )


def pick(answer: str, calls: list[tuple[TrustedCandidate, ...]]) -> BatchChoice:
    def choose(batch: tuple[TrustedCandidate, ...], /) -> str:
        calls.append(batch)
        return answer

    return choose


def test_tournament_returns_exactly_one_final_candidate() -> None:
    calls: list[tuple[TrustedCandidate, ...]] = []

    def choose(batch: tuple[TrustedCandidate, ...], /) -> str:
        calls.append(batch)
        # Wide batches keep the first candidate; head-to-head keeps the last.
        return batch[0].candidate_id if len(batch) > 2 else batch[-1].candidate_id

    # 7 candidates: supported sizes <=7 partition into [6, 1], forcing round 2.
    winner = select_candidate(tuple(cand(i) for i in range(7)), RLCD_CAPS, choose)
    assert winner.candidate_id in {f"c{i}" for i in range(7)}
    assert len(calls) >= 2  # multiple rounds happened
    assert all(len(batch) >= 2 for batch in calls)


def test_singleton_tail_no_model_call() -> None:
    calls: list[tuple[TrustedCandidate, ...]] = []
    # 5 candidates, batch size 4 -> [4, 1]; singleton never passed to choose.
    winner = select_candidate(tuple(cand(i) for i in range(5)), RLCD_CAPS, pick("c3", calls))
    assert winner.candidate_id == "c3"
    assert all(len(batch) >= 2 for batch in calls)


def test_unknown_and_abstain_fail_closed() -> None:
    cands = tuple(cand(i) for i in range(2))
    with pytest.raises(SelectionError, match="does not match any"):
        select_candidate(cands, RLCD_CAPS, pick("ghost", []))
    with pytest.raises(SelectionError, match="abstention"):
        select_candidate(cands, RLCD_CAPS, pick(ABSTAIN_ANSWER_ID, []))


def test_rlcd_batch_lengths_map_to_supported_totals() -> None:
    sizes = browser_choice_batch_sizes(RLCD_CAPS)
    assert sizes
    for size in sizes:
        assert size + 1 in RLCD_CAPS.supported_total_candidates


def test_empty_and_duplicate_inputs_fail_closed() -> None:
    with pytest.raises(SelectionError, match="no candidates"):
        select_candidate((), RLCD_CAPS, pick("c0", []))
    with pytest.raises(SelectionError, match="duplicate"):
        select_candidate((cand(0), cand(0)), RLCD_CAPS, pick("c0", []))
