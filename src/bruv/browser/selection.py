"""Recursive tournament selection of exactly one browser action candidate.

One selection consumes one or more backend Choice calls via a caller-supplied
callback. The callback represents exactly one canonical Choice call over one
batch of candidates; this module never builds backend requests and never
executes actions. Only the single final winner is returned — never one action
per batch.

Batch sizes come exclusively from ``browser_choice_batch_sizes`` so RLCD
calibrated-only ``supported_total_candidates`` totals are preserved. When no
valid batch size exists, selection fails closed with ``SelectionError``.
"""

from __future__ import annotations

from typing import Protocol

from bruv.backends.registry import browser_choice_batch_sizes
from bruv.browser.models import TrustedCandidate
from bruv.domain.results import ABSTAIN_ANSWER_ID
from bruv.domain.validation import BackendCapabilities


class SelectionError(ValueError):
    """Raised when no exactly-one candidate can be selected safely."""


class BatchChoice(Protocol):
    """One backend Choice call over one batch of trusted candidates.

    The batch length is guaranteed to be a valid canonical Choice criteria
    count (and, for RLCD, a calibrated per_k total). Implementations return
    exactly one opaque ``candidate_id`` string from that batch.
    """

    def __call__(self, batch: tuple[TrustedCandidate, ...], /) -> str: ...


# Defensive bound on tournament depth; every round at least halves the round
# length, so 64 rounds is far beyond any supported configuration.
_MAX_ROUNDS = 64


def _supported_sizes_desc(capabilities: BackendCapabilities) -> tuple[int, ...]:
    """Valid batch sizes from the registry, descending. Fails closed."""
    sizes = tuple(sorted(set(browser_choice_batch_sizes(capabilities)), reverse=True))
    if not sizes or sizes[-1] < 2:
        raise SelectionError(
            "no supported Choice batch size >=2 for this backend; "
            "cannot select a candidate without calibrated batching"
        )
    return sizes


def _partition(
    length: int,
    sizes_desc: tuple[int, ...],
    size_set: frozenset[int],
) -> list[int] | None:
    """Deterministically split ``length`` into supported batch sizes.

    Prefers fewer/larger batches. The remainder after full batches must be
    empty, a supported size, or exactly one singleton carry; otherwise the
    next smaller size is tried. Returns batch sizes in order, or None when no
    supported partition exists.
    """
    for size in sizes_desc:
        if size > length:
            continue
        remainder = length % size
        if remainder == 0:
            return [size] * (length // size)
        if remainder == 1:
            return [size] * (length // size) + [1]
        if remainder in size_set:
            return [size] * (length // size) + [remainder]
    return None


def _run_round(
    items: tuple[TrustedCandidate, ...],
    sizes_desc: tuple[int, ...],
    size_set: frozenset[int],
    choose: BatchChoice,
) -> tuple[TrustedCandidate, ...]:
    batches = _partition(len(items), sizes_desc, size_set)
    if batches is None:
        raise SelectionError(
            f"cannot partition {len(items)} candidates into supported "
            f"batch sizes {sorted(sizes_desc)}"
        )
    winners: list[TrustedCandidate] = []
    offset = 0
    for size in batches:
        batch = tuple(items[offset : offset + size])
        offset += size
        if len(batch) == 1:
            # Singleton tail carries forward without a model call.
            winners.append(batch[0])
            continue
        answer = choose(batch)
        if not isinstance(answer, str) or not answer.strip():
            raise SelectionError("backend Choice call returned no usable candidate ID")
        if answer == ABSTAIN_ANSWER_ID:
            raise SelectionError("abstention is not a selectable candidate; refusing to act")
        for candidate in batch:
            if candidate.candidate_id == answer:
                winners.append(candidate)
                break
        else:
            raise SelectionError(
                f"backend Choice answer {answer!r} does not match any "
                f"candidate in the batch of {len(batch)}"
            )
    return tuple(winners)


def select_candidate(
    candidates: tuple[TrustedCandidate, ...],
    capabilities: BackendCapabilities,
    choose: BatchChoice,
) -> TrustedCandidate:
    """Select exactly one final candidate through a recursive tournament.

    Deterministic: preserves input order, batches greedily with the largest
    supported size, and carries singleton tails without model calls. The
    callback is invoked once per batch of >=2 candidates; exactly one final
    winner is returned and nothing is executed here.
    """
    items = tuple(candidates)
    if not items:
        raise SelectionError("no candidates to select from")
    ids = [candidate.candidate_id for candidate in items]
    if len(set(ids)) != len(ids):
        raise SelectionError("duplicate candidate IDs in selection input")
    if len(items) == 1:
        return items[0]

    sizes_desc = _supported_sizes_desc(capabilities)
    size_set = frozenset(sizes_desc)

    rounds = 0
    while len(items) > 1:
        items = tuple(_run_round(items, sizes_desc, size_set, choose))
        rounds += 1
        if rounds > _MAX_ROUNDS:
            raise SelectionError("candidate selection exceeded the round limit")

    return items[0]


__all__ = [
    "BatchChoice",
    "SelectionError",
    "select_candidate",
]
