"""Registry-derived browser compatibility tests."""

from __future__ import annotations

from bruv.backends.registry import (
    backend_capabilities,
    backend_names,
    browser_choice_batch_sizes,
    browser_compatible_backends,
    browser_supports_noul_choice,
)
from bruv.domain.validation import BackendCapabilities

RLCD_BATCHES = (2, 3, 4, 5, 6, 8, 10, 16, 24)


def test_every_registered_backend_supports_noul_and_choice() -> None:
    for name in backend_names:
        capabilities = backend_capabilities(name)
        assert browser_supports_noul_choice(capabilities), name
    assert browser_compatible_backends() == backend_names


def test_support_requires_both_question_types() -> None:
    capabilities = BackendCapabilities(
        backend="partial",
        question_types=frozenset({"noul"}),
        calibrated=True,
        allows_json_state=True,
    )
    assert browser_supports_noul_choice(capabilities) is False


def test_choice_bounds_coming_from_canonical_question() -> None:
    capabilities = BackendCapabilities(
        backend="plain",
        question_types=frozenset({"noul", "choice"}),
        calibrated=True,
        allows_json_state=True,
    )
    sizes = browser_choice_batch_sizes(capabilities)
    # Without abstention, batch size equals total options: 2..50 canonical bounds.
    assert sizes == tuple(range(2, 51))


def test_explicit_abstention_offsets_batch_size() -> None:
    capabilities = BackendCapabilities(
        backend="abstaining",
        question_types=frozenset({"noul", "choice"}),
        calibrated=True,
        allows_json_state=True,
        explicit_abstention=True,
    )
    sizes = browser_choice_batch_sizes(capabilities)
    # Abstention shifts totals but never makes a one-criterion question valid.
    # With supported_total_candidates=None, explicit_abstention does not reduce criteria max (2..50 inclusive).
    assert sizes == tuple(range(2, 51))


def test_rlcd_per_k_calibrated_totals_only() -> None:
    capabilities = backend_capabilities("rlcd-modernbert")
    assert capabilities.explicit_abstention is True
    assert capabilities.supported_total_candidates is not None
    sizes = browser_choice_batch_sizes(capabilities)
    assert sizes == RLCD_BATCHES
    # Unsupported candidate totals must be absent, not approximated:
    # batch 7 (total 8) and batch 25 (total 26) have no per_k entry.
    for missing in (7, 9, 11, 25, 50):
        assert missing not in sizes


def test_rlcd_never_falls_back_outside_supported_totals() -> None:
    capabilities = backend_capabilities("rlcd-modernbert")
    sizes = browser_choice_batch_sizes(capabilities)
    for size in sizes:
        assert size + 1 in capabilities.supported_total_candidates
