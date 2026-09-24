"""Observation builder tests: filtering, gating, determinism, allowlisting."""

from __future__ import annotations

from bruv.browser.observation import ObservationBuilder, RawDomElement

BUILDER = ObservationBuilder(
    max_elements=20, max_page_text_chars=500, allowed_origins=("https://app.example.com",)
)


def el(index: int, tag: str, **kw) -> RawDomElement:
    return RawDomElement(index=index, tag=tag, **kw)


def kinds_by_id(observables):
    return {c.candidate_id: c for c in observables.candidates}


def test_sensitive_hidden_disabled_fields_excluded_editable_kept() -> None:
    result = BUILDER.build(
        "https://example.com/form",
        "Form",
        "text",
        (
            el(0, "input", input_type="hidden", attributes={"name": "csrf"}),
            el(1, "input", input_type="password"),
            el(2, "input", input_type="file"),
            el(3, "input", input_type="text", autocomplete="cc-number"),
            el(4, "input", input_type="text", autocomplete="one-time-code"),
            el(5, "input", input_type="text", disabled=True),
            el(6, "input", input_type="text", visible=False),
            el(7, "input", input_type="email"),
            el(8, "input", input_type="tel"),
            el(9, "input", input_type="search"),
        ),
    )
    kept = [c for c in result.candidates if c.kind == "request_text"]
    assert [c.description for c in kept] == [
        "input email:",
        "input tel:",
        "input search:",
    ]
    assert all(c.needs_review_kind is None for c in kept)


def test_deterministic_ordering_fingerprints_and_bounds() -> None:
    elements = (
        el(5, "a", href="https://example.com/x", text="link"),
        el(2, "input", input_type="email"),
        el(1, "button", text="Go"),
        el(0, "button", text="Go"),
    )
    first = BUILDER.build("https://example.com/", "T", "page text", elements)
    second = BUILDER.build("https://example.com/", "T", "page text", elements)
    assert first.summary.observation_id == second.summary.observation_id
    assert first.summary.page_text_ref == second.summary.page_text_ref
    order = [c.candidate_id for c in first.candidates[:4]]
    assert order == ["c0000", "c0001", "c0002", "c0003"]
    # Editable first (index 2), then controls by index, then link.
    assert first.candidates[0].kind == "request_text"
    assert [c.kind for c in first.candidates[1:3]] == ["click", "click"]
    assert first.candidates[3].kind == "navigate"
    tiny = ObservationBuilder(max_elements=2, max_page_text_chars=10, allowed_origins=())
    bounded = tiny.build("https://example.com/", "T", "x" * 100, elements)
    assert len(bounded.summary.elements) == 2
    assert len(bounded.page_text) == 10


def test_allowlisted_navigation_and_unsafe_origin_excluded() -> None:
    result = BUILDER.build(
        "https://example.com/page",
        "Nav",
        "text",
        (
            el(0, "a", href="/relative", text="rel"),
            el(1, "a", href="https://app.example.com/deep", text="app"),
            el(2, "a", href="https://evil.com/x", text="bad"),
            el(3, "a", href="javascript:void(0)", text="js"),
            el(
                4, "a", href="https://example.com/file.zip", text="zip", attributes={"download": ""}
            ),
        ),
    )
    by_id = kinds_by_id(result)
    nav = [c for c in result.candidates if c.kind == "navigate"]
    assert len(nav) == 3
    assert result.target_for(nav[0].candidate_id) == "https://example.com/relative"
    assert result.target_for(nav[1].candidate_id) == "https://app.example.com/deep"
    assert by_id[nav[2].candidate_id].needs_review_kind == "file_transfer"
    assert all("evil.com" not in (result.target_for(c.candidate_id) or "") for c in nav)


def test_submit_payment_review_gating() -> None:
    with_password = BUILDER.build(
        "https://example.com/login",
        "Login",
        "text",
        (
            el(0, "input", input_type="password"),
            el(1, "button", text="Sign in"),
            el(2, "input", input_type="submit", text=""),
            el(3, "button", text="Pay now"),
            el(4, "button", text="Delete account"),
            el(5, "div", role="tab", text="Tab"),
        ),
    )
    by_id = kinds_by_id(with_password)
    # c0000 "Sign in" (text match), c0001 blank generic input[type=submit].
    # Password input itself is excluded, so IDs shift down.
    assert by_id["c0000"].needs_review_kind == "login_submit"
    assert by_id["c0001"].needs_review_kind == "login_submit"
    assert by_id["c0002"].needs_review_kind == "payment"
    assert by_id["c0003"].needs_review_kind == "destructive"
    assert by_id["c0004"].needs_review_kind is None  # role=tab is not a submit

    no_password = BUILDER.build(
        "https://example.com/contact",
        "Contact",
        "text",
        (el(0, "button", text="Submit"),),
    )
    assert no_password.candidates[0].needs_review_kind == "message_submit"


def test_scroll_wait_present_and_no_finish() -> None:
    result = BUILDER.build("https://example.com/", "T", "text", ())
    synthetic = [c for c in result.candidates if c.candidate_id.startswith("s")]
    assert [c.kind for c in synthetic] == ["scroll", "scroll", "wait"]
    assert all(c.needs_review_kind is None for c in synthetic)
    assert not any(c.kind == "request_review" for c in result.candidates)
    assert not any("finish" in c.candidate_id for c in result.candidates)
