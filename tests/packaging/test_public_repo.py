"""Repository policy tests for the public bruv repository."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOKEN_RE = re.compile(r"sk-[A-Za-z0-9]{16,}|TYPESAFE_API_KEY\s*=\s*[\"'][A-Za-z0-9]{8,}")


REQUIRED_FILES = [
    "LICENSE",
    "NOTICE",
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "SUPPORT.md",
]


@pytest.mark.parametrize("name", REQUIRED_FILES)
def test_required_repo_files_exist(name: str) -> None:
    assert (ROOT / name).is_file(), f"missing {name}"


def test_license_is_apache_2() -> None:
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in text
    assert "Version 2.0" in text


def test_notice_names_third_party_marks() -> None:
    text = (ROOT / "NOTICE").read_text(encoding="utf-8")
    for mark in ["TypeSafe AI", "Featherless AI", "Qwen"]:
        assert mark in text
    assert "unofficial" in text.lower()


def test_readme_unofficial_disclosure_present() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "unofficial" in text.lower()
    assert "TypeSafe AI" in text or "TypeSafe" in text


def test_security_gives_private_reporting_path() -> None:
    text = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert "privately" in text.lower() or "private" in text.lower()


def test_readme_order_matches_design() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    order = [
        text.find("60-second"),
        text.find("Human use"),
        text.find("Agent use"),
        text.find("Funnel audit demo"),
        text.find("Backends and calibration"),
        text.find("Security and privacy"),
    ]
    assert order == sorted(order)
    assert all(i >= 0 for i in order)


def test_fixtures_and_examples_have_no_tokens() -> None:
    candidates = list((ROOT / "tests/fixtures").rglob("*")) + list((ROOT / "examples").rglob("*"))
    for path in candidates:
        if not path.is_file():
            continue
        if path.suffix in {".json", ".yaml", ".yml", ".md", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert not TOKEN_RE.search(text), f"token-like value in {path}"


def test_github_templates_present() -> None:
    for name in [
        "ISSUE_TEMPLATE/bug.yml",
        "ISSUE_TEMPLATE/feature.yml",
        "PULL_REQUEST_TEMPLATE.md",
        "dependabot.yml",
    ]:
        assert (ROOT / ".github" / name).is_file(), f"missing .github/{name}"
