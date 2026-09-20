"""Standard checks for the packaged bruv skill."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

SKILL = Path("skills/bruv/SKILL.md")
SCENARIOS = Path("tests/scenarios/expected.json")


def _frontmatter() -> dict:
    text = SKILL.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    _, front, _ = text.split("---\n", 2)
    return yaml.safe_load(front)


def test_skill_identifier_is_bruv() -> None:
    front = _frontmatter()
    assert front["name"] == "bruv"
    assert front["description"].strip()


def test_skill_mentions_required_commands() -> None:
    text = SKILL.read_text(encoding="utf-8")
    for needle in ["bruv validate", "--dry-run", "--output json", "calibrated: false"]:
        assert needle in text


def test_skill_documents_stable_exits() -> None:
    text = SKILL.read_text(encoding="utf-8")
    for code in ["0", "2", "3", "4", "5", "10", "11", "70"]:
        assert code in text


def test_skill_has_no_secret_examples() -> None:
    text = SKILL.read_text(encoding="utf-8")
    assert "TYPESAFE_API_KEY=" not in text
    assert "API_KEY=" not in text
    assert "sk-" not in text


def test_skill_preserves_calibration_language() -> None:
    text = SKILL.read_text(encoding="utf-8")
    assert "uncalibrated" in text
    assert "calibrated: false" in text


def test_skill_file_is_packaged() -> None:
    assert SKILL.is_file()


def test_expected_scenarios_complete() -> None:
    data = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    ids = {s["id"] for s in data["scenarios"]}
    assert {
        "01-choose-type",
        "02-validate-first",
        "03-request-json",
        "04-exit-codes",
        "05-calibration",
        "06-no-key-flags",
        "07-non-interactive",
    } <= ids


@pytest.mark.parametrize("directory", ["baseline", "with_skill"])
def test_scenario_files_exist(directory: str) -> None:
    base = Path("tests/scenarios", directory)
    assert base.is_dir()
    assert len(list(base.glob("*.md"))) == 7
