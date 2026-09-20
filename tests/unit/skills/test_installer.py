"""Skill installer tests across target and scope."""

from __future__ import annotations

from pathlib import Path

import pytest

from bruv.skills.installer import apply_skill_install, plan_skill_install, skill_source

TARGETS = ["codex", "claude", "pi"]
SCOPES = ["user", "project"]


def test_skill_source_locates_packaged_file() -> None:
    assert skill_source().is_file()


@pytest.mark.parametrize("target", TARGETS)
def test_user_scope_plan_uses_home_dir(target: str, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    plan = plan_skill_install(target, "user")  # type: ignore[arg-type]
    assert plan.destination.is_absolute()
    assert plan.destination.name == "SKILL.md"
    assert not plan.replacing


@pytest.mark.parametrize("target", TARGETS)
def test_project_scope_plan_uses_project_root(target: str, tmp_path: Path) -> None:
    plan = plan_skill_install(target, "project", tmp_path)  # type: ignore[arg-type]
    assert tmp_path in plan.destination.parents
    assert plan.destination.name == "SKILL.md"


def test_project_scope_requires_root() -> None:
    with pytest.raises(ValueError):
        plan_skill_install("pi", "project")  # type: ignore[arg-type]


def test_project_scope_missing_root_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        plan_skill_install("pi", "project", tmp_path / "missing")  # type: ignore[arg-type]


def test_apply_install_writes_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    plan = plan_skill_install("pi", "user")  # type: ignore[arg-type]
    destination = apply_skill_install(plan, force=False)
    assert destination.is_file()
    assert destination.read_text().startswith("---\n")


def test_existing_destination_refused_without_force(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    plan = plan_skill_install("pi", "user")  # type: ignore[arg-type]
    apply_skill_install(plan, force=False)
    with pytest.raises(FileExistsError):
        plan = plan_skill_install("pi", "user")  # type: ignore[arg-type]
        apply_skill_install(plan, force=False)


def test_force_replaces_existing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    plan = plan_skill_install("pi", "user")  # type: ignore[arg-type]
    apply_skill_install(plan, force=False)
    plan = plan_skill_install("pi", "user")  # type: ignore[arg-type]
    apply_skill_install(plan, force=True)
    assert plan.destination.is_file()
