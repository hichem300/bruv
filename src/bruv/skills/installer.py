"""Safe agent skill installer for Codex, Claude Code, and Pi.

The installer plans before it applies: it locates the packaged skill source,
computes the destination for the target/scope, and refuses to overwrite an
existing file without explicit ``--force``. Copies are atomic.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Target = Literal["codex", "claude", "pi"]
Scope = Literal["user", "project"]

SKILL_RELATIVE = Path("skills") / "bruv" / "SKILL.md"


@dataclass(frozen=True, slots=True)
class SkillInstallPlan:
    source: Path
    destination: Path
    replacing: bool


def skill_source() -> Path:
    """Locate the packaged SKILL.md across editable, wheel, and standalone layouts."""
    candidates: list[Path] = []
    env_path = os.environ.get("BRUV_SKILL_PATH")
    if env_path:
        candidates.append(Path(env_path))
    # Editable/source layout: src/bruv/skills/installer.py -> repo root
    candidates.append(Path(__file__).resolve().parents[3] / SKILL_RELATIVE)
    # Installed wheel: artifacts live at site-packages root (parents[2])
    candidates.append(Path(__file__).resolve().parents[2] / SKILL_RELATIVE)
    # Standalone PyInstaller bundle
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / SKILL_RELATIVE)
    # Installed wheel next to package metadata
    candidates.append(Path(__file__).resolve().parents[1] / "skills" / "bruv" / "SKILL.md")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Could not locate the packaged bruv skill.")


def _user_skill_dir(target: Target) -> Path:
    home = Path.home()
    if target == "codex":
        return home / ".codex" / "skills" / "bruv"
    if target == "claude":
        return home / ".claude" / "skills" / "bruv"
    return home / ".pi" / "agent" / "skills" / "bruv"


def _project_skill_dir(target: Target, root: Path) -> Path:
    if target == "codex":
        return root / ".codex" / "skills" / "bruv"
    if target == "claude":
        return root / ".claude" / "skills" / "bruv"
    return root / ".pi" / "skills" / "bruv"


def plan_skill_install(target: Target, scope: Scope, root: Path | None = None) -> SkillInstallPlan:
    """Plan a skill installation, refusing to overwrite without ``--force``."""
    source = skill_source()
    if scope == "user":
        destination = _user_skill_dir(target) / "SKILL.md"
    else:
        if root is None:
            raise ValueError("--project-root is required for project scope")
        if not root.is_dir():
            raise FileNotFoundError(f"project root not found: {root}")
        destination = _project_skill_dir(target, root) / "SKILL.md"
    return SkillInstallPlan(source=source, destination=destination, replacing=destination.is_file())


def apply_skill_install(plan: SkillInstallPlan, *, force: bool) -> Path:
    """Apply a planned installation atomically, honoring ``force``."""
    if plan.replacing and not force:
        raise FileExistsError(
            f"destination already exists: {plan.destination} (use --force to replace)"
        )
    plan.destination.parent.mkdir(parents=True, exist_ok=True)
    data = plan.source.read_bytes()
    fd, tmp_name = tempfile.mkstemp(prefix="bruv-skill-", dir=str(plan.destination.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, plan.destination)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return plan.destination


__all__ = [
    "Scope",
    "SkillInstallPlan",
    "Target",
    "apply_skill_install",
    "plan_skill_install",
    "skill_source",
]
