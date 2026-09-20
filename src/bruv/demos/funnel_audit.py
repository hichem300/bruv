"""Funnel audit demo resources and runner."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from bruv.domain.requests import DecisionRequest

DEMO_NAME = "funnel-audit-agent"
DEMO_RELATIVE = Path("examples") / DEMO_NAME


def demo_root() -> Path:
    """Locate the packaged demo across editable, wheel, and standalone layouts."""
    candidates: list[Path] = []
    env_path = os.environ.get("BRUV_DEMO_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path(__file__).resolve().parents[3] / DEMO_RELATIVE)
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / DEMO_RELATIVE)
    candidates.append(Path(__file__).resolve().parents[1] / "examples" / DEMO_NAME)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the packaged funnel-audit demo.")


def demo_file(name: str) -> Path:
    return demo_root() / name


def load_demo_request() -> DecisionRequest:
    """Load audit.yaml into a canonical request without executing it."""
    from bruv.command_builders import load_eval_request

    return load_eval_request(demo_file("audit.yaml"), None)


def copy_demo_to(destination: Path) -> Path:
    """Copy the complete demo directory without executing anything."""
    source = demo_root()
    destination.mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        target = destination / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, target)
    return destination


__all__ = ["copy_demo_to", "demo_file", "demo_root", "load_demo_request"]
