"""Self-describing CLI command spec from one data source."""

from __future__ import annotations

from typing import Any

from bruv import __version__

COMMANDS: list[dict[str, Any]] = [
    {
        "name": "noul",
        "description": "Ask a noul (yes/no confidence) question.",
        "arguments": [{"name": "instructions", "required": True}],
        "flags": ["--id", "--state", "--state-file", "--model", "--backend"],
        "common_flags": True,
    },
    {
        "name": "choice",
        "description": "Ask a choice question over two or more options.",
        "arguments": [{"name": "instructions", "required": True}],
        "flags": ["--id", "--option", "--state", "--state-file", "--model", "--backend"],
        "common_flags": True,
    },
    {
        "name": "score",
        "description": "Ask a score question over two or more ordered levels.",
        "arguments": [{"name": "instructions", "required": True}],
        "flags": ["--id", "--level", "--state", "--state-file", "--model", "--backend"],
        "common_flags": True,
    },
    {
        "name": "eval",
        "description": "Evaluate a multi-question request from a file.",
        "arguments": [{"name": "eval_file", "required": True}],
        "flags": ["--state", "--state-file", "--model", "--backend"],
        "common_flags": True,
    },
    {
        "name": "validate",
        "description": "Validate a canonical request without performing any paid call.",
        "arguments": [],
        "flags": ["--state", "--state-file", "--eval-file", "--backend", "--output", "--quiet"],
        "common_flags": False,
    },
    {
        "name": "spec",
        "description": "Print the machine-readable CLI command spec.",
        "arguments": [],
        "flags": ["--output"],
        "common_flags": False,
    },
    {
        "name": "schema",
        "description": "Print a JSON Schema (request|output|error).",
        "arguments": [{"name": "name", "required": True}],
        "flags": [],
        "common_flags": False,
    },
]

COMMON_FLAGS = [
    "--output",
    "--quiet",
    "--no-color",
    "--dry-run",
    "--field",
    "--fail-under",
    "--abstain-band",
]

BACKEND_VALUES = ["typesafe", "simple-jev"]
OUTPUT_MODES = ["human", "json"]

EXIT_CODES = {
    "success": 0,
    "usage_or_validation": 2,
    "authentication": 3,
    "backend_unavailable": 4,
    "provider_response": 5,
    "internal_error": 70,
    "gate_failed": 10,
    "abstained": 11,
}

PAID_REQUEST_SEMANTICS = (
    "Every error envelope includes paid_request: true when a provider call was "
    "attempted and may have incurred cost, and false otherwise. validate, "
    "schema, spec, help/version, doctor, and dry-run never incur a paid request."
)


def spec() -> dict[str, Any]:
    """Return the full CLI command spec from one data source."""
    return {
        "version": __version__,
        "commands": COMMANDS,
        "common_flags": COMMON_FLAGS,
        "backends": BACKEND_VALUES,
        "output_modes": OUTPUT_MODES,
        "exit_codes": EXIT_CODES,
        "paid_request_semantics": PAID_REQUEST_SEMANTICS,
    }


__all__ = ["spec"]
