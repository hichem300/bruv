"""Machine-readable contracts."""

from __future__ import annotations

from bruv.contracts.schemas import SCHEMAS, error_schema, request_schema, result_schema
from bruv.contracts.spec import spec

__all__ = ["SCHEMAS", "error_schema", "request_schema", "result_schema", "spec"]
