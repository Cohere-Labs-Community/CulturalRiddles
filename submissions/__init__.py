"""Validated, reviewed riddle dataset contribution tools."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

EVALUATION_FIELDS = (
    "riddle_id",
    "community",
    "language",
    "topic",
    "is_wildcard",
    "riddle_original",
    "riddle_english",
    "riddle_hint",
    "riddle_note",
    "reference_answer_original",
    "reference_answer_english",
)


def evaluation_fingerprint(row: Mapping[str, Any]) -> str:
    """Hash the public riddle content that determines benchmark answers."""
    payload = {field: row.get(field) for field in EVALUATION_FIELDS}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
