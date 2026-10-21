from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable


def make_uuid() -> str:
    return str(uuid.uuid4())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def getenv_required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Required environment variable not set: {name}")
    return value


def safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def looks_like_refusal(text: str | None) -> bool:
    """Best-effort refusal heuristic for v1.

    Important:
    - This heuristic is intentionally conservative.
    - It is English-only.
    - It is meant to catch policy/task refusals, not honest uncertainty.
    - We intentionally do NOT tag "I don't know", "cannot determine",
      or "not enough information" as refusals, because those may reflect
      epistemically honest behavior that the benchmark wants to measure.

    This should be treated as a weak signal for analysis, not as a
    definitive refusal classifier.
    """
    if not text:
        return False

    t = text.strip().lower()

    refusal_patterns = [
        "i can't help with that",
        "i cannot help with that",
        "i can\u2019t help with that",
        "i can't assist with that",
        "i cannot assist with that",
        "i can\u2019t assist with that",
        "i can't comply with that",
        "i cannot comply with that",
        "i can\u2019t comply with that",
        "i'm sorry, but i can't",
        "i\u2019m sorry, but i can\u2019t",
        "sorry, i can't help",
        "sorry, i cannot help",
        "sorry, i can't assist",
        "sorry, i cannot assist",
    ]
    return any(pattern in t for pattern in refusal_patterns)
