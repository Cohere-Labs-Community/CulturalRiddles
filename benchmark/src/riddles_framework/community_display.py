"""Format prompt context from the released canonical community metadata."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

# The root editable installation keeps the package under benchmark/src/.
CORPUS_METADATA_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "canonical" / "corpus_metadata.csv"
)

# Script-family terminology comes from corpus_metadata.csv, not community keys.
SCRIPT_FAMILY_TO_ISO: dict[str, str] = {
    "Arabic": "Arab",
    "Bengali-Assamese": "Beng",
    "Buginese": "Bugi",
    "Cyrillic": "Cyrl",
    "Devanagari": "Deva",
    "Gujarati": "Gujr",
    "Han": "Hani",
    "Han + Kana": "Jpan",
    "Kannada": "Knda",
    "Latin": "Latn",
    "Malayalam": "Mlym",
    "Sinhala": "Sinh",
    "Tamil": "Taml",
    "Telugu": "Telu",
    "Thai": "Thai",
}


@lru_cache(maxsize=1)
def _load_metadata(path: Path) -> dict[str, dict[str, str]]:
    """Read the canonical metadata once for the running process."""
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["community"].strip().lower(): row for row in csv.DictReader(handle)}


def _metadata(community: str) -> dict[str, str]:
    return _load_metadata(CORPUS_METADATA_PATH).get(community.strip().lower(), {})


def format_community_display(community: str) -> str:
    """Return canonical language and country, or format a custom key."""
    if not community:
        return "(unspecified)"
    metadata = _metadata(community)
    if metadata:
        return f"{metadata['language']}, {metadata['country']}"
    parts = community.strip().lower().split("_", 1)
    return ", ".join(part.replace("_", " ").title() for part in parts)


def format_country(community: str) -> str:
    """Return the canonical country, with a generic custom-key fallback."""
    if not community:
        return "(unspecified)"
    metadata = _metadata(community)
    if metadata:
        return metadata["country"]
    parts = community.strip().lower().split("_", 1)
    return parts[1].replace("_", " ").title() if len(parts) == 2 else "(unspecified)"


def format_language_name(community: str) -> str:
    """Return the canonical language, with a generic custom-key fallback."""
    if not community:
        return "(unspecified)"
    metadata = _metadata(community)
    if metadata:
        return metadata["language"]
    return community.strip().split("_", 1)[0].title()


def format_script_iso(community: str) -> str:
    """Return the metadata script family's ISO 15924 code, or an empty string."""
    if not community:
        return ""
    return SCRIPT_FAMILY_TO_ISO.get(_metadata(community).get("script_family", ""), "")


def format_script_name(community: str) -> str:
    """Return the metadata script family as a prompt label."""
    if not community:
        return "(unspecified)"
    script = _metadata(community).get("script_family", "")
    return f"{script} script" if script else "(unspecified)"
