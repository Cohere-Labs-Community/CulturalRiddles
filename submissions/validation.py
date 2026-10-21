"""Validate anonymous intake workbooks locally, with no remote references."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

import pandas as pd

from scripts.release_support import _is_identifying_key

STANDARD_TOPICS = (
    "Politics",
    "History",
    "Geography",
    "Arts",
    "Sports",
    "National Heritage",
    "Kid's World",
    "Everyday Life",
)
FIXED_TOPICS = set(STANDARD_TOPICS)
WILDCARDS = {"Wildcard 1 [rename]", "Wildcard 2 [rename]"}
COMMUNITY_COLUMNS = ("community", "language_code", "language", "country", "country_code")
REQUIRED_COLUMNS = (
    "Number",
    "Topic",
    "Riddle (Original Language)",
    "Reference Answer (Original)",
)
OPTIONAL_COLUMNS = (
    "Riddle (English Translation)",
    "Hints (Optional)",
    "Comments (Optional)",
    "Reference Answer (English)",
)


def text(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


MAX_LISTED_ROWS = 10


def _format_rows(rows: list[int]) -> str:
    """List worksheet rows, abbreviating long lists such as an untouched template."""
    if len(rows) <= MAX_LISTED_ROWS:
        return str(rows)
    return f"{rows[:MAX_LISTED_ROWS]} and {len(rows) - MAX_LISTED_ROWS} more"


@dataclass
class ValidationResult:
    metadata: dict = field(default_factory=dict)
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _identifying(column) -> bool:
    return _is_identifying_key(column) or any(
        token in str(column).casefold() for token in ("author", "name")
    )


def _sheet(book: pd.ExcelFile, name: str, result: ValidationResult) -> pd.DataFrame:
    raw = book.parse(name, header=None, dtype=object, keep_default_na=False)
    if raw.empty:
        result.errors.append(f"{name}: sheet is empty; keep the template's header row.")
        return pd.DataFrame()
    headers = [text(value) for value in raw.iloc[0]]
    duplicates = sorted(header for header, count in Counter(headers).items() if count > 1)
    if duplicates:
        result.errors.append(f"{name}: duplicate column headers {duplicates}; keep one of each.")
    identifying = [header for header in headers if _identifying(header)]
    if identifying:
        result.errors.append(
            f"{name}: identity/attribution columns are not accepted: {identifying}. "
            "Remove these columns and their contents."
        )
    if duplicates:
        return pd.DataFrame()
    frame = raw.iloc[1:].copy()
    frame.columns = headers
    return frame


def validate(path: Path) -> ValidationResult:
    """Return the same blocking errors and review warnings for check and import."""
    result = ValidationResult()
    if path.suffix.lower() != ".xlsx":
        result.errors.append(
            "Use an .xlsx workbook with Community and Riddles sheets; CSV cannot "
            "carry the required anonymous community metadata."
        )
        return result
    try:
        with pd.ExcelFile(path, engine="openpyxl") as book:
            missing = sorted({"Community", "Riddles"} - set(book.sheet_names))
            if missing:
                result.errors.append(
                    f"Workbook is missing sheets: {missing}. Start from the anonymous template."
                )
                return result
            frames = {name: _sheet(book, name, result) for name in book.sheet_names}
    except (ValueError, BadZipFile, OSError, KeyError, ParseError) as error:
        result.errors.append(
            f"Cannot read workbook {path}: {error}. Save it as a valid .xlsx file."
        )
        return result

    metadata = frames["Community"]
    metadata = metadata.loc[metadata.apply(lambda column: column.map(text)).ne("").any(axis=1)]
    if len(metadata) != 1:
        result.errors.append(
            f"Community: expected exactly one metadata row, found {len(metadata)}."
        )
    else:
        result.metadata = {key: text(value) for key, value in metadata.iloc[0].items()}
        for key in ("community", "language_code"):
            if not result.metadata.get(key):
                result.errors.append(f"Community: {key} is required; fill in its metadata cell.")
        community = result.metadata.get("community", "")
        if community and not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", community):
            result.errors.append(
                "Community: community must use lowercase ASCII letters, digits, and "
                "single underscores (for example, example_country)."
            )
        for key in ("language", "country", "country_code"):
            if not result.metadata.get(key):
                result.warnings.append(f"Community: fill in {key} for maintainer review.")
        extra = sorted(set(metadata.columns) - set(COMMUNITY_COLUMNS))
        if extra:
            result.warnings.append(f"Community: extra columns are not imported: {extra}.")

    frame = frames["Riddles"]
    result.frame = frame
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        result.errors.append(f"Riddles: missing required columns: {', '.join(missing)}.")
    optional_missing = sorted(set(OPTIONAL_COLUMNS) - set(frame.columns))
    if optional_missing:
        result.warnings.append(f"Riddles: optional columns are absent: {optional_missing}.")
    extra = sorted(set(frame.columns) - set(REQUIRED_COLUMNS) - set(OPTIONAL_COLUMNS))
    if extra:
        result.warnings.append(f"Riddles: extra columns are not imported: {extra}.")
    if len(frame) != 100:
        result.errors.append(f"Riddles: expected 100 riddles, found {len(frame)}.")
    for column in REQUIRED_COLUMNS:
        if column not in frame:
            continue
        blank = frame[column].map(text).eq("")
        if blank.any():
            rows = (frame.index[blank] + 1).tolist()
            result.errors.append(
                f"Riddles: {column} is blank at worksheet rows {_format_rows(rows)}."
            )

    numbers = []
    if "Number" in frame:
        for index, value in frame["Number"].items():
            value = text(value)
            if re.fullmatch(r"[0-9]+(?:\.0+)?", value):
                numbers.append(int(value.split(".")[0]))
            else:
                numbers.append(None)
                if value:
                    result.errors.append(
                        f"Riddles: worksheet row {index + 1}: Number must be an integer "
                        f"from 1 through 100; found {value!r}."
                    )
        counts = Counter(number for number in numbers if number is not None)
        duplicates = sorted(number for number, count in counts.items() if count > 1)
        if duplicates:
            result.errors.append(
                f"Riddles: Number values must be unique; duplicates: {duplicates}."
            )
        missing_numbers = sorted(set(range(1, 101)) - set(counts))
        unexpected = sorted(set(counts) - set(range(1, 101)))
        if missing_numbers or unexpected:
            result.errors.append(
                "Riddles: Number values must cover 1 through 100 exactly once; "
                f"missing: {missing_numbers}; out of range: {unexpected}."
            )
        elif numbers != list(range(1, 101)):
            result.warnings.append(
                "Riddles: rows are not ordered by Number; IDs use Number, not worksheet position."
            )

    if "Topic" in frame:
        topics = frame["Topic"].map(text)
        counts = topics.value_counts().to_dict()
        bad_counts = {topic: count for topic, count in counts.items() if count != 10}
        if bad_counts:
            result.errors.append(
                f"Riddles: each topic must contain exactly 10 riddles; found {bad_counts}."
            )
        present = set(topics) - {""}
        missing_topics = sorted(FIXED_TOPICS - present)
        if missing_topics:
            result.errors.append(
                f"Riddles: required standard topics are missing: {missing_topics}."
            )
        replacements = present - FIXED_TOPICS - WILDCARDS
        if len(replacements) != 2:
            result.errors.append(
                f"Riddles: include exactly two distinct renamed wildcard topics; found {sorted(replacements)}."
            )
        if present & WILDCARDS:
            result.errors.append(
                "Riddles: rename both wildcard topics before submission; placeholders remain: "
                f"{sorted(present & WILDCARDS)}."
            )
        moved = [
            number
            for number, topic in zip(numbers, topics)
            if number is not None
            and 1 <= number <= 80
            and topic != STANDARD_TOPICS[(number - 1) // 10]
        ]
        if moved:
            result.warnings.append(
                f"Riddles: standard topics differ from the template's Number blocks at {moved}. "
                "Review the assignments; existing IDs are retained by Number."
            )

    for column in ("Reference Answer (Original)", "Reference Answer (English)"):
        if column not in frame:
            continue
        groups = defaultdict(list)
        for index, value in frame[column].items():
            normalized = " ".join(text(value).casefold().split())
            if normalized:
                groups[normalized].append(index + 1)
        duplicates = [rows for rows in groups.values() if len(rows) > 1]
        if duplicates:
            result.warnings.append(
                f"Riddles: {column} has duplicate answers (case and whitespace ignored) "
                f"at worksheet row groups {duplicates}. Review them; repeated answers are allowed."
            )
    return result
