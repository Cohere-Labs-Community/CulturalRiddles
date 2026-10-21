"""Run reproducible LLM-as-a-judge conditions from the repository root.

Each condition freezes its inputs and configuration before any API call. Final
verdicts, technical attempts, failures, and the manifest are separate files so
that technical errors never become answer scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Literal

import polars as pl
from dotenv import load_dotenv
from litellm import completion
from pydantic import ValidationError

from analysis.constants import REPO_ROOT, RIDDLES_PATH, RUNS_DIR
from riddles_framework.judging.llm_as_a_judge import (
    JUDGE_OUTPUT_SCHEMA,
    JUDGE_SYSTEM_PROMPT,
    JudgeVerdict,
    build_judge_messages,
    load_human_review_guidelines,
)

DEFAULT_MODEL = "gemini-3.5-flash"
SCORES = frozenset({0, 1, 2, 3})
FewShotStrategy = Literal["different_cultures", "same_culture", "fixed"]
CALIBRATION_CATEGORIES = {
    0: frozenset({"0.1", "0.2", "0.3", "0.4"}),
    1: frozenset({"N/A"}),
    2: frozenset({"2.1", "2.2", "2.3", "2.4"}),
    3: frozenset({"3.1", "3.2", "3.3", "N/A"}),
}


@dataclass(frozen=True)
class JudgeModel:
    """Provider settings recorded in immutable run manifests."""

    provider_model: str
    provider: str
    native_schema: bool
    key_name: str


JUDGE_MODELS = {
    "gemini-3.5-flash": JudgeModel("gemini/gemini-3.5-flash", "gemini", True, "GEMINI_API_KEY"),
    "deepseek-v4.1-flash": JudgeModel(
        "deepseek/deepseek-flash", "deepseek", False, "DEEPSEEK_API_KEY"
    ),
}
DEEPSEEK_BETA_API_BASE = "https://api.deepseek.com/beta"
DEEPSEEK_VERDICT_TOOL = "submit_judge_verdict"
DEEPSEEK_VERDICT_TOOL_SCHEMA = {
    **JUDGE_OUTPUT_SCHEMA,
    "allOf": [
        {
            "if": {"properties": {"score": {"const": 0}}},
            "then": {
                "properties": {
                    "verdict": {"const": "incorrect"},
                    "matched_reference": {"const": False},
                    "error_category": {"enum": ["0.1", "0.2", "0.3", "0.4"]},
                }
            },
        },
        {
            "if": {"properties": {"score": {"const": 1}}},
            "then": {
                "properties": {
                    "verdict": {"const": "unclear"},
                    "matched_reference": {"const": False},
                    "error_category": {"const": "N/A"},
                }
            },
        },
        {
            "if": {"properties": {"score": {"const": 2}}},
            "then": {
                "properties": {
                    "verdict": {"const": "partially_correct"},
                    "error_category": {"enum": ["2.1", "2.2", "2.3", "2.4"]},
                }
            },
        },
        {
            "if": {"properties": {"score": {"const": 3}}},
            "then": {
                "properties": {
                    "verdict": {"const": "correct"},
                    "error_category": {"enum": ["N/A", "3.1", "3.2", "3.3"]},
                }
            },
        },
        {
            "if": {"properties": {"error_category": {"enum": ["0.4", "2.4"]}}},
            "then": {"properties": {"validator_note": {"minLength": 1}}},
            "else": {"properties": {"validator_note": {"const": ""}}},
        },
    ],
}
DIRECT_JSON_INSTRUCTIONS = (
    "\n\nReturn exactly one JSON object matching this schema:\n"
    + json.dumps(JUDGE_OUTPUT_SCHEMA, ensure_ascii=False, sort_keys=True)
)
SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|authorization|bearer)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"\b(?:sk|AIza)[-_A-Za-z0-9]{12,}\b"),
)


class JudgeRunError(RuntimeError):
    """Raised when a run cannot meet its persisted contract."""


class ResumeMismatchError(JudgeRunError):
    """Raised when resume configuration or persisted finals do not match."""


@dataclass(frozen=True)
class AttemptResult:
    """One complete provider attempt, successful or not."""

    request: dict[str, Any]
    raw_response: str | None
    returned_model: str | None
    finish_reason: str | None
    usage: dict[str, Any] | None
    cost: float | None
    started_at: str
    completed_at: str
    latency_seconds: float
    validation_outcome: str
    error: str | None


def now() -> str:
    """Return a UTC timestamp."""
    return datetime.now(UTC).isoformat()


def digest(value: bytes) -> str:
    """Return a SHA-256 digest."""
    return hashlib.sha256(value).hexdigest()


def json_digest(value: Any) -> str:
    """Hash JSON after stable serialization."""
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def candidate_answer(row: Mapping[str, Any]) -> str:
    """Extract a candidate answer from supported input shapes."""
    if row.get("model_answer") is not None:
        return str(row.get("model_answer") or "").strip()
    raw = str(row.get("raw_response") or "")
    try:
        parsed = json.loads(raw)
    except JSONDecodeError:
        return raw.strip()
    return (
        str(parsed.get("riddle_answer") or "").strip()
        if isinstance(parsed, Mapping)
        else raw.strip()
    )


def riddle_from_row(row: Mapping[str, Any]) -> str:
    """Extract original-language riddle text from supported input shapes."""
    if row.get("riddle_original") is not None:
        return str(row.get("riddle_original") or "").strip()
    text = str(row.get("model_input") or "")
    return text.rsplit("RIDDLE:", 1)[-1].strip() if "RIDDLE:" in text else text.strip()


def judge_instance(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build the public riddle context sent to the judge."""
    return {
        "riddle_original": riddle_from_row(row),
        "hint": row.get("hint") or row.get("riddle_hint"),
        "community": row.get("community"),
        "topic": row.get("topic"),
        "language": row.get("language"),
        "reference_answer_english": row.get("reference_answer_english"),
        "reference_answer_original": row.get("reference_answer_original"),
    }


def identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return the immutable identity of an evaluated answer."""
    return (
        str(row.get("riddle_id") or "").strip(),
        str(row.get("model_name") or "").strip(),
        candidate_answer(row),
    )


def riddle_id(row: Mapping[str, Any]) -> str:
    """Return a nonempty riddle identifier."""
    value = str(row.get("riddle_id") or "").strip()
    if not value:
        raise JudgeRunError("Each row requires riddle_id")
    return value


def parse_human_score(value: Any) -> int | None:
    """Accept integral scores 0--3 and reject blanks, booleans, and fractions."""
    if value is None or isinstance(value, bool) or (isinstance(value, str) and not value.strip()):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return int(numeric) if numeric.is_integer() and int(numeric) in SCORES else None


def category(value: Any) -> str | None:
    """Normalize only the historical ``NA`` spelling."""
    text = str(value or "").strip()
    return "N/A" if text == "NA" else text or None


def validate_required(rows: Sequence[Mapping[str, Any]], source: str) -> None:
    """Reject rows that cannot form a complete contextual request."""
    for index, row in enumerate(rows, 1):
        missing = []
        if not str(row.get("riddle_id") or "").strip():
            missing.append("riddle_id")
        if not riddle_from_row(row):
            missing.append("riddle_original")
        if not str(row.get("reference_answer_original") or "").strip():
            missing.append("reference_answer_original")
        if missing:
            raise JudgeRunError(f"{source} row {index} is missing {', '.join(missing)}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL, recovering only one non-newline-terminated final fragment."""
    data = path.read_bytes()
    rows: list[dict[str, Any]] = []
    lines = data.splitlines(keepends=True)
    for index, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except JSONDecodeError as error:
            if index == len(lines) - 1 and not data.endswith(b"\n"):
                break
            raise JudgeRunError(f"Malformed JSONL in {path} at line {index + 1}") from error
        if not isinstance(row, dict):
            raise JudgeRunError(f"Non-object JSONL record in {path} at line {index + 1}")
        rows.append(row)
    return rows


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load CSV, Parquet, or JSONL without silently losing bad records.

    Parquet values are read as text, with empty strings for nulls, exactly as a
    CSV reader returns them. Scored-response exports use ``model`` as their
    model key and receive missing riddle context from the canonical corpus.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".parquet":
        frame = pl.read_parquet(path)
        if {"model", "model_answer", "answer_norm", "routing"}.issubset(frame.columns):
            if "model_name" not in frame.columns:
                frame = frame.rename({"model": "model_name"})
            context = [
                column
                for column in (
                    "riddle_original",
                    "reference_answer_original",
                    "reference_answer_english",
                    "community",
                    "topic",
                    "language",
                    "riddle_hint",
                )
                if column not in frame.columns
            ]
            if context:
                corpus = pl.read_ndjson(RIDDLES_PATH, infer_schema_length=None).select(
                    "riddle_id", *context
                )
                frame = frame.join(corpus, on="riddle_id", how="left", validate="m:1")
        return frame.with_columns(pl.all().cast(pl.String).fill_null("")).to_dicts()
    return read_jsonl(path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a frozen JSONL snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Append and sync one complete record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_stratified_score_allocation(value: str) -> dict[int, int]:
    """Parse a complete `score=count` allocation for 0--3 human-score strata."""
    allocation: dict[int, int] = {}
    try:
        for part in value.split(","):
            score_text, count_text = part.split("=", 1)
            score, count = int(score_text), int(count_text)
            if score not in range(4) or count < 1 or score in allocation:
                raise ValueError
            allocation[score] = count
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must contain each score 0=COUNT,1=COUNT,2=COUNT,3=COUNT once"
        ) from error
    if set(allocation) != set(range(4)):
        raise argparse.ArgumentTypeError(
            "must contain each score 0=COUNT,1=COUNT,2=COUNT,3=COUNT once"
        )
    return allocation


def stratified_score_sample(
    rows: Sequence[Mapping[str, Any]],
    *,
    allocation: Mapping[int, int],
    seed: int,
) -> list[dict[str, Any]]:
    """Draw a disproportionate score-stratified sample with design weights."""
    strata: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        score = parse_human_score(row.get("score"))
        if score is None:
            raise JudgeRunError("Stratified sampling requires valid 0--3 human scores")
        strata[score].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for score in range(4):
        population = len(strata[score])
        requested = allocation[score]
        if requested > population:
            raise JudgeRunError(
                f"Score-{score} stratum has {population} rows, fewer than requested {requested}"
            )
        for row in rng.sample(strata[score], requested):
            selected.append(
                {
                    **row,
                    "_sampling_stratum": f"human_score={score}",
                    "_sampling_population": population,
                    "_sampling_sample_size": requested,
                    "_sampling_weight": population / requested,
                }
            )
    rng.shuffle(selected)
    return selected


def select_evaluation(
    rows: list[dict[str, Any]],
    *,
    human_only: bool,
    sample: int | None,
    seed: int,
    limit: int | None,
    stratified_score_allocation: Mapping[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Select a deduplicated simple-random or score-stratified evaluation sample."""
    filtered = [row for row in rows if not human_only or row.get("scoring_method") == "human_eval"]
    if human_only:
        invalid = sum(parse_human_score(row.get("score")) is None for row in filtered)
        if invalid:
            raise JudgeRunError(
                f"Evaluation input has {invalid} human rows without valid 0--3 scores"
            )
    validate_required(filtered, "evaluation input")
    keys = [identity(row) for row in filtered]
    if len(keys) != len(set(keys)):
        raise JudgeRunError("Evaluation input contains duplicate answer identities")
    if stratified_score_allocation is not None:
        if not human_only:
            raise JudgeRunError("--stratified-score-allocation requires --human-only")
        if sample != sum(stratified_score_allocation.values()):
            raise JudgeRunError("--sample must equal the total --stratified-score-allocation")
        if limit is not None:
            raise JudgeRunError("--limit cannot be combined with --stratified-score-allocation")
        filtered = stratified_score_sample(
            filtered, allocation=stratified_score_allocation, seed=seed
        )
    elif sample is not None:
        if not 0 < sample <= len(filtered):
            raise JudgeRunError("--sample must be between 1 and the available row count")
        filtered = random.Random(seed).sample(filtered, sample)
    if limit is not None:
        if limit < 1:
            raise JudgeRunError("--limit must be positive")
        filtered = filtered[:limit]
    if not filtered:
        raise JudgeRunError("No rows remain for evaluation")
    return filtered


def calibration_problem(row: Mapping[str, Any]) -> str | None:
    """Explain why a recorded human label cannot be a few-shot example."""
    try:
        validate_required([row], "calibration input")
    except JudgeRunError as error:
        return str(error)
    score = parse_human_score(row.get("score"))
    if score is None:
        return "missing or invalid human score"
    code = category(row.get("error_type"))
    if code not in CALIBRATION_CATEGORIES[score]:
        return "score-incompatible or unresolved human category"
    note = str(row.get("validator_note") or "").strip()
    if code in {"0.4", "2.4"} and not note:
        return "residual category lacks validator note"
    if code not in {"0.4", "2.4"} and note:
        return "validator note is incompatible with category"
    return None


def eligible_calibration(
    rows: Sequence[dict[str, Any]], evaluation_riddles: set[str]
) -> list[dict[str, Any]]:
    """Return valid development labels, unique by riddle and disjoint from evaluation."""
    result: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    used: set[str] = set()
    for row in rows:
        key = riddle_id(row)
        if key in evaluation_riddles:
            rejected["evaluation riddle"] += 1
        elif problem := calibration_problem(row):
            rejected[problem] += 1
        elif key in used:
            rejected["duplicate riddle"] += 1
        else:
            used.add(key)
            result.append(row)
    if not result:
        details = ", ".join(f"{kind}={count}" for kind, count in sorted(rejected.items()))
        raise JudgeRunError(f"No eligible calibration examples ({details})")
    return result


def per_row_seed(seed: int, row: Mapping[str, Any]) -> int:
    """Derive deterministic nested example selections for an evaluated answer."""
    return int(digest(repr((seed, identity(row))).encode())[:16], 16)


def select_examples(
    row: Mapping[str, Any],
    *,
    calibration: Sequence[dict[str, Any]],
    evaluation_riddles: set[str],
    size: int,
    strategy: FewShotStrategy,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample examples with strict riddle and culture constraints.

    Different-culture means mutually distinct example communities that exclude
    the target community. Same-culture means every example is from the target
    community. The first four entries are a prefix of the eight-entry sample.
    Fixed means the first ``size`` calibration rows in file order, identical for
    every evaluated answer so the prompt prefix can be cached.
    """
    if size == 0:
        return []
    target_riddle, target_community = (
        riddle_id(row),
        str(row.get("community") or "").strip(),
    )
    candidates = [
        item
        for item in calibration
        if riddle_id(item) not in evaluation_riddles and riddle_id(item) != target_riddle
    ]
    rng = random.Random(per_row_seed(seed, row))
    if strategy == "fixed":
        selected = candidates[:size]
    elif strategy == "same_culture":
        selected = [
            item
            for item in candidates
            if str(item.get("community") or "").strip() == target_community
        ]
        rng.shuffle(selected)
        selected = selected[:size]
    else:
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in candidates:
            community = str(item.get("community") or "").strip()
            if community != target_community:
                groups.setdefault(community, []).append(item)
        communities = sorted(groups)
        rng.shuffle(communities)
        selected = []
        for community in communities:
            group = groups[community][:]
            rng.shuffle(group)
            selected.append(group[0])
            if len(selected) == size:
                break
    ids = [riddle_id(item) for item in selected]
    if len(selected) != size:
        raise JudgeRunError(
            f"Need {size} eligible {strategy} examples for {target_riddle}; found {len(selected)}"
        )
    if len(ids) != len(set(ids)) or set(ids) & evaluation_riddles:
        raise JudgeRunError("Calibration selection leaks or duplicates a riddle")
    communities = [str(item.get("community") or "").strip() for item in selected]
    if strategy == "different_cultures" and (
        len(communities) != len(set(communities)) or target_community in communities
    ):
        raise JudgeRunError("Different-culture selection violated its definition")
    return selected


def calibration_example(row: Mapping[str, Any]) -> dict[str, Any]:
    """Create a few-shot example without manufacturing a human rationale."""
    if problem := calibration_problem(row):
        raise JudgeRunError(f"Ineligible calibration row: {problem}")
    score = parse_human_score(row.get("score"))
    code = category(row.get("error_type"))
    assert score is not None and code is not None
    human_label: dict[str, Any] = {
        "score": score,
        "error_category": code,
    }
    note = str(row.get("validator_note") or "").strip()
    if note:
        human_label["validator_note"] = note
    return {
        **judge_instance(row),
        "riddle_id": riddle_id(row),
        "candidate": candidate_answer(row),
        "human_label_json": json.dumps(human_label, ensure_ascii=False, sort_keys=True),
    }


def redact(value: Any) -> Any:
    """Remove credential-shaped data from a persisted request or response."""
    if isinstance(value, str):
        for pattern in SECRET_PATTERNS:
            value = pattern.sub("[REDACTED]", value)
        return value
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, dict):
        return {str(key): redact(item) for key, item in value.items()}
    return value


def object_dict(value: Any) -> dict[str, Any] | None:
    """Convert provider metadata when its SDK exposes a mapping or model dump."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else None
    return None


def response_cost(response: Any) -> float | None:
    """Extract a provider-reported cost when LiteLLM exposes one."""
    hidden = object_dict(getattr(response, "_hidden_params", None)) or {}
    usage = object_dict(getattr(response, "usage", None)) or {}
    for value in (
        getattr(response, "response_cost", None),
        hidden.get("response_cost"),
        hidden.get("cost"),
        usage.get("cost"),
    ):
        if value is None or isinstance(value, bool):
            continue
        try:
            cost = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(cost) and cost >= 0:
            return cost
    return None


def build_provider_request(
    *,
    model: str,
    row: Mapping[str, Any],
    examples: Sequence[Mapping[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    config = JUDGE_MODELS[model]
    system_prompt = JUDGE_SYSTEM_PROMPT
    if not config.native_schema and config.provider != "deepseek":
        system_prompt += DIRECT_JSON_INSTRUCTIONS
    messages = build_judge_messages(
        judge_instance(row),
        candidate_answer=candidate_answer(row),
        system_prompt=system_prompt,
        human_review_guidelines=load_human_review_guidelines(),
        few_shot_examples=examples,
    )
    request: dict[str, Any] = {
        "model": config.provider_model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if config.provider == "deepseek":
        request.update(
            {
                "api_base": DEEPSEEK_BETA_API_BASE,
                "reasoning_effort": "none",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": DEEPSEEK_VERDICT_TOOL,
                            "description": "Submit the final judge verdict.",
                            "strict": True,
                            "parameters": DEEPSEEK_VERDICT_TOOL_SCHEMA,
                        },
                    }
                ],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": DEEPSEEK_VERDICT_TOOL},
                },
            }
        )
    if config.native_schema:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "judge_verdict",
                "strict": True,
                "schema": JUDGE_OUTPUT_SCHEMA,
            },
        }
    return request


def parse_verdict(content: str) -> JudgeVerdict:
    """Locally validate a raw structured provider reply."""
    value = content.strip()
    if value.startswith("```json"):
        value = value.removeprefix("```json").removesuffix("```").strip()
    return JudgeVerdict.model_validate_json(value)


def response_content(choice: Any) -> str:
    """Extract structured JSON text from message content or a strict tool call."""
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    if content:
        return str(content)
    tool_calls = getattr(message, "tool_calls", None) or []
    if len(tool_calls) != 1:
        raise ValueError("Provider returned neither JSON content nor one tool call")
    function = getattr(tool_calls[0], "function", None)
    name = getattr(function, "name", None)
    arguments = getattr(function, "arguments", None)
    if name != DEEPSEEK_VERDICT_TOOL or not isinstance(arguments, str):
        raise ValueError("Provider returned an unexpected verdict tool call")
    return arguments


def is_transient(error: Exception) -> bool:
    """Classify retryable provider errors without retrying permanent failures."""
    text = str(error).lower()
    return any(
        token in text
        for token in (
            "timeout",
            "rate",
            "temporar",
            "connection",
            "503",
            "429",
            "internalserver",
        )
    )


def retry_request_after_validation_failure(
    request: Mapping[str, Any],
    *,
    finish_reason: str | None = None,
    validation_error: str | None = None,
) -> dict[str, Any]:
    """Add a trusted schema reminder and recover a truncated output when possible."""
    retry = dict(request)
    messages = [dict(message) for message in request.get("messages", [])]
    reminder = (
        "\n\nRETRY REQUIREMENT: Your previous output failed local validation. "
        "Return one object that exactly follows the schema. Use the exact field "
        "name error_category. In particular: an incorrect verdict must have "
        "score 0 and error_category 0.1, 0.2, 0.3, or 0.4; a partially_correct "
        "verdict must have score 2 and error_category 2.1, 2.2, 2.3, or 2.4; "
        "an unclear verdict must have score 1 and error_category N/A; and a "
        "correct verdict must have score 3 and error_category N/A, 3.1, 3.2, "
        "or 3.3. An empty candidate is incorrect, not unclear. Keep "
        "validator_note compatible as well."
    )
    if validation_error:
        reminder += f"\nLocal validation detail: {redact(validation_error)[:500]}"
    if messages:
        messages[0]["content"] = str(messages[0].get("content", "")) + reminder
    else:
        messages.append({"role": "system", "content": reminder.strip()})
    messages.append(
        {
            "role": "user",
            "content": (
                "Trusted schema-repair instruction: correct the previous output "
                "to satisfy every verdict, score, category, and note invariant. "
                "Return only the corrected JSON object."
            ),
        }
    )
    retry["messages"] = messages
    if finish_reason == "length":
        requested = request.get("max_tokens", 1024)
        retry["max_tokens"] = min(
            max(int(requested) * 2 if isinstance(requested, int) else 2048, 2048),
            4096,
        )
    return retry


def invoke(
    request: dict[str, Any],
    *,
    candidate: str,
    retries: int,
    backoff: float,
    call: Callable[..., Any] = completion,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[JudgeVerdict | None, list[AttemptResult], str | None]:
    """Call a provider with bounded retry, preserving every outcome separately."""
    attempts: list[AttemptResult] = []
    terminal: str | None = None
    active_request = request
    for number in range(1, retries + 1):
        started_at, started = now(), time.monotonic()
        content: str | None = None
        returned_model: str | None = None
        finish_reason: str | None = None
        usage: dict[str, Any] | None = None
        cost: float | None = None
        try:
            response = call(**active_request)
            choices = getattr(response, "choices", []) or []
            choice = choices[0] if choices else None
            content = response_content(choice)
            returned_model = str(getattr(response, "model", "") or "") or None
            finish_reason = str(getattr(choice, "finish_reason", "") or "") or None
            usage = redact(object_dict(getattr(response, "usage", None)))
            cost = response_cost(response)
            verdict = parse_verdict(content)
            outcome = "validated"
            attempts.append(
                AttemptResult(
                    redact(active_request),
                    redact(content),
                    returned_model,
                    finish_reason,
                    usage,
                    cost,
                    started_at,
                    now(),
                    time.monotonic() - started,
                    outcome,
                    None,
                )
            )
            return verdict, attempts, terminal if verdict is None else None
        except (ValidationError, JSONDecodeError, ValueError) as error:
            terminal = "failed_validation"
            attempts.append(
                AttemptResult(
                    redact(active_request),
                    redact(content) if content is not None else None,
                    returned_model,
                    finish_reason,
                    usage,
                    cost,
                    started_at,
                    now(),
                    time.monotonic() - started,
                    "invalid_provider_output",
                    redact(str(error)),
                )
            )
            active_request = retry_request_after_validation_failure(
                active_request,
                finish_reason=finish_reason,
                validation_error=str(error),
            )
        except Exception as error:  # noqa: BLE001 - LiteLLM provider errors have no stable base class.
            terminal = "failed_provider"
            attempts.append(
                AttemptResult(
                    redact(active_request),
                    None,
                    None,
                    None,
                    None,
                    None,
                    started_at,
                    now(),
                    time.monotonic() - started,
                    "provider_error",
                    redact(str(error)),
                )
            )
            if not is_transient(error):
                break
        if number < retries:
            sleep(backoff * 2 ** (number - 1))
    return None, attempts, terminal


def sampling_design(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe the frozen selection design and its population weights."""
    stratified = [row for row in rows if row.get("_sampling_stratum") is not None]
    if not stratified:
        return {"type": "simple_random"}
    if len(stratified) != len(rows):
        raise JudgeRunError("Evaluation sample mixes weighted and unweighted rows")
    if any(row.get("_sampling_inclusion_prob") is not None for row in stratified):
        return pps_sampling_design(stratified)
    strata: dict[str, dict[str, Any]] = {}
    for row in stratified:
        name = str(row["_sampling_stratum"])
        value = {
            "population": row.get("_sampling_population"),
            "sample": row.get("_sampling_sample_size"),
            "weight": row.get("_sampling_weight"),
        }
        if name in strata and strata[name] != value:
            raise JudgeRunError(f"Inconsistent sampling metadata for {name}")
        strata[name] = value
    return {"type": "disproportionate_score_stratified", "strata": strata}


def pps_sampling_design(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe a stratified sample drawn with probability proportional to size.

    Weights differ between pairs in a stratum, so each stratum records its
    population, sample size and the sum of its design weights.
    """
    strata: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row["_sampling_stratum"])
        value = strata.setdefault(
            name,
            {
                "population": row.get("_sampling_population"),
                "sample": row.get("_sampling_sample_size"),
                "weight_sum": 0.0,
            },
        )
        if (value["population"], value["sample"]) != (
            row.get("_sampling_population"),
            row.get("_sampling_sample_size"),
        ):
            raise JudgeRunError(f"Inconsistent sampling metadata for {name}")
        value["weight_sum"] += float(row["_sampling_weight"])
    return {"type": "stratified_pps", "strata": strata}


def sidecar(output: Path, name: str) -> Path:
    """Return a deterministic run sidecar path."""
    return output.with_name(f"{output.stem}.{name}")


def versions() -> dict[str, str]:
    """Capture dependency versions that affect a run."""
    result = {}
    for package in ("pydantic", "litellm", "instructor", "python-dotenv"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "unavailable"
    return result


def manifest(
    args: argparse.Namespace,
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    calibration_path: Path | None,
) -> dict[str, Any]:
    """Construct immutable run metadata before the first provider call."""
    config = {
        "manifest_version": 7,
        "judge_model": args.model,
        "provider": asdict(JUDGE_MODELS[args.model]),
        "prompt_sha256": digest(JUDGE_SYSTEM_PROMPT.encode()),
        "rubric_sha256": digest(load_human_review_guidelines().encode()),
        "schema_sha256": json_digest(JUDGE_OUTPUT_SCHEMA),
        "evaluation_source": str(Path(args.results).resolve()),
        "response_format": (
            "json_schema"
            if JUDGE_MODELS[args.model].native_schema
            else "strict_tool"
            if JUDGE_MODELS[args.model].provider == "deepseek"
            else None
        ),
        "provider_schema_sha256": (
            json_digest(DEEPSEEK_VERDICT_TOOL_SCHEMA)
            if JUDGE_MODELS[args.model].provider == "deepseek"
            else None
        ),
        "evaluation_source_sha256": digest(Path(args.results).read_bytes()),
        "evaluation_rows_sha256": json_digest(rows),
        "calibration_source": str(calibration_path.resolve()) if calibration_path else None,
        "calibration_source_sha256": digest(calibration_path.read_bytes())
        if calibration_path
        else None,
        "evaluation_seed": args.seed,
        "calibration_seed": args.few_shot_seed,
        "selection_method": args.calibration_selection,
        "evaluation_sampling": sampling_design(rows),
        "few_shot_requested": args.few_shot,
        "few_shot_strategy": args.few_shot_strategy,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "retries": args.retries,
        "expected_identities": ["\u241f".join(identity(row)) for row in rows],
    }
    return {
        "manifest_id": json_digest(config),
        "created_at": now(),
        "status": "pending",
        "output": str(output.resolve()),
        "config": config,
        "dependency_versions": versions(),
        "actual_example_counts": {},
        "cost": "unavailable",
        "latency_seconds": "unavailable",
    }


def write_manifest(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically replace a manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a structurally valid manifest."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, JSONDecodeError) as error:
        raise JudgeRunError(f"Cannot read manifest {path}") from error
    if not isinstance(value, dict) or not isinstance(value.get("config"), dict):
        raise JudgeRunError(f"Invalid manifest {path}")
    return value


def finals(
    output: Path, manifest_id: str, expected: set[tuple[str, str, str]]
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Load only validated, unique finals for this immutable manifest."""
    if not output.exists():
        return {}
    result = {}
    for record in read_jsonl(output):
        if record.get("status") != "final" or record.get("manifest_id") != manifest_id:
            raise ResumeMismatchError("Output contains a mismatched non-final record")
        key = tuple(record.get("identity") or ())
        if len(key) != 3 or key not in expected or key in result:
            raise ResumeMismatchError("Output has unexpected or duplicate identity")
        try:
            JudgeVerdict.model_validate(record.get("verdict"))
        except ValidationError as error:
            raise ResumeMismatchError("Output contains an invalid final verdict") from error
        result[key] = record
    return result


def unresolved_failure_identities(
    path: Path,
    *,
    manifest_id: str | None = None,
    expected: set[tuple[str, str, str]] | None = None,
) -> set[tuple[str, str, str]]:
    """Return failures not superseded by a later successful final judgment."""
    if not path.exists():
        return set()
    unresolved: set[tuple[str, str, str]] = set()
    statuses = {
        "failed_provider",
        "failed_validation",
        "pending_human_review",
        "resolved",
    }
    for record in read_jsonl(path):
        if manifest_id is not None and record.get("manifest_id") != manifest_id:
            raise ResumeMismatchError("Failure log contains a mismatched manifest")
        key = tuple(record.get("identity") or ())
        if len(key) != 3 or (expected is not None and key not in expected):
            raise ResumeMismatchError("Failure log contains an invalid identity")
        status = record.get("status")
        if status not in statuses:
            raise ResumeMismatchError("Failure log contains an invalid status")
        if status == "resolved":
            unresolved.discard(key)
        else:
            unresolved.add(key)
    return unresolved


def output_record(
    row: Mapping[str, Any],
    verdict: JudgeVerdict,
    manifest_id: str,
    examples: Sequence[Mapping[str, Any]],
    request: Mapping[str, Any],
    attempt_ids: Sequence[int],
) -> dict[str, Any]:
    """Build one final result record after local validation succeeds."""
    return {
        "status": "final",
        "manifest_id": manifest_id,
        "identity": list(identity(row)),
        "riddle_id": riddle_id(row),
        "community": row.get("community"),
        "model_name": row.get("model_name"),
        "candidate": candidate_answer(row),
        "reference": row.get("reference_answer_original"),
        "human_score": row.get("score"),
        "human_error_category": row.get("error_type"),
        "sampling_stratum": row.get("_sampling_stratum"),
        "sampling_weight": row.get("_sampling_weight"),
        "few_shot_count": len(examples),
        "few_shot_riddle_ids": [example["riddle_id"] for example in examples],
        "few_shot_communities": [example.get("community") for example in examples],
        "application_messages": request["messages"],
        "provider_request_sha256": json_digest(request),
        "attempt_ids": list(attempt_ids),
        "verdict": verdict.model_dump(),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the judge command interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        help="pairs to judge: CSV, Parquet (including *_unresolved.parquet), or framework results.jsonl",
    )
    parser.add_argument("--model", choices=tuple(JUDGE_MODELS), default=DEFAULT_MODEL)
    parser.add_argument("--human-only", action="store_true")
    parser.add_argument("--sample", type=int)
    parser.add_argument(
        "--stratified-score-allocation",
        type=parse_stratified_score_allocation,
        metavar="0=N,1=N,2=N,3=N",
        help=(
            "disproportionate human-score allocation; stores inverse-probability "
            "weights for population-representative reporting"
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--few-shot", type=int, default=0)
    parser.add_argument("--calibration-input", type=Path)
    parser.add_argument("--few-shot-seed", type=int, default=20260917)
    parser.add_argument(
        "--few-shot-strategy",
        choices=("different_cultures", "same_culture", "fixed"),
        default="different_cultures",
    )
    parser.add_argument("--calibration-selection", choices=("random",), default="random")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent provider requests; results are written by one thread",
    )
    parser.add_argument("--backoff-seconds", type=float, default=1.0)
    parser.add_argument("--out", type=Path, help="JSONL path within data/runs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def default_output(args: argparse.Namespace) -> Path:
    """Return a condition-specific output path."""
    name = f"{Path(args.results).stem}.{args.model}.eval{args.seed}.shots{args.few_shot}.{args.few_shot_strategy}.jsonl"
    return RUNS_DIR / "llm_judge" / name


def main(argv: list[str] | None = None) -> int:
    """Execute a manifest-backed run and fail when coverage is incomplete."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.few_shot < 0 or args.retries < 1 or args.max_tokens < 1:
        parser.error("few-shot must be non-negative; retries and max-tokens must be positive")
    source = Path(args.results)
    if not source.exists():
        parser.error(f"Evaluation input does not exist: {source}")
    output = args.out or default_output(args)
    if not output.resolve().is_relative_to(RUNS_DIR.resolve()):
        parser.error("--out must be within data/runs; canonical inputs are read-only")
    manifest_path, snapshot_path = (
        sidecar(output, "manifest.json"),
        sidecar(output, "inputs.jsonl"),
    )
    attempts_path, failures_path = (
        sidecar(output, "attempts.jsonl"),
        sidecar(output, "failures.jsonl"),
    )
    selected = select_evaluation(
        load_rows(source),
        human_only=args.human_only,
        sample=args.sample,
        seed=args.seed,
        limit=args.limit,
        stratified_score_allocation=args.stratified_score_allocation,
    )
    if args.few_shot and args.calibration_input is None:
        parser.error("--few-shot requires --calibration-input with valid recorded human labels")
    if args.calibration_input and not args.calibration_input.exists():
        parser.error(f"Calibration input does not exist: {args.calibration_input}")
    evaluation_riddles = {riddle_id(row) for row in selected}
    calibration = (
        eligible_calibration(load_rows(args.calibration_input), evaluation_riddles)
        if args.few_shot
        else []
    )
    expected_manifest = manifest(args, output, selected, args.calibration_input)
    if args.resume:
        if not manifest_path.exists() or not snapshot_path.exists():
            parser.error("--resume requires a manifest and frozen input snapshot")
        run_manifest = load_manifest(manifest_path)
        if (
            run_manifest.get("manifest_id") != expected_manifest["manifest_id"]
            or Path(str(run_manifest.get("output"))).resolve() != output.resolve()
        ):
            raise ResumeMismatchError("Resume configuration differs from the frozen manifest")
        if [identity(row) for row in read_jsonl(snapshot_path)] != [
            identity(row) for row in selected
        ]:
            raise ResumeMismatchError("Frozen snapshot differs from requested evaluation selection")
    else:
        if output.exists() or manifest_path.exists() or snapshot_path.exists():
            parser.error("Output already exists; use --resume or a new --out")
        run_manifest = expected_manifest
        write_jsonl(snapshot_path, selected)
        write_manifest(manifest_path, run_manifest)
    expected = {identity(row) for row in selected}
    existing = finals(output, str(run_manifest["manifest_id"]), expected)
    remaining = [row for row in selected if identity(row) not in existing]
    preview = remaining[0] if remaining else selected[0]
    preview_examples = [
        calibration_example(item)
        for item in select_examples(
            preview,
            calibration=calibration,
            evaluation_riddles=evaluation_riddles,
            size=args.few_shot,
            strategy=args.few_shot_strategy,
            seed=args.few_shot_seed,
        )
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "manifest_id": run_manifest["manifest_id"],
                    "remaining": len(remaining),
                    "request": redact(
                        build_provider_request(
                            model=args.model,
                            row=preview,
                            examples=preview_examples,
                            max_tokens=args.max_tokens,
                        )
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get(JUDGE_MODELS[args.model].key_name):
        parser.error(
            f"{JUDGE_MODELS[args.model].key_name} is not set in the environment or root .env"
        )

    def judge(
        row: dict[str, Any],
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, Any],
        JudgeVerdict | None,
        list[AttemptResult],
        str | None,
    ]:
        examples = [
            calibration_example(item)
            for item in select_examples(
                row,
                calibration=calibration,
                evaluation_riddles=evaluation_riddles,
                size=args.few_shot,
                strategy=args.few_shot_strategy,
                seed=args.few_shot_seed,
            )
        ]
        request = build_provider_request(
            model=args.model,
            row=row,
            examples=examples,
            max_tokens=args.max_tokens,
        )
        verdict, attempts, status = invoke(
            request,
            candidate=candidate_answer(row),
            retries=args.retries,
            backoff=args.backoff_seconds,
        )
        return examples, request, verdict, attempts, status

    start = len(read_jsonl(attempts_path)) if attempts_path.exists() else 0
    unresolved_before = unresolved_failure_identities(
        failures_path, manifest_id=str(run_manifest["manifest_id"]), expected=expected
    )
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        outcomes = zip(remaining, pool.map(judge, remaining), strict=True)
        for done, (row, (examples, request, verdict, attempts, status)) in enumerate(outcomes, 1):
            if done % 250 == 0:
                print(f"{done}/{len(remaining)} judged", flush=True)
            attempt_ids = []
            for offset, attempt in enumerate(attempts, 1):
                start += 1
                attempt_id = start
                attempt_ids.append(attempt_id)
                append_jsonl(
                    attempts_path,
                    {
                        "attempt_id": attempt_id,
                        "manifest_id": run_manifest["manifest_id"],
                        "identity": list(identity(row)),
                        **asdict(attempt),
                    },
                )
            if verdict is None:
                append_jsonl(
                    failures_path,
                    {
                        "status": status,
                        "manifest_id": run_manifest["manifest_id"],
                        "identity": list(identity(row)),
                        "attempt_ids": attempt_ids,
                    },
                )
                continue
            append_jsonl(
                output,
                output_record(
                    row,
                    verdict,
                    str(run_manifest["manifest_id"]),
                    examples,
                    request,
                    attempt_ids,
                ),
            )
            if identity(row) in unresolved_before:
                append_jsonl(
                    failures_path,
                    {
                        "status": "resolved",
                        "manifest_id": run_manifest["manifest_id"],
                        "identity": list(identity(row)),
                        "attempt_ids": attempt_ids,
                    },
                )
    completed = finals(output, str(run_manifest["manifest_id"]), expected)
    run_manifest = dict(run_manifest)
    example_counts = Counter(int(record.get("few_shot_count", 0)) for record in completed.values())
    run_manifest["actual_example_counts"] = {
        str(key): value for key, value in sorted(example_counts.items())
    }
    persisted_attempts = read_jsonl(attempts_path) if attempts_path.exists() else []
    measured_latency = [
        attempt["latency_seconds"]
        for attempt in persisted_attempts
        if isinstance(attempt.get("latency_seconds"), (int, float))
    ]
    measured_cost = [
        attempt["cost"]
        for attempt in persisted_attempts
        if isinstance(attempt.get("cost"), (int, float))
    ]
    run_manifest["latency_seconds"] = sum(measured_latency) if measured_latency else "unavailable"
    run_manifest["cost"] = sum(measured_cost) if measured_cost else "unavailable"
    unresolved = unresolved_failure_identities(
        failures_path, manifest_id=str(run_manifest["manifest_id"]), expected=expected
    )
    run_manifest["completion"] = {
        "expected": len(selected),
        "validated_final": len(completed),
        "failed_or_pending": len(unresolved),
    }
    run_manifest["completed_at"] = now()
    run_manifest["status"] = (
        "complete" if len(completed) == len(selected) and not unresolved else "incomplete"
    )
    write_manifest(manifest_path, run_manifest)
    print(
        f"{run_manifest['status']}: {len(completed)}/{len(selected)} validated final judgments -> {output}"
    )
    return 0 if run_manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
