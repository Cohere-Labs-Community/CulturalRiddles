"""Inventory approved inputs and enforce the public data/privacy contract.

Run without arguments to validate. Maintainers use --freeze after reviewing a
new public snapshot; this records schemas, row counts, keys, and content hashes.
Free-text content still requires human privacy review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.json as pajson
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = Path("data/manifests/canonical.json")
ANONYMOUS_ID_PATTERN = r"annotator_[a-z0-9]+(?:\s*;\s*annotator_[a-z0-9]+)*"
FIT_CACHE_FILES = {
    "variance_components.csv",
    "tier_effects.csv",
    "riddle_feature_effects.csv",
    "riddle_feature_variance_explained.csv",
}
IDENTITY_KEYS = {
    "annotator_name",
    "author_credit",
    "human_validator_name",
    "your_discord",
    "respondent_name",
    "respondent_email",
    "email",
    "full_name",
    "real_name",
    "email_address",
    "discord",
    "discord_handle",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_identifying_key(key: object) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
    return (
        normalized in IDENTITY_KEYS
        or normalized.endswith(("_email", "_discord"))
        or normalized.endswith("_name")
        and any(
            token in normalized
            for token in ("annotator", "respondent", "author", "validator", "writer", "contributor")
        )
    )


def _identity_keys(value: object, prefix: str = "") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, nested in value.items():
            location = f"{prefix}.{key}" if prefix else str(key)
            if _is_identifying_key(key):
                found.append(location)
            found.extend(_identity_keys(nested, location))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found.extend(_identity_keys(nested, f"{prefix}[{index}]"))
    return found


def _records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    content = json.loads(path.read_text(encoding="utf-8"))
    return content if isinstance(content, list) else [content]


def _json_schema(value: object) -> object:
    if isinstance(value, dict):
        return {key: _json_schema(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        variants = {_schema_digest(_json_schema(item)): _json_schema(item) for item in value}
        return [variants[key] for key in sorted(variants)]
    return type(value).__name__


def _schema_digest(schema: object) -> str:
    return hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _canonical_schema(path: Path) -> object:
    if path.suffix == ".parquet":
        return [[field.name, str(field.type)] for field in pq.read_schema(path)]
    if path.suffix == ".csv":
        frame = pd.read_csv(path)
        return [[name, str(dtype)] for name, dtype in frame.dtypes.items()]
    if path.suffix == ".jsonl":
        return [[field.name, str(field.type)] for field in pajson.read_json(path).schema]
    if path.suffix in {".json", ".geojson"}:
        return _json_schema(json.loads(path.read_text(encoding="utf-8")))
    return {"format": path.suffix.lstrip("."), "encoding": "utf-8"}


def public_inputs(root: Path) -> list[Path]:
    paths = [
        path
        for directory in (root / "data/canonical", root / "data/reference")
        for path in directory.rglob("*")
        if path.is_file() and path.name != ".DS_Store"
    ]
    ledger = root / "data/corrections.csv"
    if ledger.is_file():
        paths.append(ledger)
    return sorted(paths)


def _primary_key(path: str, columns: list[str]) -> list[str]:
    if path.endswith("corrections.csv"):
        return ["riddle_id", "action", "field", "old_value", "new_value"]
    if path.endswith("model_mapping.csv"):
        return ["source_name"]
    if path.endswith("annotations.parquet"):
        return ["riddle_id", "answer_norm", "annotation"]
    if path.endswith("runs.json") and "setting" in columns:
        return ["setting", "shots", "judge"]
    for keys in (
        ["shots", "judge", "model", "riddle_id", "answer_norm"],
        ["shots", "judge", "riddle_id", "answer_norm"],
        ["riddle_id", "model"],
        ["riddle_id", "answer_norm"],
        ["riddle_id"],
        ["community"],
        ["language_code"],
        ["source_name"],
        ["contribution"],
        ["model"],
        ["topic"],
        ["strategy"],
        ["resource"],
        ["use"],
        ["metric"],
    ):
        if all(key in columns for key in keys):
            return keys
    return []


def inspect_input(root: Path, path: Path) -> dict:
    relative = path.relative_to(root).as_posix()
    frame = None
    if path.suffix == ".parquet":
        frame = pq.read_table(path).to_pandas()
    elif path.suffix == ".csv":
        frame = pd.read_csv(path, keep_default_na=False)
    elif path.suffix in {".json", ".jsonl", ".geojson"}:
        records = _records(path)
        for index, row in enumerate(records):
            if found := _identity_keys(row):
                raise ValueError(f"Identifying JSON keys in {relative}, row {index}: {found}")
        if all(isinstance(row, dict) for row in records):
            frame = pd.DataFrame(records)
    if frame is not None:
        for column in frame:
            if _is_identifying_key(column):
                raise ValueError(f"Identifying column {column!r} in {relative}")
            if column in {"annotator_id", "annotator_1_id", "annotator_2_id"}:
                values = frame[column].dropna().astype(str)
                values = values[values.ne("")]
                if not values.str.fullmatch(ANONYMOUS_ID_PATTERN).all():
                    raise ValueError(f"Non-anonymous annotator IDs in {relative}")
        if "riddle_writing_analysis" in path.parts:
            from analysis.riddle_writing_analysis.survey_data import AGGREGATE_COLUMNS

            approved_columns = AGGREGATE_COLUMNS.get(path.name)
            if approved_columns is None or tuple(frame.columns) != approved_columns:
                raise ValueError(
                    f"Survey releases must contain approved aggregate information only: {relative}"
                )
            if path.name == "survey_audit.csv" and len(frame) != 1:
                raise ValueError("The survey audit must contain one aggregate row")
        primary_key = _primary_key(relative, list(frame))
        if primary_key and frame.duplicated(primary_key).any():
            raise ValueError(f"Duplicate primary key in {relative}: {primary_key}")
        rows = len(frame)
    else:
        primary_key, rows = [], None
    schema = _canonical_schema(path)
    return {
        "path": relative,
        "sha256": sha256(path),
        "rows": rows,
        "schema": schema,
        "schema_sha256": _schema_digest(schema),
        "primary_key": primary_key,
        "role": "recorded public input"
        if "/canonical/" in relative
        else "reference input or correction ledger",
    }


def freeze(root: Path, snapshot: str) -> list[dict]:
    records = [inspect_input(root, path) for path in public_inputs(root)]
    if not records:
        raise ValueError("Public data inventory mismatch: no canonical inputs")
    path = root / CONTRACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"snapshot": snapshot, "files": records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return records


def validate_data(root: Path, contract: dict | None = None) -> list[dict]:
    """Reject changed inventories, schemas, counts, keys, identifiers, or bytes."""
    if contract is None:
        path = root / CONTRACT
        if not path.is_file():
            raise ValueError("Public data inventory mismatch: no approved canonical manifest")
        contract = json.loads(path.read_text(encoding="utf-8"))
    expected = {record["path"]: record for record in contract["files"]}
    actual_paths = {path.relative_to(root).as_posix(): path for path in public_inputs(root)}
    if expected.keys() != actual_paths.keys():
        raise ValueError(
            f"Public data inventory mismatch; missing={sorted(expected.keys() - actual_paths.keys())}, unexpected={sorted(actual_paths.keys() - expected.keys())}"
        )
    records = []
    for relative, path in actual_paths.items():
        record = inspect_input(root, path)
        approved = expected[relative]
        if record["schema_sha256"] != approved["schema_sha256"]:
            raise ValueError(f"Schema drift in {relative}")
        if record["rows"] != approved["rows"]:
            raise ValueError(f"Row count mismatch in {relative}")
        if record["sha256"] != approved["sha256"]:
            raise ValueError(f"Input content differs from approved snapshot: {relative}")
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze", action="store_true", help="Record a manually reviewed public snapshot"
    )
    parser.add_argument("--snapshot", default="published", help="Snapshot identifier for --freeze")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    records = freeze(args.root, args.snapshot) if args.freeze else validate_data(args.root)
    print(
        f"{'Recorded' if args.freeze else 'Validated'} {len(records)} approved public input files"
    )


if __name__ == "__main__":
    main()
