"""Bind the two groups of retained R fit summaries to their actual fit inputs.

Only a successful R fit may call :func:`record_fit_provenance`. Replay calls
:func:`validate_fit_provenance`; it never creates or refreshes the record.
No R or package versions are inferred from the machine replaying the summaries.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

FitGroup = Literal["prepare", "riddle_features"]
FIT_OUTPUT_NAMES = {
    "prepare": ("variance_components.csv", "tier_effects.csv"),
    "riddle_features": (
        "riddle_feature_effects.csv",
        "riddle_feature_variance_explained.csv",
    ),
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _signature(group: FitGroup, *, frame_path: Path, script_path: Path, output_dir: Path) -> dict:
    return {
        "schema_version": 1,
        "group": group,
        "input_frame": {"filename": frame_path.name, "sha256": _sha256(frame_path)},
        "r_script": {"filename": script_path.name, "sha256": _sha256(script_path)},
        "fit_csvs": {name: _sha256(output_dir / name) for name in FIT_OUTPUT_NAMES[group]},
    }


def record_fit_provenance(
    group: FitGroup, *, frame_path: Path, script_path: Path, output_dir: Path
) -> Path:
    """Record hashes only after this group's actual Rscript fit has succeeded.

    The caller must leave the fitted frame and script unchanged while fitting.
    This explicit helper also supports recording a fit started before provenance
    recording was added. It must not be used to bless unverified existing caches.

    Args:
        group: ``prepare`` or ``riddle_features``, matching the fitting module.
        frame_path: The CSV frame passed to the successful Rscript invocation.
        script_path: The R script passed to that invocation (including formulas).
        output_dir: Directory containing this group's two fitted summary CSVs.

    Returns:
        The written ``<group>_fit_provenance.json`` path in ``output_dir``.
    """
    signature = _signature(
        group, frame_path=frame_path, script_path=script_path, output_dir=output_dir
    )
    path = output_dir / f"{group}_fit_provenance.json"
    path.write_text(json.dumps(signature, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_fit_provenance(
    group: FitGroup, *, frame_path: Path, script_path: Path, output_dir: Path
) -> None:
    """Reject absent or stale summaries without invoking R or updating records.

    Raises:
        RuntimeError: If provenance, a fit input, or a summary is absent, or if
            any recorded hash differs. The error identifies the refit command.
    """
    path = output_dir / f"{group}_fit_provenance.json"
    refit = f"uv run python -m analysis.difficulty_decomposition.{group} --fit"
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Missing fit provenance: {path}. Refit with `{refit}`.") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid fit provenance: {path}. Refit with `{refit}`.") from error
    try:
        current = _signature(
            group, frame_path=frame_path, script_path=script_path, output_dir=output_dir
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Missing fit input or summary: {error.filename}. Refit with `{refit}`."
        ) from error
    if recorded != current:
        raise RuntimeError(
            f"Stale fit provenance for {group}: input frame, R script, or fit CSV "
            f"hashes differ from {path}. Refit with `{refit}`."
        )
