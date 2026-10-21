"""Fit cache lineage is checked against real files without invoking R."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from analysis.difficulty_decomposition.fit_provenance import (
    FIT_OUTPUT_NAMES,
    record_fit_provenance,
    validate_fit_provenance,
)


@pytest.fixture(params=tuple(FIT_OUTPUT_NAMES))
def cached_fit(request, tmp_path: Path):
    group = request.param
    frame = tmp_path / "build" / "frame.csv"
    frame.parent.mkdir()
    frame.write_text("correct,model_name,riddle_id\n1,model_a,riddle_1\n", encoding="utf-8")
    script = tmp_path / "fit.R"
    script.write_text("formula <- correct ~ 1 + (1 | riddle_id)\n", encoding="utf-8")
    output = tmp_path / "derived" / "difficulty_decomposition"
    output.mkdir(parents=True)
    for name in FIT_OUTPUT_NAMES[group]:
        (output / name).write_text("term,estimate\nintercept,0.25\n", encoding="utf-8")
    kwargs = {"frame_path": frame, "script_path": script, "output_dir": output}
    provenance = record_fit_provenance(group, **kwargs)
    paths = {
        "frame": frame,
        "script": script,
        "first_output": output / FIT_OUTPUT_NAMES[group][0],
        "second_output": output / FIT_OUTPUT_NAMES[group][1],
        "provenance": provenance,
    }
    return group, kwargs, paths


def test_valid_replay_preserves_signatures_and_files(cached_fit) -> None:
    group, kwargs, paths = cached_fit
    before = {
        name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for name, path in paths.items()
    }
    recorded = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    assert recorded["input_frame"]["sha256"] == before["frame"][0]
    assert recorded["r_script"]["sha256"] == before["script"][0]
    for role in ("first_output", "second_output"):
        assert recorded["fit_csvs"][paths[role].name] == before[role][0]

    validate_fit_provenance(group, **kwargs)
    validate_fit_provenance(group, **kwargs)

    assert {
        name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for name, path in paths.items()
    } == before


@pytest.mark.parametrize("role", ["frame", "script", "first_output", "second_output"])
def test_replay_rejects_changed_fit_files(cached_fit, role: str) -> None:
    group, kwargs, paths = cached_fit
    provenance_before = paths["provenance"].read_bytes()
    paths[role].write_bytes(paths[role].read_bytes() + b"\nchanged\n")

    with pytest.raises(RuntimeError, match=rf"Stale fit provenance.*{group} --fit"):
        validate_fit_provenance(group, **kwargs)

    assert paths["provenance"].read_bytes() == provenance_before


@pytest.mark.parametrize("role", ["frame", "script", "first_output", "second_output"])
def test_replay_rejects_missing_fit_files(cached_fit, role: str) -> None:
    group, kwargs, paths = cached_fit
    provenance_before = paths["provenance"].read_bytes()
    paths[role].unlink()

    with pytest.raises(RuntimeError, match=rf"Missing fit input or summary.*{group} --fit"):
        validate_fit_provenance(group, **kwargs)

    assert not paths[role].exists()
    assert paths["provenance"].read_bytes() == provenance_before


def test_replay_rejects_missing_provenance_without_blessing_caches(cached_fit) -> None:
    group, kwargs, paths = cached_fit
    paths["provenance"].unlink()

    with pytest.raises(RuntimeError, match=rf"Missing fit provenance.*{group} --fit"):
        validate_fit_provenance(group, **kwargs)

    assert not paths["provenance"].exists()
