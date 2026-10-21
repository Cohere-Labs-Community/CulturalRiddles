"""Extract a public bundle, install its frozen root environment, and rebuild it.

No path in the source checkout is used by the extracted reproduction. Cached R
fits are replayed by default; --fit reruns the statistical models with R/lme4.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from analysis.difficulty_decomposition.fit_provenance import FIT_OUTPUT_NAMES
from scripts.build_release import ROOT_NAME, generated_artifacts, is_public_data_path
from scripts.release_support import FIT_CACHE_FILES, ROOT, sha256, validate_data

_FIT_PROVENANCE_FILES = {f"{group}_fit_provenance.json" for group in FIT_OUTPUT_NAMES}


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    for item in archive.infolist():
        path = Path(item.filename)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in item.filename
            or not path.parts
            or path.parts[0] != ROOT_NAME
        ):
            raise ValueError(f"Unsafe archive member: {item.filename}")
        if stat.S_ISLNK(item.external_attr >> 16):
            raise ValueError(f"Unsafe archive symlink: {item.filename}")
        relative = Path(*path.parts[1:])
        if relative.parts[:1] == ("data",) and not is_public_data_path(
            relative, is_directory=item.is_dir()
        ):
            raise ValueError(f"Unapproved archive data member: {item.filename}")
        target = (destination / path).resolve()
        if not target.is_relative_to(destination.resolve()):
            raise ValueError(f"Unsafe archive member: {item.filename}")
    archive.extractall(destination)


def _check_bundle_files(root: Path, manifest: dict) -> None:
    expected = {record["path"]: record["sha256"] for record in manifest["files"]}
    for relative in expected:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or "\\" in relative:
            raise ValueError(f"Unsafe manifest path: {relative}")
        if path.parts[:1] == ("data",) and not is_public_data_path(path):
            raise ValueError(f"Unapproved bundle data path: {relative}")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()} - {
        "release_manifest.json"
    }
    if actual != expected.keys():
        raise ValueError(
            f"Bundle file inventory mismatch: missing={expected.keys() - actual}, unexpected={actual - expected.keys()}"
        )
    for relative, digest in expected.items():
        if sha256(root / relative) != digest:
            raise ValueError(f"Bundle checksum mismatch: {relative}")


def _compare_json(expected: object, actual: object, location: str, tolerance: float) -> None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual or type(expected) is not type(actual):
            raise AssertionError(f"JSON value mismatch: {location}")
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not (math.isnan(expected) and math.isnan(actual)) and not math.isclose(
            expected, actual, rel_tol=0, abs_tol=tolerance
        ):
            raise AssertionError(f"JSON number mismatch at {location}: {expected} != {actual}")
    elif isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            raise AssertionError(f"JSON keys differ: {location}")
        for key in expected:
            _compare_json(expected[key], actual[key], f"{location}.{key}", tolerance)
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise AssertionError(f"JSON list length differs: {location}")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _compare_json(left, right, f"{location}[{index}]", tolerance)
    elif expected != actual:
        raise AssertionError(f"JSON value differs at {location}: {expected!r} != {actual!r}")


def _compare_artifact(
    expected: Path, actual: Path, tolerance: float, *, refit: bool = False
) -> None:
    if not actual.is_file():
        raise FileNotFoundError(f"Regeneration omitted {actual}")
    if actual.suffix in {".csv", ".parquet"}:
        reader = pd.read_parquet if actual.suffix == ".parquet" else pd.read_csv
        left, right = reader(expected), reader(actual)
        assert_frame_equal(
            left, right, check_dtype=False, check_exact=False, rtol=0, atol=tolerance
        )
    elif actual.suffix == ".json":
        before = json.loads(expected.read_text())
        after = json.loads(actual.read_text())
        if refit and actual.name in _FIT_PROVENANCE_FILES:
            # Solver-level numerical differences change CSV hashes. Verify the
            # new binding, compare its inputs exactly, and compare CSV values
            # separately using the full-fit numerical tolerance.
            if before["fit_csvs"].keys() != after["fit_csvs"].keys():
                raise AssertionError(f"Fit summary inventory differs: {actual}")
            for name, digest in after["fit_csvs"].items():
                if sha256(actual.parent / name) != digest:
                    raise AssertionError(
                        f"Refitted summary checksum differs: {actual.parent / name}"
                    )
            after["fit_csvs"] = before["fit_csvs"]
        _compare_json(before, after, str(actual), tolerance)
    elif actual.suffix == ".tex":
        if expected.read_bytes() != actual.read_bytes():
            raise AssertionError(f"Generated table or macro differs: {actual}")
    elif actual.suffix == ".png":
        import matplotlib.image as image

        pixels = image.imread(actual)
        if (
            pixels.ndim != 3
            or min(pixels.shape[:2]) < 100
            or pixels[..., :3].max() == pixels[..., :3].min()
        ):
            raise AssertionError(f"Empty or invalid rendered figure: {actual}")
    elif actual.suffix == ".pdf" and not actual.read_bytes().startswith(b"%PDF-"):
        raise AssertionError(f"Invalid PDF: {actual}")


def verify(bundle: Path, fit: bool = False) -> None:
    with tempfile.TemporaryDirectory(prefix="culturalriddles-repro-") as temporary:
        scratch = Path(temporary)
        with zipfile.ZipFile(bundle) as archive:
            _safe_extract(archive, scratch)
        root = scratch / ROOT_NAME
        manifest = json.loads((root / "release_manifest.json").read_text())
        _check_bundle_files(root, manifest)
        validate_data(root)
        expected = scratch / "expected"
        for relative in manifest["artifacts"]:
            destination = expected / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, destination)
        # Preserve the four R summary CSVs and their two input-bound provenance
        # records. Everything else, including both matrices, must be rebuilt.
        derived = root / "data/derived"
        for path in derived.rglob("*"):
            if path.is_file() and not (
                not fit
                and path.parent.name == "difficulty_decomposition"
                and path.name in FIT_CACHE_FILES | _FIT_PROVENANCE_FILES
            ):
                path.unlink()
        shutil.rmtree(root / "figures")
        environment = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "VIRTUAL_ENV",
                "PYTHONPATH",
                "PYTHONHOME",
                "UV_PROJECT",
                "UV_PROJECT_ENVIRONMENT",
                "UV_WORKING_DIRECTORY",
            }
        }
        subprocess.run(["uv", "sync", "--frozen"], cwd=root, env=environment, check=True)

        def run(*module_and_args: str) -> None:
            subprocess.run(
                ["uv", "run", "--frozen", "python", "-m", *module_and_args],
                cwd=root,
                env=environment,
                check=True,
            )

        if fit:  # the slow R fits are separate from run_all; run_all then replays them
            for stage in ("scoring", "corpus"):
                run("analysis.run_all", "--stage", stage)
            run("analysis.difficulty_decomposition.prepare", "--fit")
            run("analysis.difficulty_decomposition.riddle_features", "--fit")
        run("analysis.run_all")
        for module in ("analysis.export_dashboard", "dashboard.compute", "dashboard.build_static"):
            run(module)
        actual_artifacts = {path.relative_to(root).as_posix() for path in generated_artifacts(root)}
        if actual_artifacts != set(manifest["artifacts"]):
            raise AssertionError("Regenerated artifact inventory differs")
        tolerance = 1e-6 if fit else 1e-8
        for relative in manifest["artifacts"]:
            _compare_artifact(expected / relative, root / relative, tolerance, refit=fit)
        validate_data(root)
        print(
            f"Verified isolated {'full-fit' if fit else 'cached-fit'} reproduction: {len(actual_artifacts)} artifacts; absolute numeric tolerance {tolerance:g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bundle", type=Path, nargs="?", default=ROOT / "dist/culturalriddles-repro.zip"
    )
    parser.add_argument("--fit", action="store_true")
    args = parser.parse_args()
    verify(args.bundle, args.fit)


if __name__ == "__main__":
    main()
