"""Build a checksummed, self-contained reproduction archive after regeneration."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from scripts.release_support import ROOT, sha256, validate_data

ROOT_NAME = "culturalriddles"
SOURCE_DIRS = (
    "analysis",
    "dashboard",
    "data",
    "docs",
    "figures",
    "benchmark",
    "human_eval",
    "scripts",
    "submissions",
    ".github",
)
ROOT_FILES = (
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    ".pre-commit-config.yaml",
    ".gitignore",
    ".env.template",
    "README.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "LICENSE",
)
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".cache"}


def is_regenerable_derived(relative: Path) -> bool:
    """Files under data/derived that analysis.run_all rebuilds and nothing reads from a release.

    The two model input matrices under build/ are rebuilt on every run and verified against the
    hashes in the fit provenance records. The serialized R model objects (.rds) are written only
    by --fit.
    """
    return relative.parts[2:3] == ("build",) or relative.suffix == ".rds"


def is_public_data_path(relative: Path, *, is_directory: bool = False) -> bool:
    """Allow only the reviewed public data trees and named release files."""
    parts = relative.parts
    if is_directory and relative.as_posix() in {"data", "data/submissions"}:
        return True
    if len(parts) < 2 or parts[0] != "data" or ".." in parts:
        return False
    if parts[1] in {"canonical", "reference", "manifests"}:
        return len(parts) > 2 or is_directory
    if parts[1] == "derived":
        return (
            (len(parts) > 2 or is_directory)
            and parts[2:3] != ("submissions",)
            and not is_regenerable_derived(relative)
        )
    return relative.as_posix() in {
        "data/corrections.csv",
        "data/README.md",
        "data/LICENSE",
        "data/submissions/submission_template.xlsx",
    }


def release_files(root: Path) -> list[Path]:
    files = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for name in SOURCE_DIRS:
        for path in (root / name).rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root)
            if EXCLUDED_PARTS.intersection(relative.parts) or any(
                part.endswith(".egg-info") for part in relative.parts
            ):
                continue
            if (
                relative.parts[0] == "data" and not is_public_data_path(relative)
            ) or relative.as_posix() == "dashboard/index.html":
                continue
            if (
                path.name.startswith(".env")
                or path.name == ".DS_Store"
                or path.suffix in {".pyc", ".log"}
            ):
                continue
            files.append(path)
    return sorted(set(files))


def generated_artifacts(root: Path) -> list[Path]:
    from analysis.figure_registry import FIGURES, PREPRINT_FIGURES, TABLES

    paths = []
    for folder, stems in (
        ("paper", set(FIGURES)),
        ("preprint", set(FIGURES) | set(PREPRINT_FIGURES)),
    ):
        directory = root / "figures" / folder
        expected = {f"{stem}.{suffix}" for stem in stems for suffix in ("pdf", "png")}
        actual = {path.name for path in directory.iterdir() if path.is_file()}
        if actual != expected:
            raise ValueError(
                f"Figure inventory differs in {folder}: missing={expected - actual}, unexpected={actual - expected}"
            )
        paths += [directory / name for name in sorted(expected)]
    table_dir = root / "data/derived/tables"
    expected_tables = [table_dir / f"{name}.tex" for name in TABLES]
    macros = root / "data/derived/claims/macros.tex"
    for path in [*expected_tables, macros]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing generated table or macros: {path}")
    paths += [
        path
        for path in (root / "data/derived").rglob("*")
        if path.is_file()
        and path.suffix in {".csv", ".json", ".parquet", ".tex"}
        and "submissions" not in path.relative_to(root / "data/derived").parts
        and not is_regenerable_derived(path.relative_to(root))
    ]
    return sorted(set(paths))


def build(out: Path, root: Path = ROOT) -> None:
    inputs = validate_data(root)
    artifacts = generated_artifacts(root)
    files = release_files(root)
    manifest = {
        "format": 1,
        "root": ROOT_NAME,
        "inputs": inputs,
        "artifacts": [path.relative_to(root).as_posix() for path in artifacts],
        "files": [
            {"path": path.relative_to(root).as_posix(), "sha256": sha256(path)} for path in files
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            _write_member(archive, path.relative_to(root).as_posix(), path.read_bytes())
        _write_member(
            archive,
            "release_manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
    print(f"Wrote {out}: {len(files)} files, {len(artifacts)} registered analytical artifacts")


def _write_member(archive: zipfile.ZipFile, relative: str, content: bytes | str) -> None:
    info = zipfile.ZipInfo(f"{ROOT_NAME}/{relative}", date_time=(2020, 1, 1, 0, 0, 0))
    info.external_attr = 0o100644 << 16
    archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "dist/culturalriddles-repro.zip")
    build(parser.parse_args().out)


if __name__ == "__main__":
    main()
