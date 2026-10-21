"""Render registered figures in isolated paper and preprint processes.

Run ``uv run python -m analysis.figures.main`` for both targets, or pass
``--target paper`` / ``--target preprint`` for just one output set.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from analysis.constants import PAPER_FIGURES_DIR, PREPRINT_FIGURES_DIR
from analysis.figure_registry import FIGURES, PREPRINT_FIGURES
from analysis.style import TARGET_ENV


def _figure_stems(target: str = "both") -> dict[Path, set[str]]:
    folders = {}
    if target in {"paper", "both"}:
        folders[PAPER_FIGURES_DIR] = set(FIGURES)
    if target in {"preprint", "both"}:
        folders[PREPRINT_FIGURES_DIR] = set(FIGURES) | set(PREPRINT_FIGURES)
    return folders


def _require_outputs(suffixes: tuple[str, ...], target: str = "both") -> None:
    missing = [
        folder / f"{stem}{suffix}"
        for folder, stems in _figure_stems(target).items()
        for stem in sorted(stems)
        for suffix in suffixes
        if not (folder / f"{stem}{suffix}").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing generated figures: {missing}")


def _prune_outputs(target: str = "both") -> None:
    """Retire unregistered renders only after the selected outputs are complete."""
    _require_outputs((".pdf", ".png"), target)
    for folder, stems in _figure_stems(target).items():
        for path in folder.glob("fig_*"):
            if path.suffix in {".pdf", ".png"} and path.stem not in stems:
                path.unlink()


def run(module: str, target: str) -> None:
    """Import and render a module in a fresh process with an explicit target."""
    subprocess.run(
        [sys.executable, "-m", module],
        env={**os.environ, TARGET_ENV: target},
        check=True,
    )


def main(target: str = "both") -> None:
    """Render the paper figures, the preprint figures, or both."""
    if target not in {"paper", "preprint", "both"}:
        raise ValueError(f"Unknown figure target: {target!r}")
    targets = ("paper", "preprint") if target == "both" else (target,)
    for selected in targets:
        registry = FIGURES if selected == "paper" else {**FIGURES, **PREPRINT_FIGURES}
        for module in dict.fromkeys(registry.values()):
            run(module, selected)
    _prune_outputs(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("paper", "preprint", "both"), default="both")
    main(parser.parse_args().target)
