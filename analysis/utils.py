"""Small cross-cutting helpers shared by analysis suites."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_csv(frame: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    """Write a tabular artifact, creating its parent directory first.

    Args:
        frame: Table to serialize.
        path: Destination CSV path.
        index: Whether to include the pandas index in the output.

    Returns:
        The destination path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=index)
    return destination
