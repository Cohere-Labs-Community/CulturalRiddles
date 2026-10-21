"""Summarize the broad themes communities chose for their two wildcard topics.

Writes ``data/derived/riddle_analysis/wildcard_themes.csv``: for each theme, the number and share of
the wildcard selections (two per community) and of the communities that chose
it. The paper's Topics paragraph cites these shares.

Run from the repository root::

    uv run python -m analysis.riddle_analysis.wildcard_themes
"""

from __future__ import annotations

import pandas as pd

from analysis.riddle_analysis.corpus_data import (
    TABLE_DIR,
    load_corpus,
    wildcard_choices,
    wildcard_theme,
)
from analysis.utils import write_csv


def theme_summary(corpus: pd.DataFrame) -> pd.DataFrame:
    """Count wildcard selections and communities per broad theme.

    Args:
        corpus: Canonical riddle corpus.

    Returns:
        One row per theme, most selected first.
    """
    choices = wildcard_choices(corpus)
    choices["theme"] = choices["topic"].map(wildcard_theme)
    summary = (
        choices.groupby("theme")
        .agg(
            wildcard_selections=("community", "size"),
            n_communities=("community", "nunique"),
        )
        .sort_values("wildcard_selections", ascending=False)
        .reset_index()
    )
    summary["share_of_wildcard_selections"] = summary["wildcard_selections"] / len(choices)
    summary["share_of_communities"] = summary["n_communities"] / choices["community"].nunique()
    return summary


def main() -> None:
    """Write the wildcard-theme summary table."""
    write_csv(theme_summary(load_corpus()), TABLE_DIR / "wildcard_themes.csv")


if __name__ == "__main__":
    main()
