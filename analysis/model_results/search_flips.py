"""Classify the errors GPT-5.5 makes with web search on riddles it solves without.

Writes ``data/derived/model_results/search_flips.csv``: for riddles in the
human-evaluated communities that GPT-5.5 without search answers correctly and
GPT-5.5 with search answers incorrectly, the share of each human-labeled error
type among the search answers.

Run from the repository root::

    uv run python -m analysis.model_results.search_flips
"""

from __future__ import annotations

import pandas as pd

from analysis.constants import MODEL_RESULTS_OUTPUT_DIR
from analysis.model_results.evaluation_data import (
    human_evaluated_communities,
    load_flat_frame,
)
from analysis.utils import write_csv

WITHOUT_SEARCH = "GPT5.5-none"
WITH_SEARCH = "GPT5.5-search"


def search_flip_error_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Share of each human error type among answers that search made wrong.

    Args:
        frame: Flat evaluation frame from :func:`load_flat_frame`.

    Returns:
        One row per error type with ``flips`` and ``share`` (percent of the
        human-labeled flips), largest first.
    """
    frame = frame[frame["community"].isin(human_evaluated_communities(frame))]
    pair = (
        frame[frame["model_name"].eq(WITHOUT_SEARCH)]
        .set_index("riddle_id")[["is_scored", "is_correct"]]
        .join(
            frame[frame["model_name"].eq(WITH_SEARCH)].set_index("riddle_id")[
                ["is_scored", "is_correct", "is_human", "error_type"]
            ],
            lsuffix="_without",
            rsuffix="_with",
            how="inner",
        )
    )
    flips = pair[
        pair["is_scored_without"]
        & pair["is_scored_with"]
        & pair["is_correct_without"]
        & ~pair["is_correct_with"]
        & pair["is_human"]
    ]
    counts = flips["error_type"].round(1).value_counts().rename("flips")
    return (
        counts.rename_axis("error_type")
        .reset_index()
        .assign(share=lambda d: d["flips"] / d["flips"].sum() * 100)
    )


def main() -> None:
    """Write the search-flip error-type table."""
    write_csv(
        search_flip_error_types(load_flat_frame()),
        MODEL_RESULTS_OUTPUT_DIR / "search_flips.csv",
    )


if __name__ == "__main__":
    main()
