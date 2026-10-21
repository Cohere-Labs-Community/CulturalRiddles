"""Per-community, per-model accuracy shared by the community figures."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.corpus import load_corpus_metadata
from analysis.model_results.evaluation_data import (
    cell_accuracy,
    human_evaluated_communities,
    load_flat_frame,
)

# Thresholds on the human-reviewed share of a cell's non-exact-match answers
# (the rest are scored by two agreeing LLM judges or still pending). Calibrated
# against the judge study: at 50% or more human review, the judge-attributable
# error in a cell's accuracy (median 0.5 points) rarely exceeds its binomial
# sampling error; below 20%, it exceeds the sampling error in over half of
# cells.
LOW_REVIEW_COVERAGE = 0.20
HIGH_REVIEW_COVERAGE = 0.50


def culture_labels(metadata: pd.DataFrame) -> pd.Series:
    """Return labels that distinguish duplicate language-country combinations."""
    base = metadata["language"] + " (" + metadata["country"] + ")"
    return base.where(~base.duplicated(keep=False), base + " · " + metadata["community"])


def accuracy_inputs(all_communities: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return sorted community metadata and per-model accuracy values.

    Accuracy comes from :func:`cell_accuracy`, so the community set and the
    per-cell value match the model-performance table.

    Returns:
        Community metadata sorted by mean accuracy, and one row per community
        and model with ``human_correct`` and ``human_coverage``.
        ``all_communities`` adds the communities without human labels.
    """
    frame = load_flat_frame()
    selected = set(frame["community"]) if all_communities else human_evaluated_communities(frame)
    per_model = cell_accuracy(frame, selected).rename(columns={"value": "human_correct"})
    metadata = load_corpus_metadata().copy()
    metadata["culture"] = culture_labels(metadata)
    metadata["mean_human_correct"] = metadata["community"].map(
        per_model.groupby("community")["human_correct"].mean()
    )
    metadata = metadata.sort_values(
        ["mean_human_correct", "culture"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    return metadata, per_model


def community_values(
    selected: pd.DataFrame, per_model: pd.DataFrame
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return per-model accuracy values and human-reviewed share by community."""
    grouped = dict(tuple(per_model.groupby("community")))
    rows = [grouped[community] for community in selected["community"]]
    return (
        [row["human_correct"].to_numpy() for row in rows],
        [row["human_share"].to_numpy() for row in rows],
    )


Selection = tuple[pd.DataFrame, list[np.ndarray], list[np.ndarray], np.ndarray]


def top_bottom_selection(count: int) -> Selection:
    """Return top/bottom communities, values, coverages, and x positions."""
    metadata, per_model = accuracy_inputs()
    metadata = metadata.loc[metadata["mean_human_correct"].notna()].reset_index(drop=True)
    selected = pd.concat(
        [metadata.head(count), metadata.tail(count)],
        ignore_index=True,
    )
    values, coverage = community_values(selected, per_model)
    return selected, values, coverage, np.arange(1, len(selected) + 1)
