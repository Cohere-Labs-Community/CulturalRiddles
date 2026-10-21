"""Explore accuracy by cultural, regional, and resource groupings."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.constants import EVAL_ANALYSIS_OUTPUT_DIR, SCORED_RESPONSES_PATH
from analysis.corpus import load_corpus_metadata
from analysis.model_results.evaluation_data import (
    MIN_HUMAN_RESPONSES,
    eligible_communities,
    load_human_evaluation,
)
from analysis.utils import write_csv

OUTPUT_DIR = EVAL_ANALYSIS_OUTPUT_DIR
RESOURCEDNESS_ORDER: tuple[str, ...] = (
    "extremely-low",
    "low",
    "medium",
    "high",
)


BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261001


def add_group_labels(metadata: pd.DataFrame) -> pd.DataFrame:
    """Validate and expose the classification columns from corpus metadata."""
    result = metadata.copy()
    required = {"un_m49_subregion", "language_resourcedness"}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"Corpus metadata is missing classification columns: {sorted(missing)}")
    return result


def community_accuracy(
    path: Path = SCORED_RESPONSES_PATH, min_human_responses: int = MIN_HUMAN_RESPONSES
) -> pd.DataFrame:
    """Return one row per eligible community with combined score-3 accuracy.

    Eligibility requires at least ``min_human_responses`` human-graded rows
    (see :func:`eligible_communities`; default ``MIN_HUMAN_RESPONSES``).
    Communities below that bar -- including ones with zero human review -- are
    excluded, so an all-automated community can never appear as if it had been
    vetted by a human. ``min_human_responses=1`` keeps every human-evaluated
    community (the 57 behind the paper's community results).
    """
    frame = load_human_evaluation(path)
    eligible = eligible_communities(frame, min_human_responses)
    human = frame.loc[frame["is_scored"] & frame["community"].isin(eligible)].copy()
    human["correct"] = human["is_correct"].astype(float)
    community = (
        human.groupby("community", as_index=False)
        .agg(
            n_scored_responses=("correct", "size"),
            correct_responses=("correct", "sum"),
            accuracy=("correct", "mean"),
        )
        .assign(accuracy=lambda data: data["accuracy"] * 100)
    )
    metadata = add_group_labels(load_corpus_metadata())
    return metadata.merge(community, on="community", how="inner", validate="one_to_one")


def _wilson(correct: float, n: int) -> tuple[float, float]:
    """Return a 95% Wilson interval for a pooled binary estimate."""
    if n <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = correct / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return 100 * max(0.0, center - half_width), 100 * min(1.0, center + half_width)


def _community_bootstrap(correct: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    """Return a 95% percentile interval for pooled accuracy, resampling communities.

    Responses within a community are correlated, so a response-level Wilson
    interval understates the uncertainty of a group estimate. Resampling whole
    communities keeps that between-community spread in the interval.
    """
    if len(n) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(n), size=(BOOTSTRAP_REPLICATES, len(n)))
    pooled = 100 * correct[draws].sum(axis=1) / n[draws].sum(axis=1)
    low, high = np.percentile(pooled, [2.5, 97.5])
    return float(low), float(high)


def summarize_grouping(community: pd.DataFrame, column: str) -> pd.DataFrame:
    """Summarize community means and pooled responses by one grouping.

    ``pooled_ci_*`` is the response-level Wilson interval and
    ``community_ci_*`` the community-bootstrap interval that the figures draw.
    """
    observed = community.dropna(subset=["accuracy"]).copy()
    if column not in observed:
        raise ValueError(f"Unknown grouping column: {column}")
    rows: list[dict[str, object]] = []
    for group, values in observed.groupby(column, sort=False):
        values = values.copy()
        n_communities = len(values)
        community_mean = float(values["accuracy"].mean())
        between_sd = float(values["accuracy"].std(ddof=1)) if n_communities > 1 else 0.0
        pooled_n = int(values["n_scored_responses"].sum())
        pooled_correct = int(values["correct_responses"].sum())
        pooled_accuracy = 100 * pooled_correct / pooled_n if pooled_n else math.nan
        ci_low, ci_high = _wilson(pooled_correct, pooled_n)
        boot_low, boot_high = _community_bootstrap(
            values["correct_responses"].to_numpy(dtype=float),
            values["n_scored_responses"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "group": group,
                "communities": n_communities,
                "scored_responses": pooled_n,
                "mean_community_accuracy": community_mean,
                "median_community_accuracy": float(values["accuracy"].median()),
                "between_community_sd": between_sd,
                "pooled_accuracy": pooled_accuracy,
                "pooled_ci_low": ci_low,
                "pooled_ci_high": ci_high,
                "community_ci_low": boot_low,
                "community_ci_high": boot_high,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_community_accuracy", ignore_index=True)


def _order_resourcedness(summary: pd.DataFrame) -> pd.DataFrame:
    """Return Common Crawl resourcedness rows from lowest to highest."""
    return (
        summary.assign(
            _order=pd.Categorical(
                summary["group"],
                categories=RESOURCEDNESS_ORDER,
                ordered=True,
            )
        )
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def main(path: Path = SCORED_RESPONSES_PATH) -> None:
    """Write accuracy summaries by subregion and by resource level.

    The subregion grouping keeps the ``MIN_HUMAN_RESPONSES`` rule; the resource-level
    grouping pools every human-evaluated community (the 57 reported in the paper).
    """
    for column, filename, minimum in (
        ("un_m49_subregion", "accuracy_by_subregion.csv", MIN_HUMAN_RESPONSES),
        ("language_resourcedness", "accuracy_by_resourcedness.csv", 1),
    ):
        summary = summarize_grouping(community_accuracy(path, minimum), column)
        if column == "language_resourcedness":
            summary = _order_resourcedness(summary)
        write_csv(summary, OUTPUT_DIR / filename)


if __name__ == "__main__":
    main()
