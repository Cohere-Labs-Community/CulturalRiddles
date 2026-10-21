"""Pool score-3 accuracy for every represented country."""

from __future__ import annotations

import pandas as pd

from analysis.constants import EVAL_ANALYSIS_OUTPUT_DIR
from analysis.corpus import load_corpus_metadata
from analysis.model_results.evaluation_data import (
    eligible_communities,
    load_human_evaluation,
)


def country_accuracy() -> pd.DataFrame:
    """Pool scored (human + automated) response accuracy by country.

    Includes both human-judged rows and automatically-resolved rows (exact
    string match, empty/refusal/placeholder answers); excludes pending and
    invalid human-review rows. Restricting to human-only rows would keep
    just the ambiguous middle slice and drop every response the pipeline
    already resolved automatically in either direction. A community must
    still have at least ``MIN_HUMAN_RESPONSES`` human_eval rows to
    contribute to its country's pool -- otherwise a country whose only
    community has little or no human review could appear on equal footing
    with a fully human-vetted one.
    """
    metadata = load_corpus_metadata()
    frame = load_human_evaluation()
    eligible = eligible_communities(frame)
    human = frame[frame["is_scored"] & frame["community"].isin(eligible)].copy()
    human["is_correct"] = human["is_correct"].astype(float)
    by_country = (
        human.merge(metadata[["community", "country"]], on="community", validate="many_to_one")
        .groupby("country", as_index=False)
        .agg(
            human_correct=("is_correct", "mean"),
            human_scored_responses=("is_correct", "size"),
            communities_with_human_scores=("community", "nunique"),
        )
        .assign(human_correct=lambda data: data["human_correct"] * 100)
    )
    countries = metadata.groupby("country", as_index=False).agg(
        corpus_communities=("community", "nunique")
    )
    return countries.merge(by_country, on="country", how="left", validate="one_to_one")


def main() -> None:
    """Write pooled accuracy by country."""
    EVAL_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    country_accuracy().to_csv(EVAL_ANALYSIS_OUTPUT_DIR / "accuracy_by_country.csv", index=False)


if __name__ == "__main__":
    main()
