"""Rebuild released survey summaries from approved aggregate source tables.

The aggregate tables under ``data/canonical/riddle_writing_analysis`` are the
release inputs. The topic comparison is recomputed from those difficulty means
and public scored human-review results, rather than copied from a frozen join.

Run from the repository root::

    uv run python -m analysis.riddle_writing_analysis.survey_summaries
"""

from __future__ import annotations

import pandas as pd

from analysis.constants import FIXED_TOPICS
from analysis.model_results.evaluation_data import (
    eligible_communities,
    load_human_evaluation,
)
from analysis.riddle_writing_analysis.survey_data import OUTPUT_DIR, load_aggregates
from analysis.utils import write_csv


def topic_difficulty_human_accuracy(difficulty: pd.DataFrame) -> pd.DataFrame:
    """Join aggregate writing difficulty to public accuracy for fixed topics."""
    difficulty = difficulty.rename(columns={"responses": "writing_responses"})
    human = load_human_evaluation()
    eligible = eligible_communities(human)
    evaluated = human.loc[
        human["is_scored"] & human["community"].isin(eligible) & human["topic"].isin(FIXED_TOPICS)
    ]
    accuracy = (
        evaluated.groupby("topic")["is_correct"]
        .agg(human_accuracy="mean", human_responses="size")
        .reset_index()
    )
    comparison = difficulty.merge(accuracy, on="topic", validate="one_to_one")
    if set(comparison["topic"]) != set(FIXED_TOPICS):
        raise ValueError(f"Expected all fixed topics; found {list(comparison['topic'])}.")
    comparison["spearman_rho"] = (
        comparison["mean_difficulty"].rank().corr(comparison["human_accuracy"].rank())
    )
    comparison["evaluated_communities"] = len(eligible)
    return comparison.sort_values("topic", key=lambda topics: topics.map(FIXED_TOPICS.index))


def main() -> None:
    """Write approved aggregates and recompute the public-results topic join."""
    tables = load_aggregates()
    tables["topic_difficulty_human_accuracy.csv"] = topic_difficulty_human_accuracy(
        tables["topic_difficulty.csv"]
    )
    for name, table in tables.items():
        write_csv(table, OUTPUT_DIR / name)


if __name__ == "__main__":
    main()
