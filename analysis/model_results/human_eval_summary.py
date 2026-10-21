"""Summarize model performance across every human-evaluated community.

Score-3 accuracy combines valid human judgments with automatically resolved
responses. The descriptive summary retains all communities with at least one
completed human judgment; per-cell coverage remains available to identify
partially reviewed cells.
Run from the repository root with::

    uv run python -m analysis.model_results.human_eval_summary
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.constants import MODEL_RESULTS_OUTPUT_DIR, SCORED_RESPONSES_PATH
from analysis.corpus import load_corpus_metadata
from analysis.model_names import model_display_map, model_parameter_metadata
from analysis.model_results.evaluation_data import (
    cell_accuracy,
    human_evaluated_communities,
    load_flat_frame,
)
from analysis.utils import write_csv

OUTPUT_CSV = MODEL_RESULTS_OUTPUT_DIR / "human_evaluated_model_summary.csv"

COMMUNITY_BUCKET_COLUMNS: tuple[str, ...] = (
    "communities_eq_0",
    "communities_0_25",
    "communities_25_50",
    "communities_50_75",
    "communities_75_100",
)


def _accuracy_buckets(per_community: pd.DataFrame) -> pd.DataFrame:
    """Count each model's communities in the fixed accuracy intervals."""
    accuracy = per_community["human_correct"]
    buckets = pd.Series(pd.NA, index=per_community.index, dtype="string")
    buckets.loc[accuracy.eq(0)] = COMMUNITY_BUCKET_COLUMNS[0]
    buckets.loc[accuracy.gt(0) & accuracy.lt(25)] = COMMUNITY_BUCKET_COLUMNS[1]
    buckets.loc[accuracy.ge(25) & accuracy.lt(50)] = COMMUNITY_BUCKET_COLUMNS[2]
    buckets.loc[accuracy.ge(50) & accuracy.lt(75)] = COMMUNITY_BUCKET_COLUMNS[3]
    buckets.loc[accuracy.ge(75) & accuracy.le(100)] = COMMUNITY_BUCKET_COLUMNS[4]
    if buckets.isna().any():
        invalid = per_community.loc[buckets.isna(), "human_correct"].tolist()
        raise ValueError(f"Human accuracy falls outside [0, 100]: {invalid}")

    counts = pd.crosstab(per_community["model"], buckets).reindex(
        columns=list(COMMUNITY_BUCKET_COLUMNS), fill_value=0
    )
    counts.index.name = "model"
    return counts.reset_index()


def build_summary(
    flat_csv: Path = SCORED_RESPONSES_PATH, all_communities: bool = False
) -> pd.DataFrame:
    """Build the matched score-3-versus-exact summary.

    Args:
        flat_csv: ``responses_scored_consolidated.parquet``.
        all_communities: Include the benchmarked communities without human
            labels, scored by rules and the LLM judges alone.

    Returns:
        One row per evaluated model. Accuracy columns are percentages on a
        0--100 scale, and community bucket columns are integer counts.
        Exact-match accuracy shares the score-3 denominator.

    Raises:
        ValueError: If communities are missing from the corpus metadata or a
            model does not cover every evaluated community.
    """
    frame = load_flat_frame(flat_csv)

    canonical_communities = set(load_corpus_metadata()["community"])
    unknown_communities = set(frame["community"]) - canonical_communities
    if unknown_communities:
        raise ValueError(f"Communities missing from corpus_metadata.csv: {unknown_communities}")

    selected = set(frame["community"]) if all_communities else human_evaluated_communities(frame)
    cells = cell_accuracy(frame, selected).rename(columns={"model_display": "model"})
    if cells.empty:
        raise ValueError("No scored cells are available")
    communities = set(cells["community"])

    community_coverage = cells.groupby("model")["community"].nunique()
    incomplete_models = community_coverage[community_coverage.ne(len(communities))]
    if not incomplete_models.empty:
        raise ValueError(
            f"Models do not cover every evaluated community: {incomplete_models.to_dict()}"
        )

    per_model = cells.groupby("model", as_index=False).agg(
        correct=("correct", "sum"),
        exact=("n_exact", "sum"),
        n=("total", "sum"),
    )
    per_model["human_correct"] = per_model["correct"] / per_model["n"] * 100
    per_model["exact_match"] = per_model["exact"] / per_model["n"] * 100
    per_model["n"] = per_model["n"].round().astype(int)

    per_community = cells[["model", "community", "value"]].rename(
        columns={"value": "human_correct"}
    )
    best = (
        per_community.sort_values(
            ["model", "human_correct", "community"], ascending=[True, False, True]
        )
        .drop_duplicates("model")
        .rename(
            columns={
                "community": "community_with_highest_accuracy",
                "human_correct": "highest_community_human_correct",
            }
        )
    )
    worst = (
        per_community.sort_values(["model", "human_correct", "community"])
        .drop_duplicates("model")
        .rename(
            columns={
                "community": "community_with_lowest_accuracy",
                "human_correct": "lowest_community_human_correct",
            }
        )
    )

    summary = (
        per_model.merge(
            best[
                [
                    "model",
                    "community_with_highest_accuracy",
                    "highest_community_human_correct",
                ]
            ],
            on="model",
            validate="one_to_one",
        )
        .merge(
            worst[
                [
                    "model",
                    "community_with_lowest_accuracy",
                    "lowest_community_human_correct",
                ]
            ],
            on="model",
            validate="one_to_one",
        )
        .merge(_accuracy_buckets(per_community), on="model", validate="one_to_one")
    )
    search_display = model_display_map()["GPT5.5-search"]
    settings = summary["model"].map({search_display: "search enabled"})
    summary.insert(1, "setting", settings.fillna("closed book"))
    access = model_parameter_metadata().set_index("model_display")["access"]
    summary.insert(2, "access", summary["model"].map(access))
    if summary["access"].isna().any():
        missing = sorted(summary.loc[summary["access"].isna(), "model"])
        raise ValueError(f"Models missing access metadata: {missing}")
    summary.insert(3, "is_proprietary", summary["access"].eq("proprietary"))
    summary["n_evaluated_communities"] = len(communities)

    percentage_columns = [
        "exact_match",
        "human_correct",
        "highest_community_human_correct",
        "lowest_community_human_correct",
    ]
    # Rank on unrounded accuracy, so models tied after rounding keep their order.
    summary = summary.sort_values(
        ["human_correct", "model"], ascending=[False, True], ignore_index=True
    )
    summary[percentage_columns] = summary[percentage_columns].round(1)
    summary["difference_pp"] = (summary["human_correct"] - summary["exact_match"]).round(1)
    summary[list(COMMUNITY_BUCKET_COLUMNS)] = summary[list(COMMUNITY_BUCKET_COLUMNS)].astype(int)
    return summary[
        [
            "model",
            "setting",
            "access",
            "is_proprietary",
            "exact_match",
            "human_correct",
            "difference_pp",
            "highest_community_human_correct",
            "community_with_highest_accuracy",
            "lowest_community_human_correct",
            "community_with_lowest_accuracy",
            *COMMUNITY_BUCKET_COLUMNS,
            "n",
            "n_evaluated_communities",
        ]
    ].rename(
        columns={
            "highest_community_human_correct": "max_community_accuracy",
            "community_with_highest_accuracy": "max_accuracy_community",
            "lowest_community_human_correct": "min_community_accuracy",
            "community_with_lowest_accuracy": "min_accuracy_community",
        }
    )


def main() -> None:
    """Write the model summary."""
    summary = build_summary()
    write_csv(summary, OUTPUT_CSV)
    print(f"Wrote {OUTPUT_CSV} ({len(summary)} models).")


if __name__ == "__main__":
    main()
