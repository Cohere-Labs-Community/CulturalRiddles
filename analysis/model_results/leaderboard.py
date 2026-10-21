"""Model-level accuracy tables: exact match versus review, and the judge fill."""

import pandas as pd

from analysis.constants import MODEL_RESULTS_OUTPUT_DIR
from analysis.model_results.evaluation_data import (
    human_evaluated_communities,
    load_flat_frame,
)
from analysis.utils import write_csv

OUTPUT_DIR = MODEL_RESULTS_OUTPUT_DIR


def load_human_eval_leaderboard() -> pd.DataFrame:
    """Compute exact-match and score-3 accuracy on the paper-table communities.

    Exact match is the pipeline's strict ``auto_exact_match`` resolution, and
    the community set matches the model-performance table, so the figure and
    the table report the same Exact and Acc values.
    """
    frame = load_flat_frame()
    frame["is_exact_match"] = frame["scoring_method"].eq("auto_exact_match")
    scored = frame[frame["community"].isin(human_evaluated_communities(frame)) & frame["is_scored"]]
    summary = (
        scored.groupby("model_display")
        .agg(
            exact_match_accuracy=("is_exact_match", "mean"),
            human_accuracy=("is_correct", "mean"),
            n=("is_correct", "size"),
            human_scored_percent=("is_human", "mean"),
        )
        .assign(
            exact_match_accuracy=lambda data: data["exact_match_accuracy"] * 100,
            human_accuracy=lambda data: data["human_accuracy"] * 100,
            human_scored_percent=lambda data: data["human_scored_percent"] * 100,
        )
        .sort_values("exact_match_accuracy", ascending=False)
        .reset_index()
    )
    return summary


def judge_fill_effect() -> pd.DataFrame:
    """Compare each model's accuracy with and without judge-scored pending answers.

    Returns:
        One row per model with accuracy and pending share under human and
        automatic scores only (``*_before``) and after adding responses on which
        both LLM judges agree (``*_after``), over the paper-table communities.
    """
    columns = {}
    for suffix, with_judges in (("before", False), ("after", True)):
        frame = load_flat_frame(with_judges=with_judges)
        frame = frame[frame["community"].isin(human_evaluated_communities(frame))]
        grouped = frame.groupby("model_display")
        scored = frame[frame["is_scored"]].groupby("model_display")
        columns[f"accuracy_{suffix}"] = scored["is_correct"].mean() * 100
        columns[f"pending_{suffix}"] = grouped["is_pending"].sum()
    effect = pd.DataFrame(columns)
    effect["change_pp"] = effect["accuracy_after"] - effect["accuracy_before"]
    return effect.sort_values("accuracy_after", ascending=False).reset_index()


def main() -> None:
    """Write ``judge_fill_effect.csv``."""
    write_csv(judge_fill_effect(), OUTPUT_DIR / "judge_fill_effect.csv")


if __name__ == "__main__":
    main()
