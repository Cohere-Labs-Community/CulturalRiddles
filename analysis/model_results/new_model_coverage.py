"""Estimate how much of a new model's output existing labels already cover.

Each current model is treated as newly benchmarked: a response counts as
covered when rule-based scoring resolves it, or when another model gave the
same normalized answer to the same riddle and that answer has a human label.
"""

import pandas as pd

from analysis.constants import MODEL_RESULTS_OUTPUT_DIR
from analysis.model_results.evaluation_data import (
    load_flat_frame,
)
from analysis.utils import write_csv

OUTPUT_DIR = MODEL_RESULTS_OUTPUT_DIR


def label_coverage_rows() -> pd.DataFrame:
    """Flag each response as covered by rule-based scoring or by another model's label.

    Returns:
        One row per response with ``model_display``, ``community``,
        ``is_auto``, ``reused`` (human label from a different model), and
        ``covered`` (``is_auto`` or ``reused``).
    """
    frame = load_flat_frame(with_judges=False)[
        [
            "model_display",
            "community",
            "riddle_id",
            "model_answer",
            "is_auto",
            "is_human",
        ]
    ]
    frame["answer_norm"] = frame["model_answer"].fillna("").astype(str).str.strip().str.lower()
    keys = ["riddle_id", "answer_norm"]
    labelers = (
        frame.loc[frame["is_human"]]
        .groupby(keys)["model_display"]
        .agg(n_labelers="nunique", only_labeler="first")
        .reset_index()
    )
    frame = frame.merge(labelers, on=keys, how="left")
    frame["reused"] = frame["n_labelers"].fillna(0).ge(2) | (
        frame["n_labelers"].eq(1) & frame["only_labeler"].ne(frame["model_display"])
    )
    frame["covered"] = frame["is_auto"] | frame["reused"]
    return frame.drop(columns=["n_labelers", "only_labeler"])


def summarize_coverage(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate leave-one-model-out coverage by model.

    Args:
        rows: Output of :func:`label_coverage_rows`.

    Returns:
        Per-model coverage (percent covered, percent of rule-unresolved
        responses resolved by reuse, responses left for judges), sorted by
        coverage.
    """
    routed = rows.loc[~rows["is_auto"]]
    by_model = pd.DataFrame(
        {
            "covered_percent": rows.groupby("model_display")["covered"].mean() * 100,
            "reused_of_routed_percent": routed.groupby("model_display")["reused"].mean() * 100,
            "uncovered_responses": (~rows["covered"]).groupby(rows["model_display"]).sum(),
        }
    ).sort_values("covered_percent", ascending=False)
    return by_model


def main() -> None:
    """Write the per-model new-model coverage table."""
    by_model = summarize_coverage(label_coverage_rows())
    write_csv(by_model, OUTPUT_DIR / "new_model_coverage_by_model.csv", index=True)


if __name__ == "__main__":
    main()
