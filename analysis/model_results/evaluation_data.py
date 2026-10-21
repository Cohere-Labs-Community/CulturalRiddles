"""Shared scored-response data, model labels, coverage, and joins."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.constants import MODEL_RESPONSES_PATH, RIDDLES_PATH, SCORED_RESPONSES_PATH

STRICT_SCORE = 3.0
LENIENT_SCORE = 2.0

# Automated scoring methods: the pipeline auto-resolves unambiguous cases
# (exact string match -> correct; empty/refusal/placeholder answers ->
# incorrect) and only routes genuinely ambiguous answers to human review.
# Accuracy must combine both sources -- restricting to ``human_eval`` alone
# keeps only the hardest-to-judge middle slice and drops every response the
# pipeline already resolved automatically in *either* direction, which
# biases every downstream mean (see ``is_scored`` below).
AUTO_METHODS: frozenset[str] = frozenset(
    {"auto_exact_match", "auto_empty", "auto_idk", "auto_placeholder"}
)

# Answers without a valid human label that both zero-shot LLM judges scored
# the same way. The judges replace only the human-review step; disagreements
# and failed verdicts stay unscored. Judge rows count toward accuracy but never
# toward error-type breakdowns, which stay on ``human_eval`` rows.
JUDGE_METHOD = "llm_judge_agree"
JUDGE_SCORE_COLUMNS = ("score_DeepSeek", "score_Gemini")
RIDDLE_TEXT_COLUMNS = (
    "riddle_id",
    "riddle_original",
    "reference_answer_original",
    "reference_answer_english",
)


def load_flat_frame(path: Path = SCORED_RESPONSES_PATH, with_judges: bool = True) -> pd.DataFrame:
    """Load one scored row per reported model answer.

    ``scoring_method`` is the pipeline's routing (``auto_*``, ``human_eval``,
    ``pending_human_eval``, ``invalid_human_eval``), except that answers both
    LLM judges scored the same way become ``llm_judge_agree``.

    Args:
        path: ``responses_scored_consolidated.parquet``.
        with_judges: Apply the judges' agreed scores. ``False`` keeps those
            answers unscored, as human and rule-based scoring alone leave them.

    Returns:
        The rows with riddle text, ``model_name`` (the configuration key),
        ``model_display``, and the ``is_*`` score flags.
    """
    frame = pd.read_parquet(path).rename(columns={"model": "model_name"})
    riddles = pd.read_json(RIDDLES_PATH, lines=True)[list(RIDDLE_TEXT_COLUMNS)]
    frame = frame.merge(riddles, on="riddle_id", how="left")
    frame["score"] = frame["score"].astype("float64")
    judged = frame["scorer"].eq("llm_judges")
    frame["scoring_method"] = frame["routing"]
    if with_judges:
        frame.loc[judged, "scoring_method"] = JUDGE_METHOD
    else:
        frame.loc[judged, "score"] = float("nan")
    frame["is_human"] = frame["scoring_method"].eq("human_eval")
    frame["is_judged"] = frame["scoring_method"].eq(JUDGE_METHOD)
    frame["is_pending"] = frame["scoring_method"].eq("pending_human_eval")
    frame["is_auto"] = frame["scoring_method"].isin(AUTO_METHODS)
    frame["is_scored"] = frame["is_human"] | frame["is_auto"] | frame["is_judged"]
    frame["is_correct"] = frame["score"].eq(STRICT_SCORE)
    frame["is_correct_lenient"] = frame["score"].ge(LENIENT_SCORE)
    return frame


# A community only enters an accuracy comparison once a human has reviewed a
# meaningful sample of it. Combining human + automated scores (``is_scored``
# above) fixes the selection bias of human-only rows, but a community with
# zero or a handful of human_eval rows would then have its reported accuracy
# driven almost entirely by automated exact-match scoring with no human
# oversight at all -- e.g. a community that has never been human-reviewed
# would otherwise appear fully scored. Gate on raw human_eval participation,
# not on ``is_scored`` volume.
MIN_HUMAN_RESPONSES = 600


def eligible_communities(
    frame: pd.DataFrame, min_human_responses: int = MIN_HUMAN_RESPONSES
) -> set[str]:
    """Return communities with at least ``min_human_responses`` human_eval rows.

    Args:
        frame: Output from :func:`load_flat_frame` (or a frame with an
            ``is_human`` column and a ``community`` column).
        min_human_responses: Minimum human-graded row count to qualify.

    Returns:
        The set of qualifying community names.
    """
    human_counts = frame.loc[frame["is_human"]].groupby("community").size()
    return set(human_counts.index[human_counts >= min_human_responses])


def human_evaluated_communities(frame: pd.DataFrame) -> set[str]:
    """Return every community with at least one completed human judgment.

    This preserves the full human-evaluation sample for descriptive figures and
    model summaries, including partially reviewed communities.

    Args:
        frame: Output from :func:`load_flat_frame`.

    Returns:
        Community identifiers represented by a completed human evaluation.
    """
    return set(frame.loc[frame["is_human"], "community"].unique())


def cell_accuracy(frame: pd.DataFrame, communities: set[str] | None = None) -> pd.DataFrame:
    """Return score-3 accuracy per model and community.

    Accuracy is the scored accuracy: human-judged, judge-agreed, and
    automatically resolved rows, with pending rows excluded.

    Args:
        frame: Output from :func:`load_flat_frame`.
        communities: Communities to include. Defaults to those meeting the
            minimum human-response threshold.

    Returns:
        One row per model and community with row counts (``n_exact`` counts
        exact-match rows), ``human_coverage`` (human- or judge-resolved share of
        rows routed past the automatic rules), ``human_share`` (the
        human-reviewed part of those rows), the correct count in ``correct``,
        the denominator in ``total``, and accuracy as a percentage in ``value``.
    """
    selected = eligible_communities(frame) if communities is None else communities
    rows = frame.loc[frame["community"].isin(selected)].drop_duplicates(
        ["community", "riddle_id", "model_display"]
    )
    rows = rows.assign(
        scored_correct=rows["is_scored"] & rows["is_correct"],
        exact=rows["scoring_method"].eq("auto_exact_match"),
    )
    cells = rows.groupby(["model_display", "community"], as_index=False).agg(
        total=("is_scored", "sum"),
        correct=("scored_correct", "sum"),
        n_human=("is_human", "sum"),
        n_judged=("is_judged", "sum"),
        n_pending=("is_pending", "sum"),
        n_exact=("exact", "sum"),
    )
    resolved = cells["n_human"] + cells["n_judged"]
    routed = resolved + cells["n_pending"]
    cells["human_coverage"] = (resolved / routed.where(routed > 0)).fillna(1.0)
    cells["human_share"] = (cells["n_human"] / routed.where(routed > 0)).fillna(1.0)
    cells["value"] = cells["correct"] / cells["total"] * 100
    return cells


def load_human_evaluation(
    path: Path = SCORED_RESPONSES_PATH,
    responses_path: Path = MODEL_RESPONSES_PATH,
) -> pd.DataFrame:
    """Load scored rows with the exact-match result recorded for each answer.

    Args:
        path: ``responses_scored_consolidated.parquet``.
        responses_path: ``model_responses.parquet``, whose
            ``exact_match_verdict`` is the pipeline's string-match result.

    Returns:
        :func:`load_flat_frame` rows with ``is_exact_match``.
    """
    frame = load_flat_frame(path)
    exact = pd.read_parquet(
        responses_path, columns=["riddle_id", "model", "exact_match_verdict"]
    ).rename(columns={"model": "model_name"})
    frame = frame.merge(exact, on=["riddle_id", "model_name"], how="left")
    verdict = frame.pop("exact_match_verdict")
    frame["is_exact_match"] = verdict.eq("correct").where(
        verdict.notna(), frame["scoring_method"].eq("auto_exact_match")
    )
    return frame
