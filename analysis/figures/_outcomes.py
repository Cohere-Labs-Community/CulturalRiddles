"""Detailed outcome taxonomy of rule- and human-scored answers."""

from __future__ import annotations

import pandas as pd

from analysis.model_results.evaluation_data import AUTO_METHODS

OUTCOME_ORDER: tuple[str, ...] = (
    "Exact match",
    "Different valid answer",
    "Correct, in English",
    "Other correct",
    "Closely related culture",
    "Wrong level of detail",
    "Surface pattern matching",
    "Other partial",
    "Hallucinated",
    "Right category, wrong entity",
    "Too broad",
    "Other incorrect",
    "Declined to answer",
    "Placeholder",
    "Empty output",
)


def _annotation_text(frame: pd.DataFrame) -> pd.Series:
    """Combine optional human annotation fields into normalized text."""
    columns = [
        column
        for column in ("error_type_label", "error_type", "outcome_label", "label")
        if column in frame
    ]
    if not columns:
        return pd.Series("", index=frame.index, dtype="string")
    return frame[columns].fillna("").astype(str).agg(" ".join, axis=1).str.lower()


def _response_text(frame: pd.DataFrame) -> pd.Series:
    """Combine optional answer fields used to identify abstentions and blanks."""
    columns = [
        column
        for column in ("model_answer", "answer_text", "answer", "response")
        if column in frame
    ]
    if not columns:
        return pd.Series("", index=frame.index, dtype="string")
    return frame[columns].fillna("").astype(str).agg(" ".join, axis=1).str.lower()


def classify_outcomes(frame: pd.DataFrame) -> pd.Series:
    """Classify human-evaluated responses into the detailed error taxonomy."""
    if "score" not in frame:
        raise ValueError("Human-evaluation input must contain a 'score' column.")
    score = pd.to_numeric(frame["score"], errors="coerce")
    labels = _annotation_text(frame)
    answers = _response_text(frame)
    methods = frame.get("scoring_method", pd.Series("", index=frame.index, dtype="string")).fillna(
        ""
    )
    outcome = pd.Series("Other incorrect", index=frame.index, dtype="string")

    correct = score.eq(3)
    # A top score without an alternate-answer code is the deterministic exact
    # match bucket.  Human annotation only adds the non-exact 3.1--3.3 detail.
    outcome.loc[correct] = "Exact match"
    outcome.loc[correct & labels.str.contains("english", regex=False)] = "Correct, in English"
    outcome.loc[
        correct & labels.str.contains("different|valid|alternative|synonym|translation", regex=True)
    ] = "Different valid answer"
    outcome.loc[correct & labels.str.contains(r"3\.3|other correct", regex=True)] = "Other correct"

    partial = score.eq(2)
    outcome.loc[partial] = "Other partial"
    outcome.loc[partial & labels.str.contains("related.*culture|culture.*related", regex=True)] = (
        "Closely related culture"
    )
    outcome.loc[partial & labels.str.contains("level.*detail|detail", regex=True)] = (
        "Wrong level of detail"
    )
    outcome.loc[partial & labels.str.contains("surface|pattern", regex=True)] = (
        "Surface pattern matching"
    )

    incorrect = score.eq(0)
    outcome.loc[incorrect & labels.str.contains("hallucin", regex=True)] = "Hallucinated"
    outcome.loc[
        incorrect & labels.str.contains("wrong entity|right category|category", regex=True)
    ] = "Right category, wrong entity"
    outcome.loc[incorrect & labels.str.contains("too broad|broad", regex=True)] = "Too broad"

    uncertain = score.eq(1)
    outcome.loc[uncertain] = "Declined to answer"
    placeholder = labels.str.contains("placeholder", regex=False) | answers.str.contains(
        "placeholder", regex=False
    )
    answer_columns = {"answer_text", "answer", "response"} & set(frame.columns)
    empty = labels.str.contains("empty", regex=False)
    if answer_columns:
        empty |= answers.str.strip().eq("")
    outcome.loc[uncertain & placeholder] = "Placeholder"
    outcome.loc[uncertain & empty & ~placeholder] = "Empty output"

    # The flat-file automatic non-response rules are authoritative.  They
    # arrive as score-0 rows, so applying them only inside score-1 uncertainty
    # would silently misclassify them as ordinary incorrect answers.
    outcome.loc[methods.eq("auto_placeholder")] = "Placeholder"
    outcome.loc[methods.eq("auto_empty")] = "Empty output"
    return outcome


def outcome_shares(frame: pd.DataFrame) -> pd.DataFrame:
    """Return 100%-stacked detailed outcome shares for each model."""
    scored = frame.copy()
    if "scoring_method" in scored:
        valid_method = scored["scoring_method"].eq("human_eval") | scored["scoring_method"].isin(
            AUTO_METHODS
        )
        scored = scored[valid_method].copy()
    scored = scored[scored["score"].notna()].copy()
    scored["outcome"] = classify_outcomes(scored)
    shares = (
        scored.groupby("model_display")["outcome"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .reindex(columns=OUTCOME_ORDER, fill_value=0.0)
    )
    return shares.loc[shares["Exact match"].sort_values(ascending=False).index]
