"""English-translation condition: validate, judge input, and accuracy comparison.

Five low-resource communities were rerun with the English translation of each
riddle instead of the original. This module validates that export against the
canonical corpus, writes the judge input for the zero-shot judges,
and compares two-judge accuracy with the original-language accuracy of the
same five model configurations.

Run offline summaries from the repository root::

    uv run python -m analysis.llm_judge.english_condition summarize

The summary reads recorded verdicts in ``data/canonical/english_condition``.
For a separate paid rerun, prepare inputs, then run each provider::

    uv run python -m analysis.llm_judge.english_condition prepare
    for model in gemini-3.5-flash deepseek-v4.1-flash; do
      uv run riddles-judge \\
        data/derived/llm_judge/english_condition/judge_input.csv \\
        --model "$model" --retries 4 --workers 16 \\
        --out "data/runs/llm_judge/english.zero.$model.jsonl"
    done

Paid traces do not replace the canonical verdicts used by ``summarize``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import orjson
import polars as pl

from analysis.constants import DERIVED_DIR, ENGLISH_CONDITION_DIR, RIDDLES_PATH
from analysis.llm_judge.judge_data import JUDGES as JUDGE_MODELS
from analysis.llm_judge.judge_data import load_verdicts
from analysis.llm_judge.rater_agreement import cohen_kappa, confusion_counts

logger = logging.getLogger(__name__)

ENGLISH_EXPORT = ENGLISH_CONDITION_DIR / "responses.parquet"
STUDY_DIR = DERIVED_DIR / "llm_judge" / "english_condition"
JUDGE_INPUT = STUDY_DIR / "judge_input.csv"
VALIDATION_JSON = STUDY_DIR / "validation.json"
OUTPUT_DIR = STUDY_DIR

COMMUNITY_IDS: dict[str, str] = {
    "Yoruba (Nigeria)": "yoruba_nigeria",
    "Sudanese Arabic (Sudan)": "arabic_sudan",
    "Lumasaba (Uganda)": "lumasaba_uganda",
    "Buginese (Indonesia)": "bugisnese_indonesia",
    "Mongondow (Indonesia)": "mongondow_indonesia",
}
# Canonical configurations observed in the English-translation condition.
ENGLISH_MODELS: tuple[str, ...] = (
    "GPT5.5-none",
    "claude-opus-4-7",
    "deepseek-v4-pro",
    "gemini-3.1-pro-preview",
    "Qwen-SEA-LION-v4.5-27B-IT",
)
PROVIDER_ERROR = r"^Provider (returned|did not complete)"
PLACEHOLDER_ANSWER = "<your_riddle_answer>"
IDK_ANSWER = "i don't know"


def normalized(column: str) -> pl.Expr:
    """Strip and lowercase a text column, as the rule-based scoring does."""
    return pl.col(column).fill_null("").str.strip_chars().str.to_lowercase()


def rule_based(answer: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """Return the rule-based method and score for an English answer.

    The original condition scores an exact, case-insensitive match with the
    original-language reference as 3, an empty or placeholder answer as 0, and
    an exact "I don't know" as 1. English answers may also match the English
    reference exactly; both matches score 3 and are kept apart by method.

    Args:
        answer: Normalized answer expression.

    Returns:
        Method (null when a judge must score the answer) and score expressions.
    """
    original, english = (
        normalized("reference_answer_original"),
        normalized("reference_answer_english"),
    )
    method = (
        pl.when(answer.is_in(["", PLACEHOLDER_ANSWER]))
        .then(pl.lit("auto_empty"))
        .when(answer == IDK_ANSWER)
        .then(pl.lit("auto_idk"))
        .when((original != "") & (answer == original))
        .then(pl.lit("auto_exact_match_original"))
        .when((english != "") & (answer == english))
        .then(pl.lit("auto_exact_match_english"))
    )
    score = (
        pl.when(method == "auto_empty")
        .then(0)
        .when(method == "auto_idk")
        .then(1)
        .when(method.is_not_null())
        .then(3)
    )
    return method, score


def load_export(path: Path = ENGLISH_EXPORT) -> pl.DataFrame:
    """Load the English-condition export with parsed answers and corpus fields.

    The answer is the first paragraph of ``model_response``; the rest is the
    model's explanation, which the original-language judge input also omits.

    Args:
        path: ``data/canonical/english_condition/responses.parquet``.

    Returns:
        One row per riddle and model, joined to the canonical corpus.

    Raises:
        ValueError: If a riddle id is missing from the corpus.
    """
    export = pl.read_parquet(path)
    corpus = pl.read_ndjson(RIDDLES_PATH, infer_schema_length=None).select(
        "riddle_id",
        pl.col("community").alias("community_id"),
        "topic",
        "riddle_original",
        "riddle_english",
        "reference_answer_original",
        "reference_answer_english",
    )
    frame = export.join(corpus, on="riddle_id", how="left").with_columns(
        pl.col("model_response").str.contains(PROVIDER_ERROR).alias("provider_error"),
        pl.col("model_response")
        .str.split("\n\n")
        .list.first()
        .str.strip_chars()
        .alias("model_answer"),
    )
    missing = frame.filter(pl.col("community_id").is_null())["riddle_id"].to_list()
    if missing:
        raise ValueError(f"Riddles missing from the corpus: {missing[:5]}")
    method, score = rule_based(normalized("model_answer"))
    return frame.with_columns(
        pl.when(~pl.col("provider_error")).then(method).alias("rule_method"),
        pl.when(~pl.col("provider_error")).then(score).alias("rule_score"),
    )


def answer_in_riddle(frame: pl.DataFrame) -> pl.DataFrame:
    """Return riddles whose English translation contains the English answer.

    Args:
        frame: Output of :func:`load_export`.

    Returns:
        One row per riddle with ``riddle_id``, ``riddle_english``, and
        ``reference_answer_english``; answers of three characters or fewer are
        skipped because they match inside unrelated words.
    """
    reference = pl.col("reference_answer_english").str.to_lowercase()
    return (
        frame.unique("riddle_id")
        .filter(
            reference.str.len_chars() > 3,
            pl.col("riddle_english").str.to_lowercase().str.contains(reference, literal=True),
        )
        .select("riddle_id", "riddle_english", "reference_answer_english")
    )


def validate(frame: pl.DataFrame) -> dict[str, object]:
    """Collect structural and content checks on the English export.

    Args:
        frame: Output of :func:`load_export`.

    Returns:
        Named check results, JSON-serializable.
    """
    corpus = pl.read_ndjson(RIDDLES_PATH, infer_schema_length=None)
    in_scope = corpus.filter(pl.col("community").is_in(list(COMMUNITY_IDS.values())))
    answer = pl.col("model_answer").str.to_lowercase()
    riddles = frame.unique("riddle_id")
    return {
        "rows": frame.height,
        "riddles": riddles.height,
        "duplicate_riddle_model": frame.select("riddle_id", "model").is_duplicated().sum(),
        "community_label_mismatch": frame.filter(
            pl.col("community").replace_strict(COMMUNITY_IDS, default=None)
            != pl.col("community_id")
        ).height,
        "corpus_riddles_absent": in_scope.filter(
            ~pl.col("riddle_id").is_in(frame["riddle_id"].implode())
        ).height,
        "reference_english_mismatch": frame.filter(
            pl.col("reference_english").str.strip_chars()
            != pl.col("reference_answer_english").str.strip_chars()
        ).height,
        "provider_errors": frame.filter("provider_error")
        .group_by("model", "community", "model_response")
        .len()
        .sort("len", descending=True)
        .to_dicts(),
        "idk_answers": frame.filter(answer.str.contains("i don't know"))
        .group_by("model")
        .len()
        .sort("model")
        .to_dicts(),
        "reference_en_equals_original": riddles.filter(
            pl.col("reference_answer_english").str.strip_chars()
            == pl.col("reference_answer_original").str.strip_chars()
        )
        .group_by("community")
        .len()
        .to_dicts(),
        "english_riddle_contains_reference": answer_in_riddle(frame).to_dicts(),
        "long_answers_over_100_chars": frame.filter(
            pl.col("model_answer").str.len_chars() > 100
        ).height,
    }


def prepare() -> None:
    """Validate the export and write the judge input.

    Rule-based rows (exact matches, empty answers, "I don't know") and
    provider failures are not sent to the judges.
    """
    frame = load_export()
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_JSON.write_bytes(orjson.dumps(validate(frame), option=orjson.OPT_INDENT_2))
    # The judge sees the riddle text the answering model saw, as in the
    # original condition; both reference fields are unchanged.
    judge_input = frame.filter(~pl.col("provider_error"), pl.col("rule_method").is_null()).select(
        "riddle_id",
        pl.col("community_id").alias("community"),
        "topic",
        pl.col("riddle_english").alias("riddle_original"),
        pl.col("model").alias("model_name"),
        "model_answer",
        "reference_answer_original",
        "reference_answer_english",
    )
    judge_input.write_csv(JUDGE_INPUT)
    logger.info("Wrote %d judge rows to %s", judge_input.height, JUDGE_INPUT)


def judge_scores() -> pl.DataFrame:
    """Read both judges' final verdicts as ``score_<judge>`` columns.

    Returns:
        One row per riddle and English-run model with each judge's 0--3 score;
        a judge without a valid verdict leaves its column null.
    """
    frames = [
        load_verdicts("english", judge).select(
            "riddle_id", "model", pl.col("score").alias(f"score_{judge.lower()}")
        )
        for judge in JUDGE_MODELS
    ]
    return frames[0].join(frames[1], on=["riddle_id", "model"], how="full", coalesce=True)


def english_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Score English-condition rows with rule-based scoring, then two judges.

    Rule-based rows keep their rule score; judge verdicts on them are ignored.
    For the rest, accuracy is binary, so a row is scored when both judges agree on whether it
    is correct (score 3). The original pending-answer rule also requires the
    same 0--3 score; ``is_scored_same_score`` keeps it as a sensitivity. It
    differs mainly on answers one judge scores 0 and the other 2, which are
    incorrect under both verdicts.

    Args:
        frame: Output of :func:`load_export`.

    Returns:
        Rows with ``status`` (a rule-based method, ``agree``, ``disagree``,
        ``missing``, or ``provider_error``), ``is_scored``,
        ``is_scored_same_score``, and ``is_correct`` (rule score 3, or both
        judges give score 3).
    """
    scores = frame.join(judge_scores(), on=["riddle_id", "model"], how="left")
    gemini, deepseek = pl.col("score_gemini"), pl.col("score_deepseek")
    rule = pl.col("rule_method").is_not_null()
    status = (
        pl.when("provider_error")
        .then(pl.lit("provider_error"))
        .when(rule)
        .then(pl.col("rule_method"))
        .when(gemini.is_null() | deepseek.is_null())
        .then(pl.lit("missing"))
        .when((gemini == 3) == (deepseek == 3))
        .then(pl.lit("agree"))
        .otherwise(pl.lit("disagree"))
    )
    return scores.with_columns(status.alias("status")).select(
        "community_id",
        "riddle_id",
        pl.col("model").alias("model_name"),
        "status",
        "score_gemini",
        "score_deepseek",
        (rule | (pl.col("status") == "agree")).alias("is_scored"),
        (rule | ((pl.col("status") == "agree") & (gemini == deepseek))).alias(
            "is_scored_same_score"
        ),
        pl.when(rule)
        .then(pl.col("rule_score") == 3)
        .otherwise((pl.col("status") == "agree") & (gemini == 3))
        .alias("is_correct"),
        pl.col("status").str.starts_with("auto_exact_match").alias("is_exact"),
    )


def original_rows() -> pl.DataFrame:
    """Return accuracy rows for the same riddles and model configurations.

    Returns:
        One row per riddle and model with ``is_scored``, ``is_correct``, and
        ``is_empty`` (empty responses, which count as incorrect), and
        ``is_exact`` (rule-based exact match).
    """
    from analysis.model_results.evaluation_data import load_flat_frame

    flat = load_flat_frame()
    flat = flat[
        flat["community"].isin(COMMUNITY_IDS.values()) & flat["model_name"].isin(ENGLISH_MODELS)
    ].drop_duplicates(["community", "riddle_id", "model_name"])
    return pl.from_pandas(
        flat[["community", "riddle_id", "model_name", "scoring_method"]]
        .assign(is_scored=flat["is_scored"], is_correct=flat["is_correct"])
        .rename(columns={"community": "community_id"})
    ).with_columns(
        (pl.col("scoring_method") == "auto_empty").alias("is_empty"),
        (pl.col("scoring_method") == "auto_exact_match").alias("is_exact"),
    )


def high_resource_accuracy() -> float:
    """Pooled accuracy (%) of the five English-condition models on high-resource communities.

    The reference for how far translation closes the gap: the same models'
    scored answers in every human-evaluated high-resource community.
    """
    from analysis.corpus import load_corpus_metadata
    from analysis.model_results.evaluation_data import (
        human_evaluated_communities,
        load_flat_frame,
    )

    flat = load_flat_frame()
    metadata = load_corpus_metadata()
    high = set(metadata.loc[metadata["language_resourcedness"].eq("high"), "community"])
    rows = flat[
        flat["community"].isin(high & human_evaluated_communities(flat))
        & flat["model_name"].isin(ENGLISH_MODELS)
        & flat["is_scored"]
    ]
    return float(rows["is_correct"].mean() * 100)


def bootstrap_delta(
    paired: pl.DataFrame, reps: int = 2000, seed: int = 20260930
) -> tuple[float, float]:
    """Riddle-cluster bootstrap 95% interval of English minus original accuracy.

    Args:
        paired: Rows scored in both conditions, with ``riddle_id``,
            ``correct_original``, and ``correct_english``.
        reps: Bootstrap replicates.
        seed: Random seed.

    Returns:
        Lower and upper percentage-point bounds.
    """
    riddles = paired.group_by("riddle_id", maintain_order=True).agg(
        pl.len().alias("n"),
        pl.col("correct_original").sum().alias("original"),
        pl.col("correct_english").sum().alias("english"),
    )
    counts = riddles.select("n", "original", "english").to_numpy().astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(counts), size=(reps, len(counts)))
    totals = counts[draws].sum(1)
    deltas = (totals[:, 2] - totals[:, 1]) / totals[:, 0] * 100
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


def binary_kappa(first: pl.Series, second: pl.Series) -> float:
    """Cohen's kappa of two raters on correct (score 3) versus not."""
    left = (first == 3).cast(pl.Int64).to_numpy()
    right = (second == 3).cast(pl.Int64).to_numpy()
    totals = confusion_counts(left, right, 2).sum(0)
    return float(cohen_kappa(totals, 1 - np.eye(2)))


def percent(flags: pl.Series) -> float:
    """Return the share of true values in a boolean series, in percent."""
    return float(flags.to_numpy().mean()) * 100


def community_summary(english: pl.DataFrame, original: pl.DataFrame) -> pl.DataFrame:
    """Compare pooled five-model accuracy by community in both conditions.

    Args:
        english: Output of :func:`english_rows`.
        original: Output of :func:`original_rows`.

    Returns:
        One row per community plus an ``all`` row with both accuracies, the
        English bounds when judge disagreements count as all wrong or all
        right, the paired difference with its interval, and judge agreement.
    """
    keys = ["community_id", "riddle_id", "model_name"]
    paired = (
        english.filter("is_scored")
        .join(original.filter("is_scored"), on=keys, suffix="_original")
        .rename({"is_correct": "correct_english", "is_correct_original": "correct_original"})
    )
    groups = [(name, pl.col("community_id") == name) for name in COMMUNITY_IDS.values()]
    rows = []
    for name, keep in [*groups, ("all", pl.lit(True))]:
        eng, orig, pair = (
            english.filter(keep),
            original.filter(keep),
            paired.filter(keep),
        )
        answered = eng.filter(pl.col("status") != "provider_error")
        judged = eng.filter(pl.col("status").is_in(["agree", "disagree"]))
        same_score = eng.filter("is_scored_same_score")
        unresolved = eng.filter(pl.col("status").is_in(["disagree", "missing"])).height
        scored = int(eng["is_scored"].sum())
        correct = int(eng["is_correct"].sum())
        orig_scored = orig.filter("is_scored")
        no_empty = orig_scored.filter(~pl.col("is_empty"))
        low, high = bootstrap_delta(pair)
        rows.append(
            {
                "community": name,
                "responses": eng.height,
                "original_scored": orig_scored.height,
                "original_accuracy": percent(orig_scored["is_correct"]),
                "original_accuracy_without_empty": percent(no_empty["is_correct"]),
                "original_empty": int(orig["is_empty"].sum()),
                "original_exact_accuracy": percent(orig["is_exact"]),
                "english_exact_accuracy": percent(answered["is_exact"]),
                "english_exact_original_reference": answered.filter(
                    pl.col("status") == "auto_exact_match_original"
                ).height,
                "english_exact_english_reference": answered.filter(
                    pl.col("status") == "auto_exact_match_english"
                ).height,
                "english_rule_scored": answered.filter(
                    pl.col("status").str.starts_with("auto_")
                ).height,
                "english_provider_errors": eng.filter(pl.col("status") == "provider_error").height,
                "english_scored": scored,
                "english_unresolved": unresolved,
                "english_accuracy": correct / scored * 100,
                "english_accuracy_low": correct / (scored + unresolved) * 100,
                "english_accuracy_high": (correct + unresolved) / (scored + unresolved) * 100,
                "english_accuracy_same_score": percent(same_score["is_correct"]),
                "english_scored_same_score": same_score.height,
                "paired_responses": pair.height,
                "paired_original": percent(pair["correct_original"]),
                "paired_english": percent(pair["correct_english"]),
                "paired_delta_low": low,
                "paired_delta_high": high,
                "judge_score_agreement": percent(
                    judged["score_gemini"] == judged["score_deepseek"]
                ),
                "judge_gemini_accept": percent(judged["score_gemini"] == 3),
                "judge_deepseek_accept": percent(judged["score_deepseek"] == 3),
                "judge_binary_kappa": binary_kappa(
                    judged["score_gemini"], judged["score_deepseek"]
                ),
            }
        )
    return pl.DataFrame(rows).with_columns(
        (pl.col("paired_english") - pl.col("paired_original")).alias("paired_delta")
    )


def model_summary(english: pl.DataFrame, original: pl.DataFrame) -> pl.DataFrame:
    """Return accuracy per model and community in both conditions."""

    def accuracy(frame: pl.DataFrame, label: str) -> pl.DataFrame:
        return (
            frame.filter("is_scored")
            .group_by("community_id", "model_name")
            .agg(
                pl.len().alias(f"{label}_scored"),
                (pl.col("is_correct").mean() * 100).alias(f"{label}_accuracy"),
            )
        )

    return (
        accuracy(original, "original")
        .join(accuracy(english, "english"), on=["community_id", "model_name"])
        .sort("community_id", "model_name")
    )


def summarize() -> None:
    """Write accuracy comparisons and a sensitivity without answer-revealing
    translations."""
    export = load_export()
    english = english_rows(export)
    original = original_rows()
    summary = community_summary(english, original)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.write_csv(OUTPUT_DIR / "accuracy_by_community.csv", float_precision=4)
    revealing = answer_in_riddle(export)["riddle_id"].implode()
    community_summary(
        english.filter(~pl.col("riddle_id").is_in(revealing)),
        original.filter(~pl.col("riddle_id").is_in(revealing)),
    ).write_csv(
        OUTPUT_DIR / "accuracy_by_community_without_answer_in_riddle.csv",
        float_precision=4,
    )
    model_summary(english, original).write_csv(
        OUTPUT_DIR / "accuracy_by_model.csv", float_precision=4
    )
    logger.info("Wrote English-condition comparison to %s", OUTPUT_DIR)


def main() -> None:
    """Dispatch the ``prepare`` or ``summarize`` step."""
    parser = argparse.ArgumentParser(description="English-translation condition study.")
    parser.add_argument("step", choices=("prepare", "summarize"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.step == "prepare":
        prepare()
    else:
        summarize()


if __name__ == "__main__":
    main()
