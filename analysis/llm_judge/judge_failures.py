"""Characterize where the zero-shot judges disagree with human correctness labels.

Run from the repository root:
    uv run python -m analysis.llm_judge.judge_failures

Uses the stratified sample of human-labeled test pairs (setting B). Rates are weighted back to the
human-reviewed population with the study design weights, and every interval is
a 95% riddle-cluster bootstrap interval. The hybrid rater keeps the judges'
verdict when both give the same score, so its residual errors are the errors
that the proposed pipeline cannot route to a human.
"""

from __future__ import annotations

import argparse
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl
import regex
from rapidfuzz import fuzz

from analysis.constants import DERIVED_DIR, FIXED_TOPICS
from analysis.llm_judge.judge_agreement import Condition, joined, load_sample
from analysis.llm_judge.judge_data import JUDGES
from analysis.llm_judge.rater_agreement import (
    BOOTSTRAP,
    SEED,
    Bootstrap,
    interval,
    ratio,
)

PUBLIC_DIR = DERIVED_DIR / "llm_judge" / "judge_failures"
LATIN = regex.compile(r"\p{Latin}")
LETTER = regex.compile(r"\p{L}")
SIMILARITY_BINS = (0, 20, 40, 60, 80, 101)
FEATURES = (
    "reference_similarity",
    "containment",
    "script",
    "answer_words",
    "human_label",
    "language_resourcedness",
    "topic",
)


def latin_share(text: str) -> float:
    """Share of letters in Latin script (0 when the text has no letters)."""
    letters = len(LETTER.findall(text))
    return len(LATIN.findall(text)) / letters if letters else 0.0


def surface_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Add answer-reference similarity, containment, script, and length bins."""
    answers = frame["model_answer"].fill_null("").str.strip_chars().str.to_lowercase()
    references = [
        [value or "" for value in values]
        for values in zip(
            frame["reference_answer_original"].str.to_lowercase().to_list(),
            frame["reference_answer_english"].str.to_lowercase().to_list(),
        )
    ]
    similarity = [
        max(fuzz.ratio(answer, reference) for reference in refs if reference)
        for answer, refs in zip(answers.to_list(), references)
    ]
    containment = [
        "reference inside answer"
        if any(ref and ref in answer for ref in refs)
        else "answer inside reference"
        if answer and any(answer in ref for ref in refs)
        else "neither"
        for answer, refs in zip(answers.to_list(), references)
    ]
    answer_latin = [latin_share(answer) > 0.5 for answer in answers.to_list()]
    reference_latin = [latin_share(refs[0]) > 0.5 for refs in references]
    script = [
        "same script"
        if first == second
        else "Latin answer, non-Latin reference"
        if first
        else "non-Latin answer, Latin reference"
        for first, second in zip(answer_latin, reference_latin)
    ]
    words = answers.str.split(" ").list.len()
    labels = [f"{low}-{high - 1}" for low, high in pairwise(SIMILARITY_BINS)]
    return frame.with_columns(
        reference_similarity=pl.Series(similarity)
        .cut(list(SIMILARITY_BINS[1:-1]), labels=labels, left_closed=True)
        .cast(pl.String),
        containment=pl.Series(containment),
        topic=pl.when(pl.col("topic").is_in(FIXED_TOPICS))
        .then("topic")
        .otherwise(pl.lit("Wildcard")),
        script=pl.Series(script),
        answer_words=pl.when(words == 1)
        .then(pl.lit("1"))
        .when(words <= 3)
        .then(pl.lit("2-3"))
        .when(words <= 7)
        .then(pl.lit("4-7"))
        .otherwise(pl.lit("8+")),
    )


def judged_sample() -> pl.DataFrame:
    """Test sample with human and zero-shot judge correctness indicators."""
    conditions = [Condition(model, "zero") for model in JUDGES.values()]
    frame = joined(load_sample(), conditions)
    scores = {name: pl.col(f"score|{model}|zero") for name, model in JUDGES.items()}
    frame = frame.filter(
        pl.all_horizontal(score.is_not_null() for score in scores.values())
    ).with_columns(
        human_correct=pl.col("human_score") == 3,
        **{f"{name}_correct": score == 3 for name, score in scores.items()},
        Hybrid_correct=pl.when(scores["DeepSeek"] == scores["Gemini"])
        .then(scores["Gemini"] == 3)
        .otherwise(pl.col("human_score") == 3),
    )
    return surface_features(frame)


def feature_rows(frame: pl.DataFrame, boot: Bootstrap, feature: str) -> list[dict[str, object]]:
    """False-accept and false-reject rates of each rater within each feature level."""
    weight = frame["weight"].to_numpy()
    human = frame["human_correct"].to_numpy()
    levels = sorted(frame[feature].cast(pl.String).unique().to_list())
    values = frame[feature].cast(pl.String).to_numpy()
    rows: list[dict[str, object]] = []
    for rater in (*JUDGES, "Hybrid"):
        judge = frame[f"{rater}_correct"].to_numpy()
        for outcome, eligible, error in (
            ("false_accept", ~human, judge),
            ("false_reject", human, ~judge),
        ):
            for level in levels:
                in_level = eligible & (values == level)
                if not in_level.any():
                    continue
                units = np.stack([weight * in_level * error, weight * in_level], 1)
                point, reps = boot.stat(units, ratio(0, 1))
                rows.append(
                    {
                        "rater": rater,
                        "outcome": outcome,
                        "feature": feature,
                        "level": level,
                        "pairs": int(in_level.sum()),
                        "population_share": float(
                            (weight * in_level).sum() / (weight * eligible).sum()
                        ),
                    }
                    | interval(point, reps)
                )
    return rows


def error_correlation(frame: pl.DataFrame, boot: Bootstrap) -> list[dict[str, object]]:
    """How often the two judges err on the same pairs, relative to independence."""
    weight = frame["weight"].to_numpy()
    human = frame["human_correct"].to_numpy()
    errors = {name: frame[f"{name}_correct"].to_numpy() != human for name in JUDGES}
    first, second = errors.values()
    units = weight[:, None] * np.stack(
        [first, second, first & second, np.ones_like(first)], 1
    ).astype(float)
    stats = {
        "DeepSeek_error_rate": ratio(0, 3),
        "Gemini_error_rate": ratio(1, 3),
        "both_wrong_rate": ratio(2, 3),
        "share_of_DeepSeek_errors_Gemini_also_makes": ratio(2, 0),
        "share_of_Gemini_errors_DeepSeek_also_makes": ratio(2, 1),
        "lift_over_independence": lambda t: t[..., 2] * t[..., 3] / (t[..., 0] * t[..., 1]),
    }
    rows = []
    for name, statistic in stats.items():
        point, reps = boot.stat(units, statistic)
        rows.append({"statistic": name} | interval(point, reps))
    return rows


def main(argv: list[str] | None = None) -> int:
    """Write feature-level error rates and judge error correlation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=PUBLIC_DIR)
    parser.add_argument("--reps", type=int, default=BOOTSTRAP)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = judged_sample()
    boot = Bootstrap.for_frame(frame, args.reps, SEED, stratum="community")
    rows = [row for feature in FEATURES for row in feature_rows(frame, boot, feature)]
    pl.DataFrame(rows).write_csv(args.out_dir / "error_rates_by_feature.csv")
    correlation = pl.DataFrame(error_correlation(frame, boot))
    correlation.write_csv(args.out_dir / "error_correlation.csv")
    with pl.Config(tbl_rows=12, float_precision=3):
        print(correlation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
