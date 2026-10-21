"""Judge reliability in every setting, and a check of the judge-scored answers.

Run from the repository root:
    uv run python -m analysis.llm_judge.pending_labels

Setting C holds every answer without a valid human label. Accuracy accepts such
an answer only when both judges return the same score; disagreements and
failed verdicts stay unscored. ``responses_scored_consolidated.parquet``
carries those agreed scores; this module checks them against the setting-C
verdicts and writes judge-versus-judge agreement for all three settings.
"""

from __future__ import annotations

import argparse

import polars as pl

from analysis.constants import SCORED_RESPONSES_PATH
from analysis.llm_judge.judge_data import judge_labels, load_pairs
from analysis.llm_judge.rater_agreement import (
    BOOTSTRAP,
    ENSEMBLE,
    FINE_LEVELS,
    JUDGES,
    PUBLIC_DIR,
    SEED,
    Bootstrap,
    ensemble_rows,
    iaa_h1_labels,
    marginal_rows,
    pair_rows,
    tag,
    with_ensemble,
)

# Judge-only agreement per setting. The doubly annotated pairs exclude riddles
# seen while tuning the judge prompt.
SETTINGS = {
    "doubly annotated": "setting_a",
    "stratified": "setting_b",
    "pending": "setting_c",
}


def pending_labels() -> pl.DataFrame:
    """Join both judges' setting-C scores; a missing verdict stays null.

    Pairs with a valid first-pass label in the doubly annotated sample are
    dropped, because accuracy scores them with that human label.
    """
    first, *rest = (judge_labels("setting_c", judge).drop(f"fine_{judge}") for judge in JUDGES)
    for frame in rest:
        first = first.join(frame, on=["riddle_id", "answer_norm"], how="full", coalesce=True)
    return first.join(
        iaa_h1_labels().select("riddle_id", "answer_norm"),
        on=["riddle_id", "answer_norm"],
        how="anti",
    )


def check_scored_responses() -> int:
    """Confirm that every judge-scored answer carries the judges' agreed score.

    Returns:
        The number of judge-scored answers.

    Raises:
        ValueError: If a judge-scored answer disagrees with the verdicts.
    """
    scores = [f"score_{judge}" for judge in JUDGES]
    agreed = pending_labels().filter(pl.col(scores[0]) == pl.col(scores[1]))
    judged = (
        pl.read_parquet(SCORED_RESPONSES_PATH)
        .filter(pl.col("scorer") == "llm_judges")
        .join(
            agreed.select("riddle_id", "answer_norm", expected=pl.col(scores[0])),
            on=["riddle_id", "answer_norm"],
            how="left",
        )
    )
    wrong = judged.filter(pl.col("expected").is_null() | (pl.col("score") != pl.col("expected")))
    if wrong.height:
        raise ValueError(f"{wrong.height} judge-scored answers disagree with the verdicts")
    return judged.height


def judge_pair_frame(setting: str) -> pl.DataFrame:
    """Join both judges' verdicts on one setting's pairs with their community.

    Where the sample carries the benchmark's human label, it is added as rater
    ``H1``; the single pair whose label lies outside the taxonomy is dropped.
    ``weight`` is the number of model responses a pair stands for, times its
    stratified-design weight when the sample has one.
    """
    raw = load_pairs(setting)
    if setting == "setting_a":
        raw = raw.filter(~pl.col("tuning_exposed"))
    if setting == "setting_c":
        raw = raw.join(
            iaa_h1_labels().select("riddle_id", "answer_norm"),
            on=["riddle_id", "answer_norm"],
            how="anti",
        )
    header = raw.columns
    frame = raw.unique(["riddle_id", "answer_norm"]).select(
        "riddle_id",
        "answer_norm",
        "community",
        score_H1=pl.col("score").cast(pl.Int8, strict=False),
        fine_H1=pl.col("label"),
        weight=pl.col("n_rows").cast(pl.Float64)
        * pl.col("sampling_weight").cast(pl.Float64).fill_null(1.0)
        if "sampling_weight" in header
        else pl.col("n_rows").cast(pl.Float64),
    )
    if frame["score_H1"].is_null().all():
        frame = frame.drop("score_H1", "fine_H1")
    else:
        frame = frame.filter(pl.col("fine_H1").is_in(FINE_LEVELS))
    for judge in JUDGES:
        frame = frame.join(
            judge_labels(setting, judge),
            on=["riddle_id", "answer_norm"],
            how="inner",
        )
    return with_ensemble(frame)


def judge_reliability(reps: int = BOOTSTRAP) -> list[dict[str, object]]:
    """Judge-versus-judge agreement and label rates in every setting.

    Every setting weights each (riddle, answer) pair by the model responses it
    represents, so values describe each setting's responses and are comparable
    across settings. Intervals resample riddles within communities.

    Args:
        reps: Bootstrap draws.

    Returns:
        Kappa, observed agreement, and per-judge correct rate for each setting;
        with a human label, also the ensemble's kappa and its paired gain over
        each single judge.
    """
    rows: list[dict[str, object]] = []
    for offset, (setting, name) in enumerate(SETTINGS.items()):
        frame = judge_pair_frame(name)
        boot = Bootstrap.for_frame(frame, reps, SEED + 900 + offset, "community")
        raters = [*(["H1"] if "score_H1" in frame.columns else []), *JUDGES]
        pairs = [
            (first, second) for index, first in enumerate(raters) for second in raters[index + 1 :]
        ]
        if "H1" in raters:
            pairs.append(("H1", ENSEMBLE))
        kappa, draws = pair_rows(frame, boot, pairs)
        if "H1" in raters:
            kappa += ensemble_rows(draws, ("H1",))
        ids = {
            "setting": setting,
            "units": frame.height,
            "riddles": frame["riddle_id"].n_unique(),
        }
        rows += tag(kappa + marginal_rows(frame, boot, [*raters, ENSEMBLE]), **ids)
    return rows


def main(argv: list[str] | None = None) -> int:
    """Check the judge-scored answers and write the reliability table."""
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(f"judge-scored answers match the verdicts: {check_scored_responses():,}")
    reliability = pl.DataFrame(judge_reliability(), infer_schema_length=None)
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    reliability.write_csv(PUBLIC_DIR / "judge_reliability.csv", float_precision=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
