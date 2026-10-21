"""Build leakage-safe development and test inputs for the judge-agreement study.

Run from the repository root:
    uv run python -m riddles_framework.judging.judge_study_sample

The released pairs in ``data/canonical/judge/`` are the frozen output of this sampler,
drawn before nine riddles were removed from the corpus; rerunning it on the
released responses gives a similar but not identical draw.

Human labels are identical for every model that gave the same answer to the same
riddle, so the judge scores unique (riddle, normalized answer) pairs. Riddles
are split per community into a development pool (few-shot examples and prompt
tuning) and a test pool that is never used for prompt design.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path

import polars as pl

from analysis.constants import DERIVED_DIR, RIDDLES_PATH, SCORED_RESPONSES_PATH
from analysis.llm_judge.rater_agreement import double_annotations

STUDY_DIR = DERIVED_DIR / "llm_judge" / "sampling"
SEED = 20260928
DEV_FRACTION = 0.15
DEV_TUNE_PER_LABEL = 25
FEW_SHOT_LABELS = ("0.1", "0.2", "0.3", "0.4", "2.1", "2.2", "2.3", "3", "3.1", "3.2")
RUNNER_COLUMNS = (
    "riddle_id",
    "community",
    "topic",
    "riddle_original",
    "model_name",
    "model_answer",
    "reference_answer_original",
    "reference_answer_english",
    "score",
    "error_type",
    "validator_note",
    "scoring_method",
    "label",
    "answer_norm",
    "n_rows",
)


def stable_rank(value: str, seed: int = SEED) -> str:
    """Return a seed-dependent, platform-stable ordering key."""
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def scored_rows(path: Path = SCORED_RESPONSES_PATH) -> pl.DataFrame:
    """Read every reported answer as text columns, with its riddle text.

    ``scoring_method`` is the pipeline's routing, before any judge score.
    """
    riddles = pl.read_ndjson(RIDDLES_PATH, infer_schema_length=None).select(
        "riddle_id",
        "riddle_original",
        "reference_answer_original",
        "reference_answer_english",
    )
    rows = pl.read_parquet(path).rename({"model": "model_name", "routing": "scoring_method"})
    return (
        rows.with_columns(score=pl.when(pl.col("scorer") != "llm_judges").then(pl.col("score")))
        .join(riddles, on="riddle_id")
        .with_columns(pl.all().cast(pl.String))
    )


def load_pairs(path: Path) -> pl.DataFrame:
    """Collapse human-evaluated rows into unique (riddle, answer) pairs.

    Args:
        path: ``responses_scored_consolidated.parquet``.

    Returns:
        One row per pair with its human label, row multiplicity, and split.

    Raises:
        ValueError: If a pair carries conflicting human labels.
    """
    rows = (
        scored_rows(path)
        .filter((pl.col("scoring_method") == "human_eval") & pl.col("score").is_not_null())
        .with_columns(
            answer_norm=pl.col("model_answer").str.strip_chars().str.to_lowercase(),
            score=pl.col("score").cast(pl.Float64).cast(pl.Int8).cast(pl.String),
        )
        .with_columns(label=pl.coalesce("error_type", "score"))
        .with_columns(
            error_type=pl.when(pl.col("label") == "3").then(pl.lit("N/A")).otherwise("error_type")
        )
    )
    pairs = rows.group_by("riddle_id", "answer_norm", maintain_order=True).agg(
        pl.col(
            "community",
            "topic",
            "riddle_original",
            "model_answer",
            "reference_answer_original",
            "reference_answer_english",
            "score",
            "error_type",
            "validator_note",
            "scoring_method",
            "label",
        ).first(),
        n_labels=pl.col("label").n_unique(),
        n_rows=pl.len(),
    )
    if (conflicts := pairs.filter(pl.col("n_labels") > 1).height) > 0:
        raise ValueError(f"{conflicts} answer pairs have conflicting human labels")
    riddles = (
        pairs.select("community", "riddle_id")
        .unique()
        .with_columns(rank=pl.col("riddle_id").map_elements(stable_rank, return_dtype=pl.String))
        .with_columns(
            position=pl.col("rank").rank("ordinal").over("community"),
            total=pl.len().over("community"),
        )
        .select(
            "riddle_id",
            split=pl.when(pl.col("position") <= (pl.col("total") * DEV_FRACTION).ceil())
            .then(pl.lit("dev"))
            .otherwise(pl.lit("test")),
        )
    )
    return (
        pairs.drop("n_labels")
        .join(riddles, on="riddle_id")
        .with_columns(model_name=pl.lit("pair"), order=pl.col("answer_norm") + pl.col("riddle_id"))
    )


def shuffled(frame: pl.DataFrame, seed: int = SEED) -> pl.DataFrame:
    """Order rows by a stable hash so sampling is reproducible across platforms."""
    return frame.with_columns(
        _key=pl.col("order").map_elements(
            lambda value: stable_rank(value, seed), return_dtype=pl.String
        )
    ).sort("_key")


def few_shot_pool(dev: pl.DataFrame) -> pl.DataFrame:
    """Pick one short, clearly labeled example per category from distinct communities."""
    candidates = shuffled(
        dev.filter(
            pl.col("model_answer").str.len_chars().is_between(1, 60)
            & pl.col("riddle_original").str.len_chars().is_between(10, 220)
            & (pl.col("label").is_in(["0.4", "2.4"]) == pl.col("validator_note").is_not_null())
        ),
        SEED + 2,
    )
    reference = pl.col("reference_answer_original").str.to_lowercase().str.replace_all(r"\W", "")
    answer = pl.col("answer_norm").str.replace_all(r"\W", "")
    prototypical = {
        "3.1": ~reference.str.contains(answer, literal=True)
        & ~answer.str.contains(reference, literal=True),
        "3.2": pl.col("model_answer").str.contains(r"^[\x00-\x7F]+$")
        & ~pl.col("reference_answer_original").str.contains(r"^[\x00-\x7F]+$"),
    }
    chosen: list[pl.DataFrame] = []
    used: set[str] = set()
    for label in FEW_SHOT_LABELS:
        options = candidates.filter(
            (pl.col("label") == label)
            & ~pl.col("community").is_in(list(used))
            & prototypical.get(label, pl.lit(True))
        )
        if options.is_empty():
            raise ValueError(f"No development example available for label {label}")
        chosen.append(options.head(1))
        used.add(options["community"][0])
    return pl.concat(chosen)


def dev_tune_sample(dev: pl.DataFrame, excluded_riddles: set[str]) -> pl.DataFrame:
    """Draw a label-balanced tuning set whose riddles differ from the examples."""
    pool = shuffled(dev.filter(~pl.col("riddle_id").is_in(list(excluded_riddles))), SEED + 3)
    return shuffled(
        pool.filter(pl.col("label").is_in(FEW_SHOT_LABELS))
        .group_by("label", maintain_order=True)
        .head(DEV_TUNE_PER_LABEL),
        SEED + 4,
    )


def iaa_sample(pairs: pl.DataFrame, iaa: pl.DataFrame, exposed: set[str]) -> pl.DataFrame:
    """Build judge inputs for every doubly annotated (riddle, answer) pair.

    Riddle text and references come from the frozen pair table, so IAA answers
    that are absent from the current flat export can still be judged. Placeholder
    answers are resolved automatically in the benchmark and are skipped.

    Args:
        pairs: Frozen ``pairs_all.csv`` table.
        iaa: Output of ``rater_agreement.double_annotations``.
        exposed: Riddles shown to judges during prompt tuning or as examples.

    Returns:
        Runner-compatible rows labeled with annotator 1 and a tuning-exposure flag.
    """
    riddles = pairs.group_by("riddle_id").agg(
        pl.col(
            "topic",
            "riddle_original",
            "reference_answer_original",
            "reference_answer_english",
            "split",
        ).first()
    )
    return (
        iaa.filter(~pl.col("answer").str.contains(r"^<.*>$"))
        .join(riddles, on="riddle_id", how="inner", suffix="_corpus")
        .with_columns(
            model_name=pl.lit("pair"),
            model_answer=pl.col("answer"),
            answer_norm=pl.col("answer").str.strip_chars().str.to_lowercase(),
            score=pl.col("score_1").cast(pl.String),
            error_type=pl.col("error_type_label_1").str.extract(r"^\s*([0-3]\.\d)"),
            validator_note=pl.col("note_1"),
            scoring_method=pl.lit("human_eval"),
            n_rows=pl.lit(1),
            _iaa_row=pl.col("iaa_row"),
            _tuning_exposed=pl.col("riddle_id").is_in(list(exposed)),
        )
        .with_columns(label=pl.coalesce("error_type", "score"), topic=pl.col("topic"))
        .unique(["riddle_id", "answer_norm"], keep="first", maintain_order=True)
    )


def pending_pairs(path: Path, methods: tuple[str, ...] = ("pending_human_eval",)) -> pl.DataFrame:
    """Collapse responses without a valid human label into unique pairs.

    These pairs have no usable human label; the judges score them so that
    accuracy can count responses that humans have not validly reviewed. Any
    recorded (invalid) human label is dropped so the judges never see it.

    Args:
        path: ``responses_scored_consolidated.parquet``.
        methods: Scoring methods whose responses need judge labels.

    Returns:
        Runner-compatible rows with empty labels and their row multiplicity.
    """
    return (
        scored_rows(path)
        .filter(pl.col("scoring_method").is_in(methods))
        .with_columns(answer_norm=pl.col("model_answer").str.strip_chars().str.to_lowercase())
        .group_by("riddle_id", "answer_norm", maintain_order=True)
        .agg(
            pl.col(
                "community",
                "topic",
                "riddle_original",
                "model_answer",
                "reference_answer_original",
                "reference_answer_english",
                "scoring_method",
            ).first(),
            n_rows=pl.len(),
        )
        .with_columns(
            model_name=pl.lit("pair"),
            score=pl.lit(None, pl.String),
            error_type=pl.lit(None, pl.String),
            validator_note=pl.lit(None, pl.String),
            label=pl.lit(None, pl.String),
            split=pl.lit("pending"),
        )
    )


def allocate_proportional(
    measure: dict[str, float], total: int, floors: dict[str, int], caps: dict[str, int]
) -> dict[str, int]:
    """Allocate an integer total proportionally to a size measure.

    Each unit first receives its floor (never more than its cap). The remainder
    is split in proportion to ``measure`` among units below their cap, with
    largest-remainder rounding, and repeated while caps bind.

    Args:
        measure: Size measure per unit, such as model rows.
        total: Target total, reduced to the sum of caps when that is smaller.
        floors: Minimum allocation per unit.
        caps: Maximum allocation per unit, such as its population size.

    Returns:
        Integer allocation per unit that sums to the attainable total.
    """
    allocation = {unit: min(floors.get(unit, 0), caps[unit]) for unit in measure}
    remaining = min(total, sum(caps.values())) - sum(allocation.values())
    while remaining > 0:
        open_units = [unit for unit in measure if allocation[unit] < caps[unit]]
        pool = sum(measure[unit] for unit in open_units)
        exact = {unit: remaining * measure[unit] / pool for unit in open_units}
        extra = {unit: min(int(exact[unit]), caps[unit] - allocation[unit]) for unit in open_units}
        leftover = remaining - sum(extra.values())
        for unit in sorted(open_units, key=lambda u: exact[u] - int(exact[u]), reverse=True):
            if leftover == 0:
                break
            if allocation[unit] + extra[unit] < caps[unit]:
                extra[unit] += 1
                leftover -= 1
        for unit, value in extra.items():
            allocation[unit] += value
        remaining = min(total, sum(caps.values())) - sum(allocation.values())
    return allocation


def systematic_pps(stratum: pl.DataFrame, size: int, key: str) -> pl.DataFrame:
    """Draw a systematic sample with probability proportional to model rows.

    Pairs whose probability would reach 1 are taken with certainty. The rest
    are ordered by a stable hash and selected systematically from a seeded
    random start, so every non-certainty pair has inclusion probability
    ``size_left * n_rows / rows_left``.

    Args:
        stratum: Candidate pairs with ``n_rows`` and ``order`` columns.
        size: Number of pairs to draw.
        key: Stratum name that seeds the random start.

    Returns:
        Selected pairs with an ``_sampling_inclusion_prob`` column.
    """
    rows = shuffled(stratum, SEED + 5).with_columns(x=pl.col("n_rows").cast(pl.Float64))
    certain = pl.Series([False] * rows.height)
    while True:
        left = size - int(certain.sum())
        mass = rows.filter(~certain)["x"].sum()
        prob = pl.Series([1.0 if flag else left * x / mass for flag, x in zip(certain, rows["x"])])
        newly = (prob >= 1.0) & ~certain
        if not newly.any():
            break
        certain = certain | newly
    start = int(stable_rank(key, SEED + 6), 16) / 16**64
    cumulative = prob.filter(~certain).cum_sum()
    ticks = (cumulative - start).ceil()
    chosen = (ticks - ticks.shift(1, fill_value=0.0 if start > 0 else -1.0)) > 0
    selected = pl.Series([False] * rows.height)
    selected = (
        selected.scatter([index for index, flag in enumerate(certain) if not flag], chosen)
        | certain
    )
    return rows.with_columns(_sampling_inclusion_prob=prob).filter(selected).drop("x", "_key")


def test_sample(
    pairs: pl.DataFrame,
    size: int,
    community_floor: int,
    class_floor: int,
    minimum_sizes: Mapping[str, int] | None = None,
) -> pl.DataFrame:
    """Draw a lower-design-effect test sample stratified by community and score.

    Communities receive a floor and then a share proportional to their model
    rows. Within a community, strata are the coarse human scores 0-3: score 1
    (explicit abstention) is taken in full, scores 2 and 3 get a floor so
    partial and correct answers are measurable per community, and the rest is
    proportional to rows. Pairs are drawn with probability proportional to
    ``n_rows``, which makes row-level design weights nearly constant within a
    stratum. Fine subcategories are not stratified, so they appear in
    population proportions.

    Args:
        pairs: Frozen pair table; only its test split is sampled.
        size: Target number of pairs.
        community_floor: Minimum pairs per community.
        class_floor: Minimum pairs for scores 2 and 3 within a community.
        minimum_sizes: Per-community sizes that raise the proportional
            allocation, capped at the community's pairs; other communities keep
            the allocation of ``size``.

    Returns:
        Sampled pairs with stratum, inclusion probability and design weight.
    """
    test = pairs.filter(pl.col("split") == "test").with_columns(
        n_rows=pl.col("n_rows").cast(pl.Int64),
        order=pl.col("answer_norm").fill_null("") + pl.col("riddle_id"),
    )
    by_community = test.group_by("community").agg(rows=pl.col("n_rows").sum(), pairs=pl.len())
    communities = allocate_proportional(
        dict(zip(by_community["community"], by_community["rows"])),
        size,
        dict.fromkeys(by_community["community"], community_floor),
        dict(zip(by_community["community"], by_community["pairs"])),
    )
    caps = dict(zip(by_community["community"], by_community["pairs"]))
    for community, minimum in (minimum_sizes or {}).items():
        communities[community] = min(caps[community], max(communities[community], minimum))
    parts = []
    for (community,), group in test.sort("community").group_by("community", maintain_order=True):
        strata = group.group_by("score").agg(rows=pl.col("n_rows").sum(), pairs=pl.len())
        caps = dict(zip(strata["score"], strata["pairs"]))
        floors = {"1": caps.get("1", 0), "2": class_floor, "3": class_floor}
        allocation = allocate_proportional(
            dict(zip(strata["score"], strata["rows"])),
            communities[community],
            floors,
            caps,
        )
        for score, draw in allocation.items():
            if not draw:
                continue
            key = f"{community}|{score}"
            stratum = group.filter(pl.col("score") == score)
            parts.append(
                systematic_pps(stratum, draw, key).with_columns(
                    _sampling_stratum=pl.lit(key),
                    _sampling_population=pl.lit(stratum.height),
                    _sampling_sample_size=pl.lit(draw),
                    _sampling_weight=1 / pl.col("_sampling_inclusion_prob"),
                )
            )
    return shuffled(pl.concat(parts), SEED + 7)


def write(frame: pl.DataFrame, path: Path) -> None:
    """Write runner-compatible CSV columns plus any sampling metadata."""
    columns = [*RUNNER_COLUMNS, "split"] + [
        column for column in frame.columns if column.startswith(("_sampling_", "_iaa_", "_tuning_"))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.select(columns).write_csv(path)


def main(argv: list[str] | None = None) -> int:
    """Write the pair table, few-shot pool, development set, or one study sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SCORED_RESPONSES_PATH)
    parser.add_argument("--out-dir", type=Path, default=STUDY_DIR)
    parser.add_argument(
        "--iaa",
        action="store_true",
        help="only write iaa_sample.csv and iaa_sample_primary.csv (no riddle "
        "seen in prompt tuning) from the frozen pair table and IAA export",
    )
    parser.add_argument(
        "--design",
        type=int,
        metavar="SIZE",
        help="only write test_sample.csv: community x score strata, PPS by rows",
    )
    parser.add_argument(
        "--pending",
        action="store_true",
        help="only write pending_pairs.csv: unique pairs without a valid human label",
    )
    parser.add_argument(
        "--community-minimums",
        type=Path,
        help="CSV of community,target_pairs that raises some communities' sizes "
        "(data/canonical/judge/setting_b/community_minimums.csv for setting B)",
    )
    parser.add_argument("--community-floor", type=int, default=200)
    parser.add_argument("--class-floor", type=int, default=40)
    args = parser.parse_args(argv)
    if args.pending:
        pending = pending_pairs(args.source, ("pending_human_eval", "invalid_human_eval"))
        write(pending, args.out_dir / "pending_pairs.csv")
        print(
            f"pending={pending.height} rows={pending['n_rows'].cast(pl.Int64).sum()} "
            f"communities={pending['community'].n_unique()}"
        )
        return 0
    if args.design:
        sample = test_sample(
            pl.read_csv(args.out_dir / "pairs_all.csv", infer_schema_length=0),
            args.design,
            args.community_floor,
            args.class_floor,
            dict(
                pl.read_csv(args.community_minimums).iter_rows() if args.community_minimums else []
            ),
        )
        write(sample, args.out_dir / "test_sample.csv")
        print(f"test={sample.height} riddles={sample['riddle_id'].n_unique()}")
        return 0
    if args.iaa:
        frozen = [
            pl.read_csv(args.out_dir / name, infer_schema_length=0)
            for name in ("pairs_all.csv", "dev_tune.csv", "few_shot_examples.csv")
        ]
        exposed = set(frozen[1]["riddle_id"]) | set(frozen[2]["riddle_id"])
        sample = iaa_sample(frozen[0], double_annotations(), exposed)
        primary = sample.filter(~pl.col("_tuning_exposed"))
        write(sample, args.out_dir / "iaa_sample.csv")
        write(primary, args.out_dir / "iaa_sample_primary.csv")
        print(
            f"iaa={sample.height} riddles={sample['riddle_id'].n_unique()} "
            f"tuning_exposed={sample['_tuning_exposed'].sum()} -> {args.out_dir}"
        )
        return 0
    pairs = load_pairs(args.source)
    dev = pairs.filter(pl.col("split") == "dev")
    examples = few_shot_pool(dev)
    tune = dev_tune_sample(dev, set(examples["riddle_id"]))
    write(pairs, args.out_dir / "pairs_all.csv")
    write(examples, args.out_dir / "few_shot_examples.csv")
    write(tune, args.out_dir / "dev_tune.csv")
    print(
        f"pairs={pairs.height} dev_pairs={dev.height} examples={examples.height} "
        f"dev_tune={tune.height} -> {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
