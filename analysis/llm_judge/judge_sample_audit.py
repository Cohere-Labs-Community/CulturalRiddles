"""Audit whether the judge-study test sample represents the reviewed population.

Run from the repository root:
    uv run python -m analysis.llm_judge.judge_sample_audit

The test sample is stratified by community and fine human label, with a floor
per stratum, so it over-samples rare labels and small communities on purpose.
This audit compares the sample's unweighted and design-weighted label
distributions with the test-split population at both the coarse (0-3 score)
and fine (subcategory) levels, and reports how much the weights cost in
effective sample size.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from analysis.constants import DERIVED_DIR, JUDGE_DATA_DIR
from analysis.llm_judge.judge_data import load_pairs

PUBLIC_DIR = DERIVED_DIR / "llm_judge" / "judge_sample_audit"


def frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Test-split population pairs and the setting-B sample, with weights."""
    population = (
        pl.read_parquet(JUDGE_DATA_DIR / "setting_b" / "population.parquet")
        .filter(pl.col("split") == "test")
        .with_columns(n_rows=pl.col("n_rows").cast(pl.Int64))
    )
    sample = load_pairs("setting_b").with_columns(
        n_rows=pl.col("n_rows").cast(pl.Int64),
        pair_weight=pl.col("sampling_weight").cast(pl.Float64),
    )
    sample = sample.with_columns(row_weight=pl.col("pair_weight") * pl.col("n_rows"))
    return population, sample


def distribution(population: pl.DataFrame, sample: pl.DataFrame, column: str) -> pl.DataFrame:
    """Population and sample shares of each level, unweighted and weighted.

    Pair shares count unique (riddle, answer) pairs; row shares count model
    responses, which is the unit every judge-study metric is weighted to.
    """
    pop = population.group_by(column).agg(
        population_pairs=pl.len(), population_rows=pl.col("n_rows").sum()
    )
    smp = sample.group_by(column).agg(
        sample_pairs=pl.len(),
        weighted_pairs=pl.col("pair_weight").sum(),
        weighted_rows=pl.col("row_weight").sum(),
    )
    shares = [
        "population_pairs",
        "population_rows",
        "sample_pairs",
        "weighted_pairs",
        "weighted_rows",
    ]
    return (
        pop.join(smp, on=column, how="full", coalesce=True)
        .fill_null(0)
        .with_columns(
            [(pl.col(name) / pl.col(name).sum()).alias(f"{name}_share") for name in shares]
        )
        .with_columns(
            oversampling=pl.col("sample_pairs_share") / pl.col("population_pairs_share"),
            weighted_minus_population=pl.col("weighted_rows_share")
            - pl.col("population_rows_share"),
        )
        .rename({column: "level"})
        .with_columns(level=pl.col("level").cast(pl.String), granularity=pl.lit(column))
        .sort("level")
    )


def effective_size(sample: pl.DataFrame) -> pl.DataFrame:
    """Kish effective sample size of the row weights, pooled and per community."""
    kish = (pl.col("row_weight").sum() ** 2 / (pl.col("row_weight") ** 2).sum()).alias(
        "effective_pairs"
    )
    pooled = sample.select(pl.lit("Pooled").alias("community"), pl.len().alias("pairs"), kish)
    per = sample.group_by("community").agg(pl.len().alias("pairs"), kish).sort("community")
    return pl.concat([pooled, per]).with_columns(
        design_effect=pl.col("pairs") / pl.col("effective_pairs")
    )


def strata(sample: pl.DataFrame) -> pl.DataFrame:
    """Stratum sizes: census strata, floor-limited strata and the largest weights."""
    return (
        sample.group_by("sampling_stratum")
        .agg(
            population=pl.col("sampling_population").first().cast(pl.Int64),
            sampled=pl.len(),
            weight=pl.col("pair_weight").first(),
        )
        .with_columns(
            community=pl.col("sampling_stratum").str.split("|").list.first(),
            label=pl.col("sampling_stratum").str.split("|").list.last(),
            census=pl.col("sampled") == pl.col("population"),
        )
        .sort("weight", descending=True)
    )


def main(argv: list[str] | None = None) -> int:
    """Write distribution, effective-size and stratum tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=PUBLIC_DIR)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    population, sample = frames()
    population = population.with_columns(coarse=pl.col("score"))
    sample = sample.with_columns(coarse=pl.col("score"))
    pl.concat(
        [
            distribution(population, sample, "coarse"),
            distribution(population, sample, "label"),
        ]
    ).write_csv(args.out_dir / "label_distribution.csv", float_precision=10)
    effective_size(sample).write_csv(args.out_dir / "effective_size.csv")
    table = strata(sample).sort(pl.all())
    table.write_csv(args.out_dir / "strata.csv")
    with pl.Config(tbl_rows=20, float_precision=3, tbl_width_chars=200):
        print(
            pl.read_csv(args.out_dir / "label_distribution.csv").select(
                "granularity",
                "level",
                "population_pairs",
                "sample_pairs",
                "population_rows_share",
                "sample_pairs_share",
                "weighted_rows_share",
                "oversampling",
            )
        )
        print(effective_size(sample).head(1))
        print(table.select("census").to_series().value_counts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
