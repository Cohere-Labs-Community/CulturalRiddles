"""Variance shares of riddle difficulty, as the variance figure draws them.

Reads ``data/derived/difficulty_decomposition/variance_components.csv`` from
``fit_lme4.R``: each source's
share of benchmark difficulty (riddle, language, community, topic) and its share
of all variation on the latent log-odds scale.
"""

from __future__ import annotations

import polars as pl

from analysis.difficulty_decomposition.prepare import OUTPUT_DIR

LABELS = {
    "riddle_id": "Riddle",
    "language_code": "Language",
    "community": "Community",
    "topic_core": "Topic",
    "model_name": "Model",
    "residual": "Residual",
    "interactions": "Model $\\times$ facet",
}
BENCHMARK = ["riddle_id", "language_code", "community", "topic_core"]


def load_shares() -> pl.DataFrame:
    """Return one row per plotted source with both variance shares.

    Model-by-facet interactions are pooled into one row, and only the four
    benchmark facets carry a share of benchmark difficulty.

    Returns:
        Frame with ``label``, ``share`` (of all variation) and
        ``benchmark_share`` columns, ordered for plotting.
    """
    components = pl.read_csv(OUTPUT_DIR / "variance_components.csv").with_columns(
        pl.when(pl.col("component").str.contains(":"))
        .then(pl.lit("interactions"))
        .otherwise(pl.col("component"))
        .alias("source")
    )
    shares = components.group_by("source").agg(
        pl.col("share").sum(), pl.col("share_of_benchmark_difficulty").sum()
    )
    order = [*BENCHMARK, "model_name", "interactions", "residual"]
    return (
        shares.with_columns(
            pl.col("source").replace_strict(LABELS).alias("label"),
            pl.col("source").replace_strict({s: i for i, s in enumerate(order)}).alias("rank"),
            pl.when(pl.col("source").is_in(BENCHMARK))
            .then(pl.col("share_of_benchmark_difficulty"))
            .alias("benchmark_share"),
        )
        .sort("rank")
        .select("label", "share", "benchmark_share")
    )
