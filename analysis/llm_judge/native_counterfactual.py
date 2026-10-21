"""What would the results say without native speakers?

Replaces every human label with the LLM jury's verdict and recomputes the
headline results. Judges scored only the setting-B sample of human-labeled test
pairs, so the human-free score of the rest is estimated: for a model, a
community, or a model-community cell, the jury-only correct count is the current
count plus the design-weighted sum, over sampled pairs, of (jury correct - human
correct) times the pair's responses in that group. The denominator changes the
same way when a rule leaves a pair unscored. Rule-scored answers and
jury-scored answers without a human label stay as they are, and both pipelines
use test riddles only (the development riddles tuned the judge prompt).

Rules that resolve the two judges without a human:

* ``agree``: keep the score when both judges give the same score, otherwise
  leave the answer unscored (the rule the benchmark uses for unlabeled answers);
* ``lower``: the lower of the two scores;
* ``gemini`` and ``deepseek``: one judge alone.

Uncertainty comes from a stratified bootstrap over the sampled pairs (strata
sampled in full are kept fixed). Outputs go to
``data/derived/llm_judge/native_counterfactual/``.

Run from the repository root::

    uv run python -m analysis.llm_judge.native_counterfactual
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import orjson
import polars as pl
from scipy.sparse import csr_matrix
from scipy.stats import kendalltau

from analysis.constants import DERIVED_DIR, JUDGE_DATA_DIR, SCORED_RESPONSES_PATH
from analysis.corpus import load_corpus_metadata
from analysis.llm_judge.judge_data import judge_labels, load_pairs

OUT_DIR = DERIVED_DIR / "llm_judge" / "native_counterfactual"
RULES = ("agree", "lower", "gemini", "deepseek")
CORRECT = 3
N_BOOT = 2000
SEED = 20261017
N_EXTREME = 8  # easiest and hardest communities, as in the cultures figure
N_FRONTIER = 3
HARDEST = ("lumasaba_uganda", "mongondow_indonesia")
CI = (2.5, 97.5)


@dataclass(frozen=True)
class Design:
    """Current counts per cell and per-pair corrections for each rule.

    Attributes:
        models: Model display names, column order of every cell matrix.
        communities: Community keys, row order of every cell matrix.
        correct: Current correct count per (community, model).
        total: Current scored count per (community, model).
        pair_community: Sparse (communities, sampled pairs) membership matrix.
        pair_rows: Weighted responses per (sampled pair, model).
        d_correct: Per rule, jury-correct minus human-correct per sampled pair.
        d_total: Per rule, -1 where the rule leaves a sampled pair unscored.
        draw_start: First index of each pair's stratum in the sorted pair order.
        draw_size: Number of sampled pairs in that stratum, 0 if fully sampled.
    """

    models: list[str]
    communities: list[str]
    correct: np.ndarray
    total: np.ndarray
    pair_community: csr_matrix
    pair_rows: np.ndarray
    d_correct: dict[str, np.ndarray]
    d_total: dict[str, np.ndarray]
    draw_start: np.ndarray
    draw_size: np.ndarray


# a model that answered "Null" has its normalized answer stored as a null
ANSWER_KEY = pl.col("answer_norm").fill_null("null")


def jury_scores() -> pl.DataFrame:
    """Return setting-B pairs with design weights, human score, and rule scores."""
    pairs = load_pairs("setting_b").select(
        "riddle_id",
        ANSWER_KEY,
        "sampling_stratum",
        pl.col("sampling_weight").cast(pl.Float64),
        (
            pl.col("sampling_population").cast(pl.Int64)
            == pl.col("sampling_sample_size").cast(pl.Int64)
        ).alias("census"),
    )
    for judge in ("Gemini", "DeepSeek"):
        pairs = pairs.join(
            judge_labels("setting_b", judge).select("riddle_id", ANSWER_KEY, f"score_{judge}"),
            on=["riddle_id", "answer_norm"],
            how="left",
        )
    gem, ds = pl.col("score_Gemini"), pl.col("score_DeepSeek")
    return pairs.with_columns(
        pl.when(gem == ds).then(gem).alias("agree"),
        pl.min_horizontal(gem, ds).alias("lower"),
        gem.alias("gemini"),
        ds.alias("deepseek"),
    ).sort("sampling_stratum", "riddle_id", "answer_norm")


def build_design() -> Design:
    """Assemble current cell counts and the sampled-pair corrections.

    Returns:
        The :class:`Design` for the human-evaluated communities on test riddles.

    Raises:
        ValueError: If a sampled pair has no human-labeled response or the
            weighted sample does not reproduce the human-labeled population.
    """
    responses = pl.read_parquet(SCORED_RESPONSES_PATH)
    population = pl.read_parquet(JUDGE_DATA_DIR / "setting_b" / "population.parquet")
    dev = population.filter(pl.col("split") == "dev")["riddle_id"].unique()
    human_communities = responses.filter(pl.col("scorer") == "human")["community"].unique()
    rows = responses.filter(
        pl.col("community").is_in(human_communities.implode())
        & ~pl.col("riddle_id").is_in(dev.implode())
    ).with_columns(
        pl.col("score").is_not_null().alias("scored"),
        (pl.col("score") == CORRECT).fill_null(False).alias("is_correct"),
    )
    models = sorted(rows["model_display"].unique())
    communities = sorted(rows["community"].unique())
    cells = rows.group_by("community", "model_display").agg(
        pl.col("is_correct").sum().alias("correct"),
        pl.col("scored").sum().alias("total"),
    )

    def matrix(column: str) -> np.ndarray:
        wide = cells.pivot(on="model_display", index="community", values=column)
        wide = wide.sort("community").select(models).fill_null(0)
        return wide.to_numpy().astype(float)

    pairs = jury_scores()
    human = rows.filter(pl.col("scorer") == "human")
    per_model = (
        human.group_by("riddle_id", "answer_norm", "model_display")
        .len()
        .pivot(on="model_display", index=["riddle_id", "answer_norm"], values="len")
    )
    human_score = human.group_by("riddle_id", "answer_norm").agg(
        pl.col("community").first(), pl.col("score").first().alias("human")
    )
    pairs = pairs.join(human_score, on=["riddle_id", "answer_norm"], how="left").join(
        per_model, on=["riddle_id", "answer_norm"], how="left"
    )
    if pairs["human"].null_count():
        raise ValueError("sampled pairs without a human-labeled test response")
    counts = pairs.select(models).fill_null(0).to_numpy().astype(float)
    weight = pairs["sampling_weight"].to_numpy()
    covered = (counts.sum(axis=1) * weight).sum()
    if abs(covered - human.height) > 0.001 * human.height:
        raise ValueError(f"weighted sample covers {covered:.0f} of {human.height}")

    human_correct = (pairs["human"] == CORRECT).to_numpy().astype(float)
    d_correct, d_total = {}, {}
    for rule in RULES:
        score = pairs[rule]
        d_correct[rule] = (score == CORRECT).fill_null(False).to_numpy() - (human_correct)
        d_total[rule] = -score.is_null().to_numpy().astype(float)

    stratum = pairs["sampling_stratum"].to_numpy()
    _, start, size = np.unique(stratum, return_index=True, return_counts=True)
    stratum_index = np.repeat(np.arange(len(start)), size)
    census = pairs["census"].to_numpy()
    community_index = {c: i for i, c in enumerate(communities)}
    return Design(
        models=models,
        communities=communities,
        correct=matrix("correct"),
        total=matrix("total"),
        pair_community=csr_matrix(
            (
                np.ones(pairs.height),
                (
                    [community_index[c] for c in pairs["community"]],
                    np.arange(pairs.height),
                ),
            ),
            shape=(len(communities), pairs.height),
        ),
        pair_rows=counts * weight[:, None],
        d_correct=d_correct,
        d_total=d_total,
        draw_start=start[stratum_index],
        draw_size=np.where(census, 0, size[stratum_index]),
    )


def cell_counts(
    design: Design, rule: str, multiplicity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return jury-only correct and scored counts per (community, model).

    Args:
        design: Output of :func:`build_design`.
        rule: One of ``RULES``.
        multiplicity: Times each sampled pair is counted (all ones for the
            point estimate, bootstrap counts otherwise).

    Returns:
        Correct and scored counts, each of shape (communities, models).
    """
    rows = design.pair_rows * multiplicity[:, None]
    correct = design.correct + design.pair_community @ (rows * design.d_correct[rule][:, None])
    total = design.total + design.pair_community @ (rows * design.d_total[rule][:, None])
    return correct, total


def summarize(
    correct: np.ndarray,
    total: np.ndarray,
    extremes: tuple[np.ndarray, np.ndarray],
    frontier: np.ndarray,
    hardest: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Return the headline quantities for one set of cell counts.

    Args:
        correct: Correct count per (community, model).
        total: Scored count per (community, model).
        extremes: Row indices of the easiest and hardest communities.
        frontier: Column indices of the frontier models.
        hardest: Row indices of Lumasaba and Mongondow.

    Returns:
        Model accuracy, community mean accuracy, cell accuracy, the
        easiest-minus-hardest gap, frontier accuracy on the hardest
        communities, the best model's accuracy there, and frontier counts of
        communities below 50%.
    """
    cell = 100 * correct / total
    community = cell.mean(axis=1)
    easiest, hardest_rows = extremes
    return {
        "model": 100 * correct.sum(axis=0) / total.sum(axis=0),
        "community": community,
        "cell": cell,
        "gap": community[easiest].mean() - community[hardest_rows].mean(),
        "hardest_mean": community[hardest_rows].mean(),
        "frontier_hardest": cell[np.ix_(hardest, frontier)].mean(),
        "best_in_hardest": cell[hardest].max(),
        "frontier_below_50": (cell[:, frontier] < 50).sum(axis=0).astype(float),
    }


def bootstrap_draws(design: Design, rng: np.random.Generator) -> np.ndarray:
    """Return one stratified bootstrap multiplicity per sampled pair."""
    n = len(design.draw_start)
    picks = design.draw_start + np.floor(rng.random(n) * np.maximum(design.draw_size, 1)).astype(
        int
    )
    picks = np.where(design.draw_size == 0, np.arange(n), picks)
    return np.bincount(picks, minlength=n).astype(float)


def interval(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval along the first axis."""
    low, high = np.percentile(samples, CI, axis=0)
    return low, high


def run() -> dict[str, object]:
    """Estimate every rule, bootstrap the differences, and write the outputs.

    Returns:
        The summary written to ``summary.json``.
    """
    design = build_design()
    ones = np.ones(len(design.draw_start))
    selection = selectors(design, design.correct, design.total)
    base = summarize(design.correct, design.total, *selection)
    rng = np.random.default_rng(SEED)
    draws = [bootstrap_draws(design, rng) for _ in range(N_BOOT)]

    metadata = load_corpus_metadata().set_index("community")
    model_frames, community_frames, summary = [], [], {}
    for rule in RULES:
        point = summarize(*cell_counts(design, rule, ones), *selection)
        boot = [summarize(*cell_counts(design, rule, m), *selection) for m in draws]

        def diffs(key: str, boot=boot) -> np.ndarray:
            return np.array([b[key] - base[key] for b in boot])

        model_low, model_high = interval(diffs("model"))
        comm_low, comm_high = interval(diffs("community"))
        model_frames.append(
            pl.DataFrame(
                {
                    "rule": rule,
                    "model": design.models,
                    "current": base["model"],
                    "jury_only": point["model"],
                    "diff": point["model"] - base["model"],
                    "diff_low": model_low,
                    "diff_high": model_high,
                }
            )
        )
        community_frames.append(
            pl.DataFrame(
                {
                    "rule": rule,
                    "community": design.communities,
                    "resource": metadata.loc[design.communities, "language_resourcedness"].tolist(),
                    "current": base["community"],
                    "jury_only": point["community"],
                    "diff": point["community"] - base["community"],
                    "diff_low": comm_low,
                    "diff_high": comm_high,
                }
            )
        )
        rank_base = np.argsort(np.argsort(-base["model"]))
        rank_jury = np.argsort(np.argsort(-point["model"]))
        taus = [kendalltau(base["model"], b["model"]).statistic for b in boot]
        summary[rule] = {
            "kendall_tau": kendalltau(base["model"], point["model"]).statistic,
            "kendall_tau_ci": list(np.percentile(taus, CI)),
            "rank_changes": int((rank_base != rank_jury).sum()),
            "max_rank_change": int(np.abs(rank_base - rank_jury).max()),
            **{
                key: scalar_entry(base[key], point[key], diffs(key))
                for key in (
                    "gap",
                    "hardest_mean",
                    "frontier_hardest",
                    "best_in_hardest",
                )
            },
            "frontier_below_50": {
                "current": base["frontier_below_50"].tolist(),
                "jury_only": point["frontier_below_50"].tolist(),
                "jury_only_ci": np.percentile(
                    [b["frontier_below_50"] for b in boot], CI, axis=0
                ).T.tolist(),
            },
        }

    (easiest, hardest), frontier, _ = selection
    summary["design"] = {
        "communities": len(design.communities),
        "models": len(design.models),
        "sampled_pairs": len(design.draw_start),
        "bootstrap": N_BOOT,
        "easiest": [design.communities[i] for i in easiest],
        "hardest": [design.communities[i] for i in hardest],
        "frontier": [design.models[i] for i in frontier],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.concat(model_frames).write_csv(OUT_DIR / "model_accuracy.csv")
    pl.concat(community_frames).write_csv(OUT_DIR / "community_accuracy.csv")
    (OUT_DIR / "summary.json").write_bytes(
        orjson.dumps(summary, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY)
    )
    return summary


def selectors(
    design: Design, correct: np.ndarray, total: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    """Fix the easiest/hardest communities and frontier models on current scores."""
    cell = 100 * correct / total
    order = np.argsort(-cell.mean(axis=1), kind="stable")
    frontier = np.argsort(-correct.sum(axis=0) / total.sum(axis=0))[:N_FRONTIER]
    hardest = np.array([design.communities.index(c) for c in HARDEST])
    return (order[:N_EXTREME], order[-N_EXTREME:]), frontier, hardest


def scalar_entry(current: float, jury: float, diffs: np.ndarray) -> dict[str, object]:
    """Current value, jury-only value, and the difference with its interval."""
    return {
        "current": float(current),
        "jury_only": float(jury),
        "diff": float(jury - current),
        "diff_ci": [float(v) for v in np.percentile(diffs, CI)],
    }


if __name__ == "__main__":
    result = run()
    for rule in RULES:
        entry = result[rule]
        print(
            f"{rule:9s} tau {entry['kendall_tau']:.3f}  gap {entry['gap']['diff']:+.2f}"
            f"  frontier-hardest {entry['frontier_hardest']['diff']:+.2f}"
        )
