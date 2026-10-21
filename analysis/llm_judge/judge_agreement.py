"""Measure LLM-judge agreement with the exact-match plus human evaluation pipeline.

Run from the repository root:
    uv run python -m analysis.llm_judge.judge_agreement

Every judged unit is a unique (riddle, normalized answer) pair. Metrics are
weighted back to human-evaluated rows with the stratified design weight times
the number of model rows that share the pair. Confidence intervals come from a
riddle-cluster bootstrap.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from analysis.constants import CORPUS_METADATA_PATH, DERIVED_DIR, SCORED_RESPONSES_PATH
from analysis.llm_judge.judge_data import (
    JUDGES,
    has_condition,
    load_pairs,
    load_verdicts,
    run_summaries,
)

PUBLIC_DIR = DERIVED_DIR / "llm_judge"
MODELS = ("gemini-3.5-flash", "deepseek-v4.1-flash")
JUDGE_KEYS = {model: key for key, model in JUDGES.items()}
SETTING = "setting_b"
SHOTS = ("zero", "few10")
SCORES = (0, 1, 2, 3)
FINE = ("0.1", "0.2", "0.3", "0.4", "2.1", "2.2", "2.3", "2.4", "3", "3.1", "3.2")
AUTO_SCORED = ("auto_exact_match", "auto_empty", "auto_idk", "auto_placeholder")
BOOTSTRAP = 1000
SEED = 20260928


@dataclass(frozen=True)
class Condition:
    """One judge model and example condition evaluated on setting B."""

    model: str
    shots: str

    @property
    def name(self) -> str:
        """Short label used in tables."""
        return f"{self.model}|{self.shots}"


def normalize(column: str) -> pl.Expr:
    """Normalize answers exactly as the sampler does."""
    return pl.col(column).str.strip_chars().str.to_lowercase()


def fine_label(score: pl.Expr, category: pl.Expr) -> pl.Expr:
    """Map a score and category to the human taxonomy label (bare 3 = variant)."""
    return (
        pl.when(category.is_in(["N/A", "NA"]) | category.is_null())
        .then(score.cast(pl.String))
        .otherwise(category)
    )


def load_condition(condition: Condition) -> pl.DataFrame:
    """Read one condition's verdicts keyed by riddle and normalized answer."""
    return (
        load_verdicts(SETTING, JUDGE_KEYS[condition.model], condition.shots)
        .with_columns(
            judge_score=pl.col("score"),
            judge_label=fine_label(pl.col("score"), pl.col("error_category")),
        )
        .select("riddle_id", "answer_norm", "judge_score", "judge_label")
    )


def load_sample() -> pl.DataFrame:
    """Load the setting-B pairs with row-level population weights."""
    frame = load_pairs(SETTING)
    weight = pl.col("sampling_weight").cast(pl.Float64)
    metadata = pl.read_csv(CORPUS_METADATA_PATH).select(
        "community", "language_resourcedness", "script_family", "un_m49_subregion"
    )
    return frame.with_columns(
        human_score=pl.col("score").cast(pl.Int8),
        human_label=pl.col("label"),
        weight=weight * pl.col("n_rows").cast(pl.Float64),
        pair_weight=weight,
    ).join(metadata, on="community", how="left")


def confusion(truth: np.ndarray, pred: np.ndarray, weight: np.ndarray, k: int) -> np.ndarray:
    """Weighted k-by-k confusion matrix from integer-coded labels."""
    return np.bincount(truth * k + pred, weights=weight, minlength=k * k).reshape(k, k)


def kappa_from(matrix: np.ndarray, quadratic: bool = False) -> float:
    """Cohen's kappa (optionally quadratic-weighted) from a confusion matrix."""
    total = matrix.sum()
    if total <= 0:
        return float("nan")
    k = matrix.shape[0]
    grid = np.arange(k)
    disagreement = (
        ((grid[:, None] - grid[None, :]) / max(k - 1, 1)) ** 2
        if quadratic
        else (grid[:, None] != grid[None, :]).astype(float)
    )
    expected = np.outer(matrix.sum(1), matrix.sum(0)) / total
    denominator = (disagreement * expected).sum()
    if denominator == 0:
        return float("nan")
    return float(1 - (disagreement * matrix).sum() / denominator)


def bootstrap_weights(
    clusters: np.ndarray, reps: int = BOOTSTRAP, strata: np.ndarray | None = None
) -> np.ndarray:
    """Riddle-cluster bootstrap multipliers, shape (reps, rows).

    Args:
        clusters: Riddle identifier of every row.
        reps: Number of bootstrap draws.
        strata: Optional sampling stratum (community) of every row. When given,
            riddles are resampled within each stratum, matching a design in
            which every community's sample size is fixed.

    Returns:
        Integer multiplicity of each row's riddle in each draw.
    """
    codes, index = np.unique(clusters, return_inverse=True)
    rng = np.random.default_rng(SEED)
    if strata is None:
        draws = rng.multinomial(len(codes), np.full(len(codes), 1 / len(codes)), reps)
        return draws[:, index]
    cluster_stratum = np.empty(len(codes), dtype=object)
    cluster_stratum[index] = strata
    draws = np.zeros((reps, len(codes)), dtype=np.int64)
    for stratum in np.unique(cluster_stratum):
        members = np.flatnonzero(cluster_stratum == stratum)
        draws[:, members] = rng.multinomial(
            len(members), np.full(len(members), 1 / len(members)), reps
        )
    return draws[:, index]


def metric_row(
    truth: np.ndarray,
    pred: np.ndarray,
    weight: np.ndarray,
    multipliers: np.ndarray,
    k: int,
    quadratic: bool = False,
) -> dict[str, float]:
    """Point estimate and 95% bootstrap interval for one kappa."""
    point = kappa_from(confusion(truth, pred, weight, k), quadratic)
    reps = np.array(
        [kappa_from(confusion(truth, pred, weight * m, k), quadratic) for m in multipliers]
    )
    low, high = np.nanpercentile(reps, [2.5, 97.5])
    return {"kappa": point, "low": float(low), "high": float(high)}


def codes(values: pl.Series, levels: Sequence[str | int]) -> np.ndarray:
    """Integer-code a label column against a fixed level list."""
    lookup = {str(level): i for i, level in enumerate(levels)}
    return np.array([lookup[str(value)] for value in values.to_list()])


def agreement(frame: pl.DataFrame, multipliers: np.ndarray) -> list[dict[str, object]]:
    """Kappa at every granularity plus correct-class precision and recall."""
    weight = frame["weight"].to_numpy()
    human3 = (frame["human_score"] == 3).cast(pl.Int8).to_numpy()
    judge3 = (frame["judge_score"] == 3).cast(pl.Int8).to_numpy()
    human_s, judge_s = (
        codes(frame["human_score"], SCORES),
        codes(frame["judge_score"], SCORES),
    )
    labels = sorted(set(FINE) | set(frame["human_label"]) | set(frame["judge_label"]))
    merged = {"3.1": "3", "3.2": "3", "3.3": "3"}
    coarse_levels = sorted({merged.get(label, label) for label in labels})
    rows: list[dict[str, object]] = [
        {"level": "binary (score 3 vs rest)"} | metric_row(human3, judge3, weight, multipliers, 2),
        {"level": "score 0-3 (unweighted)"} | metric_row(human_s, judge_s, weight, multipliers, 4),
        {"level": "score 0-3 (quadratic)"}
        | metric_row(human_s, judge_s, weight, multipliers, 4, quadratic=True),
        {"level": "subcategory (all 3.x merged)"}
        | metric_row(
            codes(frame["human_label"].replace(merged), coarse_levels),
            codes(frame["judge_label"].replace(merged), coarse_levels),
            weight,
            multipliers,
            len(coarse_levels),
        ),
        {"level": "subcategory (fine)"}
        | metric_row(
            codes(frame["human_label"], labels),
            codes(frame["judge_label"], labels),
            weight,
            multipliers,
            len(labels),
        ),
    ]
    if "accepted" in frame.columns:
        accepted = frame["accepted"].to_numpy()
        rows.append(
            {"level": "binary (judge-accepted rows only)"}
            | metric_row(
                human3[accepted],
                judge3[accepted],
                weight[accepted],
                multipliers[:, accepted],
                2,
            )
        )
    true_positive = (weight * human3 * judge3).sum()
    rows.append(
        {
            "level": "correct-class precision / recall",
            "precision": true_positive / (weight * judge3).sum(),
            "recall": true_positive / (weight * human3).sum(),
            "human_correct_rate": (weight * human3).sum() / weight.sum(),
            "judge_correct_rate": (weight * judge3).sum() / weight.sum(),
            "exact_label_agreement": float(
                (weight * (frame["human_label"] == frame["judge_label"]).to_numpy()).sum()
                / weight.sum()
            ),
        }
    )
    return rows


def joined(sample: pl.DataFrame, conditions: Sequence[Condition]) -> pl.DataFrame:
    """Attach each condition's verdict columns, suffixed by condition name."""
    frame = sample
    for condition in conditions:
        verdicts = load_condition(condition).rename(
            {
                "judge_score": f"score|{condition.name}",
                "judge_label": f"label|{condition.name}",
            }
        )
        frame = frame.join(verdicts, on=["riddle_id", "answer_norm"], how="left")
    return frame


def view(frame: pl.DataFrame, name: str) -> pl.DataFrame:
    """Select one condition's verdicts as judge_score and judge_label."""
    return frame.with_columns(
        judge_score=pl.col(f"score|{name}"), judge_label=pl.col(f"label|{name}")
    ).filter(pl.col("judge_score").is_not_null())


def ensembles(frame: pl.DataFrame, names: Sequence[str]) -> dict[str, pl.DataFrame]:
    """Build human-in-the-loop combinations of judge conditions.

    ``agree-or-human`` keeps the judges' label when they agree and routes the
    rest to the human label, as a benchmarker would. ``ensemble-min`` uses no
    human label: it keeps the lower of the two scores, so an answer is correct
    only when both judges accept it. Rows with a missing verdict (technical
    failure) receive the human label under either rule.
    """
    complete = frame.with_columns(
        [pl.col(f"score|{name}").fill_null(-1) for name in names]
        + [pl.col(f"label|{name}").fill_null("missing") for name in names]
    )
    scores = complete.select([f"score|{name}" for name in names]).to_numpy()
    labels = complete.select([f"label|{name}" for name in names]).to_numpy()
    results = {}
    best_pairs = [
        (f"{MODELS[0]}|{shots}", f"{MODELS[1]}|{shots}")
        for shots in SHOTS
        if f"{MODELS[0]}|{shots}" in names and f"{MODELS[1]}|{shots}" in names
    ]
    for first, second in best_pairs:
        i, j = names.index(first), names.index(second)
        shots = first.split("|")[1]
        agree = (scores[:, i] == scores[:, j]) & (scores[:, i] >= 0)
        results[f"agree-or-human ({shots})"] = (agree, i)
        # Human-free ensemble: the lower score wins, so "correct" needs both.
        both = (scores[:, i] >= 0) & (scores[:, j] >= 0)
        results[f"ensemble-min ({shots})"] = (
            both,
            np.where(scores[:, j] < scores[:, i], j, i),
        )
    if len(names) == 4:
        votes = np.stack([(scores == s).sum(1) for s in SCORES], 1)
        winner = votes.argmax(1)
        agree = votes.max(1) >= 3
        pick = np.argmax(scores == np.array(SCORES)[winner][:, None], axis=1)
        results["3-of-4 majority-or-human"] = (agree, pick)
    output = {}
    for name, (accepted, pick) in results.items():
        chosen = np.broadcast_to(pick, accepted.shape)
        judge_score = scores[np.arange(len(scores)), chosen]
        judge_label = labels[np.arange(len(labels)), chosen]
        output[name] = complete.with_columns(
            accepted=pl.Series(accepted),
            judge_score=pl.when(pl.Series(accepted))
            .then(pl.Series(judge_score.astype(np.int64)))
            .otherwise(pl.col("human_score").cast(pl.Int64)),
            judge_label=pl.when(pl.Series(accepted))
            .then(pl.Series(judge_label.astype(str)))
            .otherwise(pl.col("human_label")),
        )
    return output


def reported_rows() -> pl.DataFrame:
    """Read every reported answer with its routing and final score."""
    return pl.read_parquet(SCORED_RESPONSES_PATH).rename(
        {"model": "model_name", "routing": "scoring_method"}
    )


def model_rows(sample: pl.DataFrame) -> pl.DataFrame:
    """Expand sampled pairs to the model rows that produced them."""
    rows = (
        reported_rows()
        .filter(pl.col("scoring_method") == "human_eval")
        .select("riddle_id", "answer_norm", "model_name")
    )
    return sample.drop("model_name").join(rows, on=["riddle_id", "answer_norm"])


def leaderboard(expanded: pl.DataFrame, judged: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Score-3 accuracy per answering model under EM+human and EM+judge.

    Each pipeline's share correct among a model's human-reviewed rows is a
    weighted ratio estimate from the sample, scaled by that model's full count
    of reviewed rows. Both pipelines use the same sampled rows, so their
    difference isolates the judge.
    """
    flat = reported_rows()
    scored = flat.filter(pl.col("scoring_method").is_in([*AUTO_SCORED, "human_eval"])).with_columns(
        correct=pl.col("score").cast(pl.Float64) == 3,
        human=pl.col("scoring_method") == "human_eval",
    )
    totals = scored.group_by("model_name").agg(
        n_scored=pl.len(),
        n_human=pl.col("human").sum(),
        auto_correct=(pl.col("correct") & ~pl.col("human")).sum(),
        human_correct_full=(pl.col("correct") & pl.col("human")).sum(),
    )

    def rate(score: str, alias: str) -> pl.Expr:
        """Weighted share of a model's sampled reviewed rows scored 3."""
        valid = pl.col(score).is_not_null()
        return (
            (pl.col("pair_weight") * (pl.col(score) == 3)).filter(valid).sum()
            / pl.col("pair_weight").filter(valid).sum()
        ).alias(alias)

    base = expanded.group_by("model_name").agg(rate("human_score", "human_est"))
    for name, frame in judged.items():
        estimate = (
            expanded.select("riddle_id", "answer_norm", "model_name", "pair_weight")
            .join(
                frame.select("riddle_id", "answer_norm", "judge_score"),
                on=["riddle_id", "answer_norm"],
                how="left",
            )
            .group_by("model_name")
            .agg(rate("judge_score", name))
        )
        base = base.join(estimate, on="model_name", how="left")

    def accuracy(share: pl.Expr) -> pl.Expr:
        return (pl.col("auto_correct") + pl.col("n_human") * share) / pl.col("n_scored")

    result = totals.join(base, on="model_name").with_columns(
        [
            accuracy(pl.col("human_correct_full") / pl.col("n_human")).alias("acc_human_full"),
            accuracy(pl.col("human_est")).alias("acc_human"),
        ]
        + [accuracy(pl.col(name)).alias(f"acc|{name}") for name in judged]
    )
    return result.sort("acc_human_full", descending=True)


def false_accept_by_model(expanded: pl.DataFrame, judged: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Share of each answering model's human-rejected rows that a judge accepts."""
    rows = []
    for name, frame in judged.items():
        merged = expanded.join(
            frame.select("riddle_id", "answer_norm", "judge_score"),
            on=["riddle_id", "answer_norm"],
        ).filter(pl.col("human_score") != 3)
        rows.append(
            merged.group_by("model_name").agg(
                rejected_rows=pl.len(),
                false_accept_rate=(pl.col("pair_weight") * (pl.col("judge_score") == 3)).sum()
                / pl.col("pair_weight").sum(),
                condition=pl.lit(name),
            )
        )
    return pl.concat(rows)


def grouped_binary(frame: pl.DataFrame, column: str, name: str, reps: int = 300) -> pl.DataFrame:
    """Binary kappa with bootstrap interval for every level of a grouping column."""
    rows = []
    for (group,), part in frame.group_by(column, maintain_order=True):
        truth = (part["human_score"] == 3).cast(pl.Int8).to_numpy()
        pred = (part["judge_score"] == 3).cast(pl.Int8).to_numpy()
        result = metric_row(
            truth,
            pred,
            part["weight"].to_numpy(),
            bootstrap_weights(part["riddle_id"].to_numpy(), reps),
            2,
        )
        weight = part["weight"].to_numpy()
        rows.append(
            {
                "condition": name,
                column: group,
                "pairs": part.height,
                "human_acc": float((weight * truth).sum() / weight.sum()),
                "judge_acc": float((weight * pred).sum() / weight.sum()),
            }
            | result
        )
    return pl.DataFrame(rows)


def run_costs() -> pl.DataFrame:
    """Provider cost and completion of each setting-B judge run."""
    rows = [
        {
            "condition": f"{JUDGES[run['judge']]}|{run['shots']}",
            "cost_usd": run["provider_cost_usd"],
            "pairs": run["pairs"],
            "prompt_tokens": run["prompt_tokens"],
            "completion_tokens": run["completion_tokens"],
        }
        for run in run_summaries()
        if run["setting"] == SETTING
    ]
    return pl.DataFrame(rows).with_columns(cost_per_call=pl.col("cost_usd") / pl.col("pairs"))


def leaderboard_intervals(
    expanded: pl.DataFrame,
    judged: dict[str, pl.DataFrame],
    board: pl.DataFrame,
    draws: tuple[np.ndarray, np.ndarray],
) -> pl.DataFrame:
    """Bootstrap intervals for each pipeline's accuracy shift and rank agreement.

    Args:
        expanded: Model rows of the sampled pairs with ``pair_weight``.
        judged: Pipeline frames with ``judge_score`` per pair.
        board: Output of ``leaderboard`` with per-model row counts.
        draws: Riddle identifiers and their multiplicity per draw, shape
            (reps, riddles).

    Returns:
        One row per pipeline and answering model: the signed accuracy shift
        from exact match plus human labels, its 95% interval, and the
        pipeline's Kendall tau with its interval. Both pipelines use the same
        rows with a valid verdict, so each shift is paired.
    """
    from scipy.stats import kendalltau

    riddles, multiplicity = draws
    models = board["model_name"].to_list()
    totals = board.select("model_name", "n_scored", "n_human", "auto_correct")
    rows = expanded.join(totals, on="model_name")
    model_index = np.array([models.index(m) for m in rows["model_name"].to_list()])
    onehot = np.zeros((rows.height, len(models)))
    onehot[np.arange(rows.height), model_index] = 1.0
    riddle_index = np.searchsorted(riddles, rows["riddle_id"].to_numpy())
    base_weight = rows["pair_weight"].to_numpy()
    scale = board.select("n_scored", "n_human", "auto_correct").to_numpy().astype(float)

    def accuracy(correct: np.ndarray, valid: np.ndarray, chunk: np.ndarray) -> np.ndarray:
        weight = chunk * (base_weight * valid)
        share = (weight * correct) @ onehot / (weight @ onehot)
        return (scale[:, 2] + scale[:, 1] * share) / scale[:, 0]

    human = (rows["human_score"].cast(pl.Int8) == 3).to_numpy().astype(float)
    output = []
    for name, frame in judged.items():
        scores = rows.select("riddle_id", "answer_norm").join(
            frame.select("riddle_id", "answer_norm", "judge_score"),
            on=["riddle_id", "answer_norm"],
            how="left",
        )["judge_score"]
        valid = scores.is_not_null().to_numpy().astype(float)
        judge = (scores == 3).fill_null(False).to_numpy().astype(float)
        shifts, taus = [], []
        for begin in range(0, multiplicity.shape[0], 100):
            chunk = multiplicity[begin : begin + 100][:, riddle_index].astype(float)
            reference = accuracy(human, valid, chunk)
            shifted = accuracy(judge, valid, chunk)
            shifts.append(shifted - reference)
            taus += [kendalltau(a, b).statistic for a, b in zip(reference, shifted)]
        ones = np.ones((1, rows.height))
        point_ref = accuracy(human, valid, ones)[0]
        point_shift = accuracy(judge, valid, ones)[0] - point_ref
        low, high = np.nanpercentile(np.vstack(shifts), [2.5, 97.5], axis=0)
        tau_low, tau_high = np.nanpercentile(taus, [2.5, 97.5])
        output.append(
            pl.DataFrame(
                {
                    "condition": name,
                    "model_name": models,
                    "shift": point_shift,
                    "low": low,
                    "high": high,
                    "significant": (low > 0) | (high < 0),
                    "kendall_tau": kendalltau(point_ref, point_shift + point_ref).statistic,
                    "kendall_tau_low": tau_low,
                    "kendall_tau_high": tau_high,
                }
            )
        )
    return pl.concat(output)


def effective_size(frame: pl.DataFrame) -> dict[str, float]:
    """Kish effective number of pairs implied by the row weights."""
    weight = frame["weight"].to_numpy()
    effective = weight.sum() ** 2 / (weight**2).sum()
    return {"pairs": frame.height, "effective_pairs": float(effective)}


def main(argv: list[str] | None = None) -> int:
    """Compute and write every setting-B agreement table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=PUBLIC_DIR / "judge_study")
    parser.add_argument("--reps", type=int, default=BOOTSTRAP)
    args = parser.parse_args(argv)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    conditions = [
        Condition(model, shots)
        for model in MODELS
        for shots in SHOTS
        if has_condition(SETTING, shots)
    ]
    names = [condition.name for condition in conditions]
    frame = joined(load_sample(), conditions)
    judged = {name: view(frame, name) for name in names} | ensembles(frame, names)
    by_riddle = frame.group_by("riddle_id").agg(pl.col("community").first()).sort("riddle_id")
    riddles = by_riddle["riddle_id"].to_numpy()
    draws = bootstrap_weights(
        riddles,
        args.reps,
        by_riddle["community"].to_numpy(),
    )
    summary: list[dict[str, object]] = []
    communities, resources = [], []
    for name, data in judged.items():
        key = data.height
        multipliers = draws[:, np.searchsorted(riddles, data["riddle_id"].to_numpy())]
        accepted = data["accepted"] if "accepted" in data.columns else None
        routed = (
            float(
                (data["weight"].to_numpy() * ~accepted.to_numpy()).sum()
                / data["weight"].to_numpy().sum()
            )
            if accepted is not None
            else 0.0
        )
        for row in agreement(data, multipliers):
            summary.append({"condition": name, "routed_to_human": routed, "n": key} | row)
        communities.append(grouped_binary(data, "community", name))
        resources.append(grouped_binary(data, "language_resourcedness", name))
    pl.DataFrame(summary).write_csv(out / "agreement_summary.csv")
    pl.concat(communities).write_csv(out / "agreement_by_community.csv")
    pl.concat(resources).write_csv(out / "agreement_by_resourcedness.csv")
    expanded = model_rows(frame)
    board = leaderboard(expanded, judged)
    leaderboard_intervals(expanded, judged, board, (riddles, draws)).write_csv(
        out / "leaderboard_intervals.csv"
    )
    false_accept_by_model(expanded, judged).sort("condition", "model_name").write_csv(
        out / "false_accept_by_model.csv"
    )
    run_costs().write_csv(out / "run_costs.csv")
    with pl.Config(tbl_rows=60, tbl_width_chars=200, float_precision=3):
        print(
            pl.DataFrame(summary)
            .filter(pl.col("level") != "correct-class precision / recall")
            .select("condition", "level", "kappa", "low", "high", "routed_to_human")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
