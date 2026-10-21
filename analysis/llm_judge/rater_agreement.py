"""Rater agreement on the doubly annotated (IAA) subset.

Run from the repository root:
    uv run python -m analysis.llm_judge.rater_agreement

Raters are the benchmark human label (H1), the independent second human pass
(H2), and the zero-shot DeepSeek and Gemini judges. A "hybrid" rater
keeps the judges' label when both judges give the same score and otherwise falls
back to H1, mirroring the proposed evaluation pipeline; it is only compared with
H2.

Every unit is one unique (riddle, answer) pair. Every interval is a 95%
percentile interval from a riddle-cluster bootstrap. All statistics in a scope
share the same bootstrap draws, so differences between them are paired.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

from analysis.constants import DERIVED_DIR, HUMAN_ANNOTATIONS_PATH
from analysis.llm_judge.judge_data import (
    JUDGES,
    NA_LABELS,
    judge_labels,
    load_pairs,
    load_verdicts,
)

PUBLIC_DIR = DERIVED_DIR / "llm_judge" / "rater_agreement"
HUMANS = ("H1", "H2")
RATERS = (*HUMANS, *JUDGES)
HYBRID = "Hybrid"
ENSEMBLE = "Ensemble"
CATEGORY_CODES = ("0.1", "0.2", "0.3", "0.4", "2.1", "2.2", "2.3", "2.4")
CATEGORY_CODES += ("3.1", "3.2", "3.3")
FINE_LEVELS = ("0.1", "0.2", "0.3", "0.4", "1", "2.1", "2.2", "2.3", "2.4")
FINE_LEVELS += ("3", "3.1", "3.2", "3.3")
NOTE_REQUIRED = ("0.4", "2.4")
POOLED = "Pooled"
# Aggregate tables.
WRITTEN_TABLES = ("agreement", "consensus", "marginals")
BOOTSTRAP = 1000
SEED = 20260928
ALT_TEST_MARGINS = (0.05, 0.10, 0.15)


@dataclass(frozen=True)
class Level:
    """One label granularity and the disagreement weights used at that level."""

    name: str
    column: str
    levels: tuple[str, ...]
    quadratic: bool
    tree: bool = False

    @property
    def delta(self) -> np.ndarray:
        """Disagreement weights: nominal 0/1, squared normalized distance, or tree.

        Tree weights are normalized path lengths in the score -> type taxonomy:
        0 for the same type, 0.5 for two types under one score, 1 otherwise.
        """
        grid = np.arange(len(self.levels))
        if self.tree:
            parent = np.array([label.split(".")[0] for label in self.levels])
            same_parent = parent[:, None] == parent[None, :]
            return np.where(grid[:, None] == grid[None, :], 0.0, 1.0 - 0.5 * same_parent)
        if self.quadratic:
            return ((grid[:, None] - grid[None, :]) / (len(grid) - 1)) ** 2
        return (grid[:, None] != grid[None, :]).astype(float)


LEVELS = (
    Level("binary", "binary", ("0", "1"), quadratic=False),
    Level("score", "score", ("0", "1", "2", "3"), quadratic=True),
    Level("fine", "fine", FINE_LEVELS, quadratic=False),
    Level("fine_tree", "fine", FINE_LEVELS, quadratic=False, tree=True),
)
# Scores whose types are compared by the within-score kappa; score 1 has one type.
TYPED_SCORES = ("0", "2", "3")


def human_fine(score: pl.Expr, category: pl.Expr) -> pl.Expr:
    """Map a human score and category display label to the analysis taxonomy.

    Scores 1 and bare 3 carry N/A and become ``1`` and ``3``. Other labels must
    carry a category code whose leading digit matches the score.
    """
    text = category.fill_null("").str.strip_chars()
    code = text.str.extract(r"^([0-3]\.\d+)\s*-")
    score_text = score.cast(pl.String)
    return (
        pl.when(score.is_in([1, 3]) & text.is_in(NA_LABELS))
        .then(score_text)
        .when(
            score.is_in([0, 2, 3])
            & code.is_in(CATEGORY_CODES)
            & (code.str.slice(0, 1) == score_text)
        )
        .then(code)
        .otherwise(None)
    )


def missing_note(fine: pl.Expr, note: pl.Expr) -> pl.Expr:
    """Flag an "Other" type (0.4, 2.4), which the guidelines say needs a note."""
    empty = note.fill_null("").str.strip_chars() == ""
    return (fine.is_in(NOTE_REQUIRED) & empty).fill_null(False)


def double_annotations(path: Path = HUMAN_ANNOTATIONS_PATH) -> pl.DataFrame:
    """Pair the first and second human labels of every doubly annotated answer.

    Args:
        path: ``human_eval/annotations.parquet``.

    Returns:
        One row per doubly annotated (riddle, answer) pair, with ``_1`` and
        ``_2`` columns for the two passes.
    """
    labels = pl.read_parquet(path).filter(pl.col("iaa_row").is_not_null())
    keys = ["riddle_id", "answer_norm"]
    fields = ["annotator_id", "score", "error_type_label", "note"]

    def annotation(name: str, suffix: str) -> pl.DataFrame:
        return labels.filter(pl.col("annotation") == name).select(
            *keys, *[pl.col(f).alias(f"{f}_{suffix}") for f in fields]
        )

    return (
        labels.filter(pl.col("annotation") == "first")
        .select(*keys, "community", "topic", "answer", "self_review", "iaa_row")
        .join(annotation("first", "1"), on=keys)
        .join(annotation("second", "2"), on=keys)
        .sort("iaa_row")
    )


def human_labels(
    path: Path = HUMAN_ANNOTATIONS_PATH,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load both human passes and drop self-reviews and invalid labels.

    A label is invalid when its type does not match its score, or when it uses
    an "Other" type without the validator note that the guidelines require.

    Args:
        path: ``human_eval/annotations.parquet``.

    Returns:
        Retained pairs with ``score_H1``, ``fine_H1``, ``score_H2`` and
        ``fine_H2``, and per-row exclusion flags (``self_review``,
        ``mismatched_type``, ``missing_note``, ``invalid_label``) for the audit.
    """

    def person(column: str) -> pl.Expr:
        return (
            pl.col(column).fill_null("").str.strip_chars().str.replace_all(r"\s+", " ")
        ).str.to_lowercase()

    parsed = double_annotations(path).with_columns(
        placeholder=pl.col("answer").fill_null("").str.contains(r"^<.*>$"),
        score_H1=pl.col("score_1"),
        score_H2=pl.col("score_2"),
        self_review=pl.col("self_review") | (person("annotator_id_1") == person("annotator_id_2")),
    )
    parsed = (
        parsed.with_columns(
            fine_H1=human_fine(pl.col("score_H1"), pl.col("error_type_label_1")),
            fine_H2=human_fine(pl.col("score_H2"), pl.col("error_type_label_2")),
        )
        .with_columns(
            mismatched_type=pl.any_horizontal(
                pl.col("score_H1", "score_H2", "fine_H1", "fine_H2").is_null()
            ),
            missing_note=missing_note(pl.col("fine_H1"), pl.col("note_1"))
            | missing_note(pl.col("fine_H2"), pl.col("note_2")),
        )
        .with_columns(invalid_label=pl.col("mismatched_type") | pl.col("missing_note"))
    )
    retained = parsed.filter(~pl.col("self_review") & ~pl.col("invalid_label"))
    return retained, parsed


def iaa_h1_labels(path: Path = HUMAN_ANNOTATIONS_PATH) -> pl.DataFrame:
    """Return the valid benchmark (H1) labels recorded in the IAA export.

    A few IAA answers carry H1 labels that the flat export lists as pending.
    Accuracy uses these labels as human labels, and the judge-only setting
    excludes them. Only pairs that pass the IAA cleaning are returned.

    Args:
        path: ``human_eval/annotations.parquet``.

    Returns:
        One row per (riddle, normalized answer) with ``score`` (float),
        ``error_type`` (null for bare 1 and 3), and ``annotator_id``.
    """
    retained, _ = human_labels(path)
    return (
        retained.filter(~pl.col("placeholder"))
        .select(
            "riddle_id",
            "answer_norm",
            score=pl.col("score_H1").cast(pl.Float64),
            error_type=pl.when(pl.col("fine_H1").str.contains(r"\."))
            .then("fine_H1")
            .otherwise(None),
            annotator_id=pl.col("annotator_id_1"),
        )
        .unique(["riddle_id", "answer_norm"], keep="first")
    )


def with_ensemble(frame: pl.DataFrame) -> pl.DataFrame:
    """Add the two-judge ensemble, which uses no human label.

    It takes the lower of the two judges' scores (and that judge's type;
    DeepSeek's on a tie), so an answer is correct only when both judges accept it.
    """
    lower = pl.col("score_DeepSeek") <= pl.col("score_Gemini")
    return frame.with_columns(
        score_Ensemble=pl.min_horizontal("score_DeepSeek", "score_Gemini"),
        fine_Ensemble=pl.when(lower).then("fine_DeepSeek").otherwise("fine_Gemini"),
    )


def with_hybrid(frame: pl.DataFrame) -> pl.DataFrame:
    """Add the judges-agree-else-H1 rater, its routing flag, and the ensemble."""
    routed = pl.col("score_DeepSeek") != pl.col("score_Gemini")
    return with_ensemble(frame).with_columns(
        routed=routed,
        score_Hybrid=pl.when(routed).then("score_H1").otherwise("score_Gemini"),
        fine_Hybrid=pl.when(routed).then("fine_H1").otherwise("fine_Gemini"),
    )


def four_rater_frame(humans: pl.DataFrame, shots: str = "zero") -> pl.DataFrame:
    """Join human passes with both judges; keep pairs every rater labeled."""
    exposure = load_pairs("setting_a").select("riddle_id", "answer_norm", "tuning_exposed")
    frame = humans.join(exposure, on=["riddle_id", "answer_norm"], how="inner")
    for judge in JUDGES:
        frame = frame.join(
            judge_labels("setting_a", judge, shots=shots),
            on=["riddle_id", "answer_norm"],
            how="inner",
        )
    return with_hybrid(frame)


def codes(frame: pl.DataFrame, rater: str, level: Level) -> np.ndarray:
    """Integer-code one rater's labels at one level."""
    if level.column == "binary":
        return (frame[f"score_{rater}"] == 3).cast(pl.Int64).to_numpy()
    lookup = {label: index for index, label in enumerate(level.levels)}
    values = frame[f"{level.column}_{rater}"].cast(pl.String).to_list()
    unknown = sorted({value for value in values if value not in lookup})
    if unknown:
        raise ValueError(f"{rater} has labels outside the {level.name} taxonomy: {unknown}")
    return np.fromiter((lookup[value] for value in values), dtype=np.int64)


def cluster_sum(cluster: np.ndarray, n_clusters: int, values: np.ndarray) -> np.ndarray:
    """Sum unit-level count vectors (units x d) within riddle clusters."""
    output = np.zeros((n_clusters, values.shape[1]))
    np.add.at(output, cluster, values)
    return output


def confusion_counts(first: np.ndarray, second: np.ndarray, size: int) -> np.ndarray:
    """One-hot unit contributions to a flattened size-by-size confusion matrix."""
    output = np.zeros((len(first), size * size))
    output[np.arange(len(first)), first * size + second] = 1.0
    return output


def coincidence_counts(labels: np.ndarray, size: int) -> np.ndarray:
    """Unit contributions to Krippendorff's coincidence matrix (complete data).

    Args:
        labels: Units x raters integer codes, with no missing values.
        size: Number of label levels.

    Returns:
        Units x size**2 array of pairable-value counts divided by raters - 1.
    """
    counts = np.stack([(labels == level).sum(1) for level in range(size)], 1)
    pairs = counts[:, :, None] * counts[:, None, :] - np.einsum("ui,ij->uij", counts, np.eye(size))
    return (pairs / (labels.shape[1] - 1)).reshape(len(labels), -1)


def cohen_kappa(totals: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Weighted Cohen's kappa over any leading batch dimensions."""
    size = delta.shape[0]
    matrix = totals.reshape(*totals.shape[:-1], size, size)
    total = matrix.sum((-2, -1))[..., None, None]
    expected = matrix.sum(-1)[..., :, None] * matrix.sum(-2)[..., None, :] / total
    denominator = (expected * delta).sum((-2, -1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, 1 - (matrix * delta).sum((-2, -1)) / denominator, np.nan)


def krippendorff_alpha(totals: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Krippendorff's alpha from coincidence totals over any batch dimensions."""
    size = delta.shape[0]
    matrix = totals.reshape(*totals.shape[:-1], size, size)
    marginal = matrix.sum(-1)
    n = marginal.sum(-1)
    observed = (matrix * delta).sum((-2, -1))
    expected = np.einsum("...i,ij,...j->...", marginal, delta, marginal)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(expected > 0, 1 - (n - 1) * observed / expected, np.nan)


def within_score_kappa(totals: np.ndarray) -> np.ndarray:
    """Kappa on fine types among pairs that share a typed score.

    Observed and chance agreement are pooled over the score blocks of the fine
    confusion matrix, with chance computed within each score, so score
    disagreements (off-block cells) and score 1 (a single type) do not count.
    """
    size = len(FINE_LEVELS)
    matrix = totals.reshape(*totals.shape[:-1], size, size)
    parent = np.array([label.split(".")[0] for label in FINE_LEVELS])
    observed = chance = units = 0.0
    for score in TYPED_SCORES:
        index = np.flatnonzero(parent == score)
        block = matrix[..., index[:, None], index[None, :]]
        n = block.sum((-2, -1))
        observed = observed + np.trace(block, axis1=-2, axis2=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            chance = chance + np.where(
                n > 0,
                np.einsum("...i,...i->...", block.sum(-1), block.sum(-2)) / n,
                0.0,
            )
        units = units + n
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray((observed - chance) / (units - chance))


def observed_agreement(totals: np.ndarray, size: int) -> np.ndarray:
    """Share of units on the confusion-matrix diagonal."""
    matrix = totals.reshape(*totals.shape[:-1], size, size)
    return np.trace(matrix, axis1=-2, axis2=-1) / matrix.sum((-2, -1))


def interval(point: float, reps: np.ndarray) -> dict[str, float]:
    """Point estimate with a 95% percentile bootstrap interval."""
    finite = reps[np.isfinite(reps)]
    low, high = np.percentile(finite, [2.5, 97.5]) if finite.size else (np.nan,) * 2
    return {"estimate": float(point), "low": float(low), "high": float(high)}


@dataclass
class Bootstrap:
    """Riddle clusters and shared multinomial resampling draws for one scope."""

    cluster: np.ndarray
    draws: np.ndarray

    @classmethod
    def for_frame(
        cls, frame: pl.DataFrame, reps: int, seed: int, stratum: str | None = None
    ) -> Bootstrap:
        """Draw cluster multiplicities once so every statistic is paired.

        Args:
            frame: Units with a ``riddle_id`` column.
            reps: Number of bootstrap draws.
            seed: Random seed.
            stratum: Optional column of the sampling stratum (for example
                ``community``); riddles are then resampled within each stratum.
        """
        _, cluster = np.unique(frame["riddle_id"].to_numpy(), return_inverse=True)
        n_clusters = int(cluster.max()) + 1
        rng = np.random.default_rng(seed)
        if stratum is None:
            draws = rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters), reps)
            return cls(cluster, draws.astype(float))
        cluster_stratum = np.empty(n_clusters, dtype=object)
        cluster_stratum[cluster] = frame[stratum].to_numpy()
        draws = np.zeros((reps, n_clusters))
        for value in np.unique(cluster_stratum):
            members = np.flatnonzero(cluster_stratum == value)
            draws[:, members] = rng.multinomial(
                len(members), np.full(len(members), 1 / len(members)), reps
            )
        return cls(cluster, draws)

    def totals(self, unit_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Full-sample totals and bootstrap totals, shape (reps, d)."""
        per_cluster = cluster_sum(self.cluster, self.draws.shape[1], unit_counts)
        return per_cluster.sum(0), self.draws @ per_cluster

    def stat(
        self, unit_counts: np.ndarray, statistic: Callable[[np.ndarray], np.ndarray]
    ) -> tuple[float, np.ndarray]:
        """Evaluate a statistic of summed counts on the sample and every draw."""
        point, reps = self.totals(unit_counts)
        with np.errstate(invalid="ignore", divide="ignore"):
            return float(statistic(point)), statistic(reps)


def ratio(numerator: int, denominator: int) -> Callable[[np.ndarray], np.ndarray]:
    """Statistic that divides two columns of an indicator-count total."""

    def compute(totals: np.ndarray) -> np.ndarray:
        return totals[..., numerator] / totals[..., denominator]

    return compute


KappaDraws = dict[tuple[str, str, str], tuple[float, np.ndarray]]


def pair_rows(
    frame: pl.DataFrame, boot: Bootstrap, pairs: Sequence[tuple[str, str]]
) -> tuple[list[dict[str, object]], KappaDraws]:
    """Cohen's kappa and observed agreement for rater pairs at every level.

    A ``weight`` column, when present, weights each unit (for example by the
    number of model responses it stands for). The fine level also reports the
    within-score kappa on its confusion matrix.

    Returns:
        Output rows, and each kappa's point estimate and bootstrap draws keyed by
        (level, rater, rater) for paired comparisons.
    """
    rows: list[dict[str, object]] = []
    draws: KappaDraws = {}
    for level in LEVELS:
        size = len(level.levels)
        for first, second in pairs:
            counts = confusion_counts(codes(frame, first, level), codes(frame, second, level), size)
            if "weight" in frame.columns:
                counts = counts * frame["weight"].to_numpy()[:, None]
            point, reps = boot.stat(counts, partial(cohen_kappa, delta=level.delta))
            draws[(level.name, first, second)] = (point, reps)
            agree, agree_reps = boot.stat(counts, partial(observed_agreement, size=size))
            base = {"level": level.name, "raters": f"{first}~{second}"}
            rows.append(base | {"statistic": "cohen_kappa"} | interval(point, reps))
            rows.append(base | {"statistic": "observed_agreement"} | interval(agree, agree_reps))
            if level.name == "fine":
                within, within_reps = boot.stat(counts, within_score_kappa)
                rows.append(
                    base | {"statistic": "within_score_kappa"} | interval(within, within_reps)
                )
    return rows, draws


def alpha_rows(frame: pl.DataFrame, boot: Bootstrap) -> list[dict[str, object]]:
    """Krippendorff's alpha for humans only, judges only, and all four raters."""
    groups = {"H1+H2": HUMANS, "DeepSeek+Gemini": tuple(JUDGES), "all four": RATERS}
    rows: list[dict[str, object]] = []
    for level in LEVELS:
        for name, raters in groups.items():
            labels = np.stack([codes(frame, rater, level) for rater in raters], 1)
            counts = coincidence_counts(labels, len(level.levels))
            point, reps = boot.stat(counts, partial(krippendorff_alpha, delta=level.delta))
            rows.append(
                {"level": level.name, "raters": name, "statistic": "krippendorff_alpha"}
                | interval(point, reps)
            )
    return rows


def alt_test_rows(draws: KappaDraws) -> list[dict[str, object]]:
    """Paired test of whether a judge can replace one human annotator.

    For held-out human ``h`` and remaining human ``r``, the difference is
    kappa(judge, r) - kappa(h, r). The judge is non-inferior at margin ``e``
    when the lower 95% bound (one-sided 97.5%) exceeds ``-e``.
    """
    rows: list[dict[str, object]] = []
    candidates = [(judge, judge) for judge in JUDGES] + [(HYBRID, HYBRID)]
    for level in LEVELS:
        for label, rater in candidates:
            for held_out, reference in (("H1", "H2"), ("H2", "H1")):
                if rater == HYBRID and reference != "H2":
                    continue
                human_key = (level.name, "H1", "H2")
                first, second = ordered(rater, reference)
                rater_point, rater_reps = draws[(level.name, first, second)]
                human_point, human_reps = draws[human_key]
                delta = rater_reps - human_reps
                result = interval(rater_point - human_point, delta)
                rows.append(
                    {
                        "level": level.name,
                        "candidate": label,
                        "replaces": held_out,
                        "reference": reference,
                        "kappa_candidate": rater_point,
                        "kappa_human": human_point,
                    }
                    | {f"delta_{key}": value for key, value in result.items()}
                    | {
                        f"noninferior_{margin:.2f}": bool(result["low"] > -margin)
                        for margin in ALT_TEST_MARGINS
                    }
                    | {"share_draws_judge_better": float(np.nanmean(delta > 0))}
                )
    return rows


def ensemble_rows(draws: KappaDraws, humans: Sequence[str] = HUMANS) -> list[dict[str, object]]:
    """Paired kappa gain of the two-judge ensemble over each single judge.

    The difference kappa(Ensemble, h) - kappa(judge, h) uses the same bootstrap
    draws for both terms, for each human ``h``.
    """
    rows: list[dict[str, object]] = []
    for level in LEVELS:
        for human in humans:
            ensemble_point, ensemble_reps = draws[(level.name, human, ENSEMBLE)]
            for judge in JUDGES:
                point, reps = draws[(level.name, human, judge)]
                rows.append(
                    {
                        "level": level.name,
                        "raters": f"{human}~{ENSEMBLE} - {human}~{judge}",
                        "statistic": "kappa_difference",
                    }
                    | interval(ensemble_point - point, ensemble_reps - reps)
                )
    return rows


def ordered(first: str, second: str) -> tuple[str, str]:
    """Return a rater pair in the stable order used for pair keys."""
    order = (*RATERS, HYBRID, ENSEMBLE)
    return (first, second) if order.index(first) < order.index(second) else (second, first)


def consensus_rows(frame: pl.DataFrame, boot: Bootstrap) -> list[dict[str, object]]:
    """Judge behavior on items where the two humans agree versus disagree.

    Reports, at each level, how often each judge matches the human consensus,
    how often it disagrees with H1 when humans agree versus when they disagree,
    whom it sides with on human disagreements, and whether judge-judge
    disagreement flags human disagreement.
    """
    rows: list[dict[str, object]] = []
    for level in LEVELS:
        h1, h2 = codes(frame, "H1", level), codes(frame, "H2", level)
        ds, gm = codes(frame, "DeepSeek", level), codes(frame, "Gemini", level)
        humans_agree = h1 == h2
        judges_agree = ds == gm
        base = {"level": level.name}
        for judge, value in (("DeepSeek", ds), ("Gemini", gm)):
            indicators = np.stack(
                [
                    humans_agree,
                    humans_agree & (value == h1),
                    ~humans_agree,
                    ~humans_agree & (value == h1),
                    ~humans_agree & (value == h2),
                    humans_agree & (value != h1),
                    ~humans_agree & (value != h1),
                ],
                1,
            ).astype(float)
            stats = {
                "accuracy_on_human_consensus": ratio(1, 0),
                "sides_with_H1_when_humans_split": ratio(3, 2),
                "sides_with_H2_when_humans_split": ratio(4, 2),
                "disagrees_H1_when_humans_agree": ratio(5, 0),
                "disagrees_H1_when_humans_split": ratio(6, 2),
            }
            for name, statistic in stats.items():
                point, reps = boot.stat(indicators, statistic)
                rows.append(base | {"judge": judge, "statistic": name} | interval(point, reps))

            def lift(t: np.ndarray) -> np.ndarray:
                return (t[..., 6] / t[..., 2]) / (t[..., 5] / t[..., 0])

            point, reps = boot.stat(indicators, lift)
            rows.append(
                base
                | {
                    "judge": judge,
                    "statistic": "disagreement_risk_ratio_split_vs_agree",
                }
                | interval(point, reps)
            )
        flags = np.stack(
            [
                ~judges_agree,
                ~judges_agree & ~humans_agree,
                judges_agree,
                judges_agree & ~humans_agree,
                ~humans_agree,
                ~humans_agree & ~judges_agree,
            ],
            1,
        ).astype(float)
        stats = {
            "humans_split_when_judges_split": ratio(1, 0),
            "humans_split_when_judges_agree": ratio(3, 2),
            "judges_split_recall_of_human_splits": ratio(5, 4),
            "judges_split_rate": lambda t: t[..., 0] / (t[..., 0] + t[..., 2]),
        }
        for name, statistic in stats.items():
            point, reps = boot.stat(flags, statistic)
            rows.append(base | {"judge": "both", "statistic": name} | interval(point, reps))
        counts = {"items": len(h1), "human_splits": int((~humans_agree).sum())}
        rows.extend(
            base | {"judge": "both", "statistic": key, "estimate": float(value)}
            for key, value in counts.items()
        )
    return rows


def marginal_rows(
    frame: pl.DataFrame, boot: Bootstrap, raters: Sequence[str]
) -> list[dict[str, object]]:
    """Share of pairs each rater scores as correct (score 3), and mean score."""
    rows: list[dict[str, object]] = []
    for rater in raters:
        score = frame[f"score_{rater}"].cast(pl.Float64).to_numpy()
        units = np.stack([score == 3, score, np.ones_like(score)], 1).astype(float)
        if "weight" in frame.columns:
            units = units * frame["weight"].to_numpy()[:, None]
        for name, statistic in (
            ("correct_rate", ratio(0, 2)),
            ("mean_score", ratio(1, 2)),
        ):
            point, reps = boot.stat(units, statistic)
            rows.append({"rater": rater, "statistic": name} | interval(point, reps))
    return rows


def scopes(frame: pl.DataFrame) -> list[tuple[str, pl.DataFrame]]:
    """The pooled sample followed by one scope per community."""
    return [(POOLED, frame)] + [
        (str(name), part)
        for (name,), part in frame.sort("community").group_by("community", maintain_order=True)
    ]


def tag(rows: list[dict[str, object]], **fields: object) -> list[dict[str, object]]:
    """Prefix every row with scope and sample identifiers."""
    return [fields | row for row in rows]


def four_rater_tables(
    frame: pl.DataFrame, sample: str, reps: int
) -> dict[str, list[dict[str, object]]]:
    """Compute every four-rater table for the pooled and per-community scopes."""
    pairs = [
        ordered(*pair)
        for pair in (
            *combinations(RATERS, 2),
            (HYBRID, "H2"),
            *((human, ENSEMBLE) for human in HUMANS),
        )
    ]
    output: dict[str, list[dict[str, object]]] = {
        "agreement": [],
        "alt_test": [],
        "consensus": [],
        "ensemble": [],
        "marginals": [],
    }
    for index, (scope, part) in enumerate(scopes(frame)):
        boot = Bootstrap.for_frame(part, reps, SEED + index)
        ids = {
            "sample": sample,
            "scope": scope,
            "units": part.height,
            "riddles": part["riddle_id"].n_unique(),
        }
        rows, draws = pair_rows(part, boot, pairs)
        output["agreement"] += tag(rows + alpha_rows(part, boot), **ids)
        output["alt_test"] += tag(alt_test_rows(draws), **ids)
        output["consensus"] += tag(consensus_rows(part, boot), **ids)
        output["ensemble"] += tag(ensemble_rows(draws), **ids)
        output["marginals"] += tag(
            marginal_rows(part, boot, (*RATERS, HYBRID, ENSEMBLE))
            + [
                {
                    "rater": HYBRID,
                    "statistic": "routed_to_human",
                    "estimate": int(part["routed"].sum()) / part.height,
                }
            ],
            **ids,
        )
    return output


def run_validity(judge: str) -> dict[str, object]:
    """Share of setting-A pairs whose first or every attempt failed validation."""
    verdicts = load_verdicts("setting_a", judge)
    return {
        "judge": judge,
        "pairs": verdicts.height,
        "invalid_attempts": int(verdicts["invalid_attempts"].sum()),
        "first_attempt_invalid": int((~verdicts["first_attempt_valid"]).sum()) / verdicts.height,
        "no_valid_verdict": int(verdicts["score"].is_null().sum()),
    }


def condition_rows(
    frame: pl.DataFrame,
    other: pl.DataFrame,
    names: tuple[str, str],
    reps: int,
    seed: int,
) -> list[dict[str, object]]:
    """Compare each judge under two conditions against both humans.

    Args:
        frame: Sample that has every rater under the reference condition.
        other: The same pairs under the alternative condition.
        names: Reference and alternative condition names, such as
            ``("zero", "few10")``.
        reps: Bootstrap draws.
        seed: Bootstrap seed.

    Returns:
        Kappa rows, including paired (reference - alternative) differences on the
        pairs both conditions labeled.
    """
    reference, suffix = names
    renamed = other.select(
        "riddle_id",
        "answer_norm",
        *[
            pl.col(f"{kind}_{judge}").alias(f"{kind}_{judge}-{suffix}")
            for judge in JUDGES
            for kind in ("score", "fine")
        ],
    )
    shared = frame.join(renamed, on=["riddle_id", "answer_norm"])
    boot = Bootstrap.for_frame(shared, reps, seed)
    pairs = [
        (first, second)
        for judge in JUDGES
        for first, second in (
            *((human, judge) for human in HUMANS),
            *((human, f"{judge}-{suffix}") for human in HUMANS),
            (judge, f"{judge}-{suffix}"),
        )
    ]
    rows, draws = pair_rows(shared, boot, pairs)
    for level in LEVELS:
        for judge in JUDGES:
            for human in HUMANS:
                new_point, new_reps = draws[(level.name, human, judge)]
                old_point, old_reps = draws[(level.name, human, f"{judge}-{suffix}")]
                rows.append(
                    {
                        "level": level.name,
                        "raters": f"{human}~{judge}: {reference} - {suffix}",
                        "statistic": "kappa_difference",
                    }
                    | interval(new_point - old_point, new_reps - old_reps)
                )
    ids = {"units": shared.height, "riddles": shared["riddle_id"].n_unique()}
    return tag(rows, **ids)


def sample_audit(parsed: pl.DataFrame, frame: pl.DataFrame) -> list[dict[str, object]]:
    """Count raw, excluded, and retained pairs at each filtering step."""
    retained = parsed.filter(~pl.col("self_review") & ~pl.col("invalid_label"))
    steps = [
        ("raw IAA pairs", parsed),
        ("excluded: self-review", parsed.filter(pl.col("self_review"))),
        (
            "excluded: type does not match score",
            parsed.filter(~pl.col("self_review") & pl.col("mismatched_type")),
        ),
        (
            "excluded: Other type (0.4, 2.4) without note",
            parsed.filter(
                ~pl.col("self_review") & ~pl.col("mismatched_type") & pl.col("missing_note")
            ),
        ),
        ("cleaned human pairs", retained),
        (
            "excluded: placeholder answer (auto-scored)",
            retained.filter(pl.col("placeholder")),
        ),
        (
            "excluded: missing judge verdict",
            retained.filter(~pl.col("placeholder")).join(
                frame, on=["riddle_id", "answer_norm"], how="anti"
            ),
        ),
        ("four-rater pairs (all judges returned a verdict)", frame),
        (
            "excluded: riddle used in judge prompt tuning",
            frame.filter(pl.col("tuning_exposed")),
        ),
        ("primary four-rater sample", frame.filter(~pl.col("tuning_exposed"))),
    ]
    return [
        {
            "step": name,
            "pairs": part.height,
            "riddles": part["riddle_id"].n_unique(),
            "communities": part["community"].n_unique(),
        }
        for name, part in steps
    ]


def write(rows: list[dict[str, object]], path: Path) -> None:
    """Write aggregate rows as CSV."""
    pl.DataFrame(rows, infer_schema_length=None).write_csv(path)


def main(argv: list[str] | None = None) -> int:
    """Compute and write every rater-agreement table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=PUBLIC_DIR)
    parser.add_argument("--reps", type=int, default=BOOTSTRAP)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    humans, parsed = human_labels()
    frame = four_rater_frame(humans)
    primary = frame.filter(~pl.col("tuning_exposed"))
    tables = four_rater_tables(primary, "primary", args.reps)
    sensitivity = four_rater_tables(frame, "incl. tuning-exposed riddles", args.reps)
    for name in WRITTEN_TABLES:
        rows = tables[name]
        extra = [row for row in sensitivity[name] if row["scope"] == POOLED]
        write(rows + extra, args.out_dir / f"{name}.csv")
    few_shot = four_rater_frame(humans, shots="few10")
    few_shot_primary = few_shot.join(
        primary.select("riddle_id", "answer_norm"), on=["riddle_id", "answer_norm"]
    )
    few_shot_rates = [
        row | {"rater": f"{row['rater']}-few10"}
        for row in marginal_rows(
            few_shot_primary,
            Bootstrap.for_frame(few_shot_primary, args.reps, SEED + 801),
            list(JUDGES),
        )
    ]
    write(
        condition_rows(primary, few_shot, ("zero", "few10"), args.reps, SEED + 800)
        + few_shot_rates,
        args.out_dir / "few_shot.csv",
    )
    write([run_validity(judge) for judge in JUDGES], args.out_dir / "validity.csv")
    write(sample_audit(parsed, frame), args.out_dir / "sample_audit.csv")
    pooled = [
        row
        for row in tables["agreement"]
        if row["scope"] == POOLED and row["statistic"] in ("cohen_kappa", "krippendorff_alpha")
    ]
    with pl.Config(tbl_rows=40, float_precision=3, tbl_width_chars=160):
        print(
            pl.DataFrame(pooled).select("level", "raters", "statistic", "estimate", "low", "high")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
