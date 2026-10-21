"""Tests for the four-rater agreement estimators."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import polars as pl
import pytest

from analysis.llm_judge.judge_agreement import kappa_from
from analysis.llm_judge.rater_agreement import (
    FINE_LEVELS,
    LEVELS,
    Bootstrap,
    cohen_kappa,
    coincidence_counts,
    confusion_counts,
    human_fine,
    krippendorff_alpha,
    missing_note,
    with_ensemble,
    within_score_kappa,
)


def textbook_alpha(labels: np.ndarray, delta: np.ndarray) -> float:
    """Krippendorff's alpha from explicit pairable values (complete data)."""
    values = labels.ravel()
    n = values.size
    within = sum(
        delta[a, b] / (labels.shape[1] - 1) for unit in labels for a, b in permutations(unit, 2)
    )
    between = sum(delta[a, b] for a in values for b in values)
    return 1 - (n - 1) * within / between


@pytest.mark.parametrize("level", LEVELS[:2], ids=lambda level: level.name)
def test_cohen_kappa_matches_reference(level) -> None:
    # Arrange
    rng = np.random.default_rng(0)
    size = len(level.levels)
    first = rng.integers(0, size, 300)
    second = np.where(rng.random(300) < 0.7, first, rng.integers(0, size, 300))
    totals = confusion_counts(first, second, size).sum(0)

    # Act
    result = cohen_kappa(totals, level.delta)

    # Assert
    expected = kappa_from(totals.reshape(size, size), quadratic=level.quadratic)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("raters", [2, 4])
@pytest.mark.parametrize("level", LEVELS[:2], ids=lambda level: level.name)
def test_alpha_matches_textbook(level, raters: int) -> None:
    # Arrange
    rng = np.random.default_rng(raters)
    size = len(level.levels)
    labels = rng.integers(0, size, (40, raters))

    # Act
    result = krippendorff_alpha(coincidence_counts(labels, size).sum(0), level.delta)

    # Assert
    assert result == pytest.approx(textbook_alpha(labels, level.delta))


def test_perfect_agreement_scores_one() -> None:
    # Arrange
    labels = np.repeat(np.array([[0], [1], [2], [3]]), 4, axis=1)
    delta = LEVELS[1].delta

    # Act
    alpha = krippendorff_alpha(coincidence_counts(labels, 4).sum(0), delta)
    kappa = cohen_kappa(confusion_counts(labels[:, 0], labels[:, 1], 4).sum(0), delta)

    # Assert
    assert alpha == pytest.approx(1.0)
    assert kappa == pytest.approx(1.0)


def test_human_fine_rejects_mismatched_categories() -> None:
    # Arrange
    frame = pl.DataFrame(
        {
            "score": [0, 1, 1, 2, 3, 3, 0],
            "category": [
                "0.2 - Wrong entity",
                "N/A",
                "0.1 - Wrong",
                "2.3 - Too broad",
                "N/A",
                "3.1 - Alternative",
                "2.1 - Partial",
            ],
        },
        schema_overrides={"score": pl.Int8},
    )

    # Act
    result = frame.select(human_fine(pl.col("score"), pl.col("category")))

    # Assert
    assert result.to_series().to_list() == ["0.2", "1", None, "2.3", "3", "3.1", None]


def test_bootstrap_point_uses_full_sample() -> None:
    # Arrange
    frame = pl.DataFrame({"riddle_id": ["a", "a", "b", "c"]})
    boot = Bootstrap.for_frame(frame, reps=50, seed=1)
    units = np.ones((4, 1))

    # Act
    point, reps = boot.stat(units, lambda totals: totals[..., 0])

    # Assert
    assert point == 4
    assert reps.shape == (50,)
    assert reps.mean() == pytest.approx(4, rel=0.2)


def test_stratified_bootstrap_keeps_stratum_sizes() -> None:
    # Arrange
    frame = pl.DataFrame(
        {
            "riddle_id": ["a1", "a2", "a3", "b1", "b2"],
            "community": ["a", "a", "a", "b", "b"],
        }
    )

    # Act
    boot = Bootstrap.for_frame(frame, reps=200, seed=3, stratum="community")

    # Assert
    in_a = boot.draws[:, :3].sum(axis=1)
    in_b = boot.draws[:, 3:].sum(axis=1)
    assert (in_a == 3).all()
    assert (in_b == 2).all()


def test_other_type_without_note_is_flagged() -> None:
    # Arrange
    frame = pl.DataFrame(
        {"fine": ["0.4", "2.4", "0.4", "0.3", None], "note": ["x", " ", None, "", ""]}
    )

    # Act
    flags = frame.select(missing_note(pl.col("fine"), pl.col("note")))

    # Assert
    assert flags.to_series().to_list() == [False, True, True, False, False]


def test_ensemble_takes_lower_score_and_its_type() -> None:
    # Arrange
    frame = pl.DataFrame(
        {
            "score_DeepSeek": [3, 0, 2],
            "fine_DeepSeek": ["3", "0.2", "2.2"],
            "score_Gemini": [3, 3, 0],
            "fine_Gemini": ["3.1", "3", "0.3"],
        },
        schema_overrides={"score_DeepSeek": pl.Int8, "score_Gemini": pl.Int8},
    )

    # Act
    result = with_ensemble(frame)

    # Assert
    assert result["score_Ensemble"].to_list() == [3, 0, 0]
    assert result["fine_Ensemble"].to_list() == ["3", "0.2", "0.3"]


def test_tree_delta_follows_taxonomy_path_length() -> None:
    # Arrange
    delta = next(level for level in LEVELS if level.tree).delta
    index = {label: position for position, label in enumerate(FINE_LEVELS)}

    # Act
    same, siblings, cousins = (
        delta[index[a], index[b]] for a, b in (("0.1", "0.1"), ("0.1", "0.3"), ("0.1", "2.1"))
    )

    # Assert
    assert (same, siblings, cousins) == (0.0, 0.5, 1.0)


def test_within_score_kappa_ignores_score_disagreement() -> None:
    # Arrange
    index = {label: position for position, label in enumerate(FINE_LEVELS)}
    size = len(FINE_LEVELS)
    pairs = [("0.1", "0.1"), ("0.2", "0.2"), ("0.1", "0.2"), ("3", "3.1")]
    within = np.zeros((size, size))
    for first, second in pairs:
        within[index[first], index[second]] += 1
    noisy = within.copy()
    noisy[index["0.3"], index["3.1"]] += 5
    noisy[index["1"], index["1"]] += 5

    # Act
    clean, with_noise = (
        within_score_kappa(within.ravel()),
        within_score_kappa(noisy.ravel()),
    )

    # Assert
    # Score 0: rows (2, 1), columns (1, 2), n 3; score 3 block has no chance term.
    observed, chance = 2, (2 * 1 + 1 * 2) / 3
    assert clean == pytest.approx((observed - chance) / (4 - chance))
    assert with_noise == pytest.approx(clean)
