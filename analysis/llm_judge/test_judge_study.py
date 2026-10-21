"""Tests for the judge-agreement study sampler, metrics, and ensembles."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from riddles_framework.judging.llm_judge_run import JudgeRunError, select_examples
from sklearn.metrics import cohen_kappa_score

from analysis.llm_judge.judge_agreement import confusion, ensembles, kappa_from


@pytest.mark.parametrize("quadratic", [False, True])
def test_weighted_kappa_matches_sklearn(quadratic: bool) -> None:
    rng = np.random.default_rng(0)
    truth, pred = rng.integers(0, 4, 400), rng.integers(0, 4, 400)
    pred[:250] = truth[:250]
    weight = rng.uniform(0.5, 3.0, 400)

    ours = kappa_from(confusion(truth, pred, weight, 4), quadratic)

    expected = cohen_kappa_score(
        truth,
        pred,
        weights="quadratic" if quadratic else None,
        sample_weight=weight,
    )
    assert ours == pytest.approx(expected, abs=1e-9)


def test_fixed_examples_are_identical_and_exclude_target_riddle() -> None:
    pool = [{"riddle_id": f"r{i}", "community": "c"} for i in range(5)]
    first = {"riddle_id": "x", "community": "c"}
    second = {"riddle_id": "r0", "community": "c"}

    def pick(row: dict[str, str], size: int) -> list[dict[str, str]]:
        return select_examples(
            row,
            calibration=pool,
            evaluation_riddles=set(),
            size=size,
            strategy="fixed",
            seed=1,
        )

    chosen_first, chosen_second = pick(first, 3), pick(second, 3)

    assert [r["riddle_id"] for r in chosen_first] == ["r0", "r1", "r2"]
    assert [r["riddle_id"] for r in chosen_second] == ["r1", "r2", "r3"]
    with pytest.raises(JudgeRunError):
        pick(first, 6)


def test_agree_or_human_routes_disagreements_to_human_label() -> None:
    frame = pl.DataFrame(
        {
            "human_score": [3, 0, 2],
            "human_label": ["3", "0.3", "2.2"],
            "score|gemini-3.5-flash|zero": [3, 3, 0],
            "label|gemini-3.5-flash|zero": ["3", "3", "0.2"],
            "score|deepseek-v4.1-flash|zero": [3, 0, 0],
            "label|deepseek-v4.1-flash|zero": ["3.1", "0.2", "0.3"],
        }
    )

    result = ensembles(frame, ["gemini-3.5-flash|zero", "deepseek-v4.1-flash|zero"])[
        "agree-or-human (zero)"
    ]

    assert result["accepted"].to_list() == [True, False, True]
    assert result["judge_score"].to_list() == [3, 0, 0]
    assert result["judge_label"].to_list() == ["3", "0.3", "0.2"]


def test_missing_verdict_is_routed_to_human() -> None:
    frame = pl.DataFrame(
        {
            "human_score": [2],
            "human_label": ["2.2"],
            "score|gemini-3.5-flash|zero": [None],
            "label|gemini-3.5-flash|zero": [None],
            "score|deepseek-v4.1-flash|zero": [None],
            "label|deepseek-v4.1-flash|zero": [None],
        },
        schema_overrides={
            "score|gemini-3.5-flash|zero": pl.Int64,
            "score|deepseek-v4.1-flash|zero": pl.Int64,
            "label|gemini-3.5-flash|zero": pl.String,
            "label|deepseek-v4.1-flash|zero": pl.String,
        },
    )

    result = ensembles(frame, ["gemini-3.5-flash|zero", "deepseek-v4.1-flash|zero"])[
        "agree-or-human (zero)"
    ]

    assert result["accepted"].to_list() == [False]
    assert result["judge_label"].to_list() == ["2.2"]
