"""Tests for the jury-only counterfactual estimator on a hand-built design."""

import numpy as np
from scipy.sparse import csr_matrix

from analysis.llm_judge.native_counterfactual import (
    Design,
    bootstrap_draws,
    cell_counts,
)


def make_design() -> Design:
    """Two communities, one model; three sampled pairs, the last in a census stratum."""
    return Design(
        models=["m"],
        communities=["a", "b"],
        correct=np.array([[10.0], [4.0]]),
        total=np.array([[20.0], [10.0]]),
        pair_community=csr_matrix((np.ones(3), ([0, 0, 1], [0, 1, 2])), shape=(2, 3)),
        pair_rows=np.array([[2.0], [3.0], [1.0]]),
        d_correct={"agree": np.array([1.0, -1.0, 0.0])},
        d_total={"agree": np.array([0.0, 0.0, -1.0])},
        draw_start=np.array([0, 0, 2]),
        draw_size=np.array([2, 2, 0]),
    )


def test_cell_counts_adds_weighted_corrections():
    design = make_design()

    correct, total = cell_counts(design, "agree", np.ones(3))

    assert correct.tolist() == [[9.0], [4.0]]
    assert total.tolist() == [[20.0], [9.0]]


def test_bootstrap_keeps_census_pairs_and_stratum_size():
    design = make_design()
    rng = np.random.default_rng(0)

    draws = np.array([bootstrap_draws(design, rng) for _ in range(200)])

    assert (draws[:, 2] == 1).all()
    assert (draws[:, :2].sum(axis=1) == 2).all()
    assert draws[:, 0].min() == 0 and draws[:, 0].max() == 2
