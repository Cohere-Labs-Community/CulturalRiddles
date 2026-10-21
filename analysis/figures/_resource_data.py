"""Data for the resource-level figures, in two community sets.

* ``"57"``: every human-evaluated community (the set behind fig:cultures and
  ``\\numhumanscoredcommunities``), all pooled.
* ``"55"``: communities with >= ``MIN_HUMAN_RESPONSES`` (600) human-graded rows,
  the old ``community_accuracy()`` rule. The two excluded communities (Sudanese
  Arabic 591, Moroccan Arabic 274 human rows) are returned flagged ``hollow``
  and are NOT pooled.
"""

from __future__ import annotations

import pandas as pd

from analysis.eval_analysis.accuracy_by_group import (
    _order_resourcedness,
    community_accuracy,
    summarize_grouping,
)

ROW_ORDER = ["high", "medium", "low", "extremely-low"]  # top -> bottom
SWAHILI = "swahili_kenya"
SUDANESE = "arabic_sudan"


def load_variant(variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(points, summary)``.

    ``points``: one row per plotted community (columns community, language,
    language_resourcedness, accuracy, n_scored_responses, hollow).
    ``summary``: per-level pooled stats (from ``summarize_grouping``) indexed by
    level in ``ROW_ORDER``; pooled over the non-hollow points only.
    """
    everyone = community_accuracy(min_human_responses=1)  # the 57
    if variant == "57":
        pooled = everyone
        points = pooled.assign(hollow=False)
    elif variant == "55":
        pooled = community_accuracy()  # >= MIN_HUMAN_RESPONSES human rows
        extra = everyone.loc[~everyone["community"].isin(pooled["community"])]
        points = pd.concat(
            [pooled.assign(hollow=False), extra.assign(hollow=True)], ignore_index=True
        )
    else:
        raise ValueError(variant)
    summary = _order_resourcedness(summarize_grouping(pooled, "language_resourcedness"))
    summary = summary.set_index("group").loc[ROW_ORDER]
    return points, summary
