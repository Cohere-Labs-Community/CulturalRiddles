"""Redesigned error-type figure: dot matrix, rows = models by accuracy, cols = outcome classes."""

from __future__ import annotations

import matplotlib.transforms as mt
import numpy as np
import pandas as pd

from analysis.figures._common import FULL_W, OUTCOME, C, apply_style, new_fig, save
from analysis.figures._outcomes import outcome_shares
from analysis.style import OUTLINE_WIDTH
from analysis.tables.human_error_types import error_type_frame

GROUPS = {
    "correct": [
        "Exact match",
        "Different valid answer",
        "Correct, in English",
        "Other correct",
    ],
    "partial": [
        "Closely related culture",
        "Wrong level of detail",
        "Surface pattern matching",
        "Other partial",
    ],
    "halluc": ["Hallucinated"],
    "incorrect": ["Right category, wrong entity", "Too broad", "Other incorrect"],
    "declined": ["Declined to answer"],
    "noanswer": ["Placeholder", "Empty output"],
}
MARK = {
    "correct": "o",
    "partial": "D",
    "halluc": "s",
    "incorrect": "^",
    "declined": "v",
    "noanswer": "P",
}

# Same population as the error-type table and macros: rule-scored +
# human-scored responses of the reported models (no LLM-jury, no unscored).
frame = error_type_frame()
assert not frame["scoring_method"].str.contains("judge|pending").any()
_n = frame["model_display"].value_counts()
print("N per model min/max", _n.min(), _n.max(), "total", len(frame))
sh = outcome_shares(frame) * 100  # already sorted by exact-match share (old order)
cls = pd.DataFrame({k: sh[v].sum(axis=1) for k, v in GROUPS.items()})
assert np.allclose(cls.sum(axis=1), 100, atol=0.5), cls.sum(axis=1)
n = len(cls)
_w = _n.reindex(cls.index)
print(
    f"pooled halluc {(cls.halluc * _w).sum() / _w.sum():.2f} declined {(cls.declined * _w).sum() / _w.sum():.2f}; per-model halluc mean {cls.halluc.mean():.1f} min {cls.halluc.min():.1f} max {cls.halluc.max():.1f}; declined max {cls.declined.max():.2f}"
)
print(
    n,
    "models; declined max",
    cls.declined.max(),
    "mean",
    cls.declined.mean(),
    "halluc corr w/ correct",
    cls.halluc.corr(cls.correct),
)

apply_style()
H = 0.125 * n + 0.7
fig = new_fig(FULL_W, H)
axA, axB = fig.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [3.2, 1.5], "wspace": 0.0})
y = np.arange(n)
xmax_b = float(np.ceil(cls[["partial", "declined", "noanswer"]].max().max() / 10) * 10)
assert xmax_b >= cls[["partial", "declined", "noanswer"]].max().max()
HEAD = {
    "correct": "Correct",
    "halluc": "Hallucinated",
    "incorrect": "Other incorrect",
    "partial": "Partially\ncorrect",
    "declined": "Declined\nto answer",
    "noanswer": "Placeholder/\nempty",
}
SIZE = {"halluc": 5.2, "declined": 4.8}
LAB_X = {"correct": 70, "halluc": 38, "incorrect": 62}


def series(ax, k, xs=None, header_x=None, lw=0.0):
    _, col, _ = OUTCOME[k]
    ms = SIZE.get(k, 3.8)
    ax.plot(
        cls[k].values,
        y,
        MARK[k],
        color=col,
        ms=ms,
        mec=C["ink"],
        mew=OUTLINE_WIDTH,
        ls="none",
        zorder=3,
        clip_on=False,
    )


# Panel A: three large outcomes on one axis, light connector per row.
lo = cls[["correct", "halluc", "incorrect"]]
axA.hlines(y, lo.min(axis=1), lo.max(axis=1), color=C["lightgrey"], lw=0.6, zorder=1)
for k in ("correct", "incorrect", "halluc"):
    series(axA, k)
axA.set_xlim(0, 100)
axA.set_xticks([0, 25, 50, 75, 100])
# Panel B: zoomed axis for the three small outcomes.
bk = ("partial", "noanswer", "declined")
for k in bk:
    series(axB, k)
axB.set_xlim(0, xmax_b)
axB.set_xticks(range(0, int(xmax_b) + 1, 10))
for ax in (axA, axB):
    ax.set_ylim(n - 0.4, -0.8)
    ax.tick_params(axis="x", labelsize=6.5)
axB.spines["left"].set_visible(True)
axB.tick_params(axis="y", left=False, labelleft=False)
axA.tick_params(axis="y", length=0)

# direct headers: glyph + ink text, drawn in axes coords above the plot


def header(ax, k, xf, align="center"):
    _, col, _ = OUTCOME[k]
    focal = k in ("halluc", "declined")
    ax.text(
        xf,
        1.0,
        HEAD[k],
        transform=ax.transAxes + mt.ScaledTranslation(0, 10 / 72, fig.dpi_scale_trans),
        ha=align,
        va="bottom",
        fontsize=8,
        fontweight="bold" if focal else "normal",
        color=C["ink"],
        linespacing=1.0,
    )
    ax.plot(
        [xf],
        [1.0],
        MARK[k],
        color=col,
        ms=(6.0 if focal else 4.8),
        mec=C["ink"],
        mew=OUTLINE_WIDTH,
        clip_on=False,
        zorder=4,
        transform=ax.transAxes + mt.ScaledTranslation(0, 3.5 / 72, fig.dpi_scale_trans),
    )
    # lift text clear of glyph


header(axA, "correct", 0.80)
header(axA, "halluc", 0.20)
header(axA, "incorrect", 0.50)


def inline(k, row):
    _, col, _ = OUTCOME[k]
    focal = k == "declined"
    axB.plot(
        [15.2],
        [row],
        MARK[k],
        color=col,
        ms=(6.0 if focal else 4.8),
        mec=C["ink"],
        mew=OUTLINE_WIDTH,
        clip_on=False,
        zorder=4,
    )
    axB.text(
        17.2,
        row,
        HEAD[k],
        ha="left",
        va="center",
        fontsize=8,
        linespacing=1.0,
        fontweight="bold" if focal else "normal",
        color=C["ink"],
    )


inline("declined", 5.5)
inline("partial", 9.0)
inline("noanswer", 15.5)
j = int(np.argmax(cls.declined.values))
axB.annotate(
    f"max {cls.declined.max():.1f}%",
    (cls.declined.iloc[j], y[j]),
    xytext=(cls.declined.iloc[j] + 3, y[j]),
    fontsize=7,
    color=OUTCOME["declined"][1],
    fontweight="bold",
    va="center",
)
axA.set_yticks(y)
axA.set_yticklabels(cls.index, fontsize=7)
for t in axA.get_yticklabels()[:3]:
    t.set_fontweight("bold")
axA.set_xlabel("Share of rule- and human-scored responses (%)", fontsize=8)
axB.set_xlabel(f"Same, zoomed: 0\u2013{int(xmax_b)} % scale", fontsize=8)
save(fig, "fig_errors")
