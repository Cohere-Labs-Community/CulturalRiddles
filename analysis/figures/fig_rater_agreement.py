"""Rater agreement in the three judge-validation settings, at three granularities.

One dumbbell per rater pair, grouped by setting (A: two annotators, B: one
annotator, C: no human label). The three markers on a row are Cohen's kappa
for accuracy (score 3 versus the rest, solid), the quadratic-weighted 0--3
score (hatched), and the tree-weighted error type (hollow), each with its
95% bootstrap interval. The human--human intervals are shaded, so each LLM pair
reads against them. Setting names, annotator glyphs, and pair counts sit in the
left margin, before the rater pairs. Same data as the ``judge_agreement``
table.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import polars as pl

from analysis.figures._common import PREPRINT_W, C, apply_style, new_fig, save
from analysis.tables import judge_agreement as ja

apply_style()
mpl.rcParams["hatch.linewidth"] = 0.6

# Pair type -> (label, colour, marker)
KIND = {
    "hh": ("Human\u2013human", C["black"], "o"),
    "lh": ("LLM\u2013human", C["orange"], "s"),
    "ll": ("LLM\u2013LLM", C["purple"], "D"),
}
LBL = ja.RATER_LABELS
# kappa level -> (fill, legend label); solid > hatched > hollow reads as
# coarse > graded > fine labels
LEVELS = (
    ("binary", "solid", "Accuracy (correct vs. incorrect)"),
    ("score", "hatch", "Score 0\u20133 (weighted)"),
    ("fine_tree", "none", "13 error types (weighted)"),
)
HATCH = "//////"
XMIN, XMAX = 0.30, 1.0


def kind(pair: str) -> str:
    n = sum(r in ja.HUMANS for r in pair.split("~"))
    return {2: "hh", 1: "lh", 0: "ll"}[n]


def label(pair: str) -> str:
    a, b = pair.split("~")
    return "LLM judges" if pair == ja.JUDGE_PAIR else f"{LBL[a]}\u2013{LBL[b]}"


def draw_marker(ax, x, y, col, mk, fill, ms, zorder=5):
    """One kappa marker (solid, hatched, or hollow) in the pair colour."""
    if fill == "hatch":
        ax.scatter(
            [x],
            [y],
            marker=mk,
            s=ms**2,
            facecolor="white",
            edgecolor=C["ink"],
            linewidths=1.0,
            hatch=HATCH,
            zorder=zorder,
        )
    else:
        ax.plot(
            x,
            y,
            mk,
            mfc=col if fill == "solid" else "white",
            mec=C["ink"],
            mew=1.0,
            ms=ms,
            zorder=zorder,
        )


# ---- data (same functions/files as the table) --------------------------------
d = ja.RATER_DIR
pooled = (pl.col("sample") == "primary") & (pl.col("scope") == "Pooled")
agr = pl.read_csv(d / "agreement.csv").filter(pooled)
rel = pl.read_csv(d / "judge_reliability.csv")
strat = rel.filter(pl.col("setting") == "stratified")
pend = rel.filter(pl.col("setting") == "pending")
groups = [
    ("A. Two annotators", int(agr["units"].max()), 2, ja.kappa_rows(agr, ja.DOUBLY_PAIRS)),
    (
        "B. One annotator",
        int(strat["units"].max()),
        1,
        ja.kappa_rows(strat, ja.SETTING_PAIRS["stratified"]),
    ),
    (
        "C. No human label",
        int(pend["units"].max()),
        0,
        ja.kappa_rows(pend, ja.SETTING_PAIRS["pending"]),
    ),
]

GAP = 1.0
rows = []  # (y, pair, kappa dict)
heads = []  # (y of first row, title, pairs, annotators)
y = 0.0
for title, units, nh, kap in groups:
    heads.append((y, len(kap), title, units, nh))
    for pair, v in kap.items():
        rows.append((y, pair, v))
        y += 1
    y += GAP
ymax = y - GAP
ref = groups[0][3]["H1~H2"]

H = 4.1
fig = new_fig(PREPRINT_W, H, layout=None)
# the value columns sit right of the axes, so the layout is set by hand
ax = fig.add_axes([0.305, 0.085, 0.515, 0.80])

# ---- human--human reference bands, one per kappa ---------------------------------
for lvl, _, _ in LEVELS:
    est, lo, hi = ref[lvl]
    ax.axvspan(lo, hi, color=C["lightgrey"], alpha=0.5, lw=0, zorder=0)
    ax.axvline(est, color=C["ink"], lw=0.7, ls=(0, (3, 2)), zorder=1)

# ---- dumbbells ----------------------------------------------------------------------
VALUE_X = (1.07, 1.185, 1.30)  # axes-fraction x of the accuracy, score, type columns
for yy, pair, v in rows:
    _, col, mk = KIND[kind(pair)]
    ests = [v[lvl][0] for lvl, _, _ in LEVELS]
    ax.plot([min(ests), max(ests)], [yy, yy], color=col, lw=1.4, zorder=2, solid_capstyle="butt")
    for lvl, fill, _ in LEVELS:
        e, lo, hi = v[lvl]
        ax.errorbar(
            e,
            yy,
            xerr=[[e - lo], [hi - e]],
            fmt="none",
            ecolor=col,
            elinewidth=0.8,
            capsize=1.8,
            capthick=0.8,
            alpha=0.7,
            zorder=3,
        )
    # the hatched marker is largest and underneath, so equal kappas stay visible
    for (lvl, fill, _), ms, z in zip(LEVELS, (4.6, 7.2, 5.2), (6, 4, 5), strict=True):
        draw_marker(ax, v[lvl][0], yy, col, mk, fill, ms * (1.05 if mk == "D" else 1.0), z)
    # numeric columns right of the axes: accuracy, score, error type
    for (lvl, _, _), cx in zip(LEVELS, VALUE_X, strict=True):
        ax.text(
            cx,
            yy,
            f"{v[lvl][0]:.2f}",
            transform=ax.get_yaxis_transform(),
            fontsize=7,
            ha="center",
            va="center",
            color=C["ink"],
        )

for (lvl, fill, _), cx, text in zip(LEVELS, VALUE_X, ("Acc.", "Score", "Type"), strict=True):
    ax.text(
        cx,
        -0.85,
        text,
        transform=ax.get_yaxis_transform(),
        fontsize=6.5,
        ha="center",
        va="center",
        color=C["grey"],
    )

# ---- setting names, annotator glyphs, and pair counts in the left margin -----------
# x in figure fractions, y in data units, so each block sits above its group's rows
margin = mtrans.blended_transform_factory(fig.transFigure, ax.transData)
MX = 0.012
for yy, nrows, title, units, nh in heads:
    ty = yy - 0.85
    gy = yy - 0.25
    ax.text(
        MX,
        ty,
        title,
        transform=margin,
        fontsize=7,
        fontweight="bold",
        ha="left",
        va="center",
        color=C["ink"],
    )
    for k in range(2):
        if k < nh:
            style = dict(color=C["black"], mec="none")
        else:
            style = dict(mfc="none", mec=C["grey"], mew=0.7)
        ax.plot(MX + 0.008 + 0.022 * k, gy, "o", ms=4.2, transform=margin, clip_on=False, **style)
    ax.text(
        MX + 0.058,
        gy,
        f"{units:,} pairs",
        transform=margin,
        fontsize=6.5,
        ha="left",
        va="center",
        color=C["ink"],
    )

ax.set_xlim(XMIN, XMAX)
ax.set_ylim(ymax + 0.55, -1.45)
ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xlabel("Cohen\u2019s $\\kappa$")
ax.set_yticks([r[0] for r in rows])
ax.set_yticklabels([label(r[1]) for r in rows], fontsize=7)
ax.tick_params(axis="y", length=0, pad=3)
ax.spines["left"].set_visible(False)

# ---- legend: pair type (colour + shape) and kappa level (fill) ------------------------
grey = C["grey"]
pair_handles = [
    plt.Line2D([], [], marker=marker, color=color, ls="", ms=4.5, label=label)
    for label, color, marker in KIND.values()
]
band = plt.Line2D(
    [], [], color=C["ink"], lw=0.8, ls=(0, (3, 2)), label="Human\u2013human, 95% interval"
)
level_handles = [
    plt.scatter(
        [],
        [],
        marker="o",
        s=4.5**2,
        facecolor=grey if fill == "solid" else "white",
        edgecolor=grey,
        linewidths=1.0,
        hatch=HATCH if fill == "hatch" else None,
        label=text,
    )
    for _, fill, text in LEVELS
]
blank = plt.Line2D([], [], ls="", label="")
# legend fills column-wise: top row = pair types, bottom row = kappa levels
handles = [
    pair_handles[0],
    level_handles[0],
    pair_handles[1],
    level_handles[1],
    pair_handles[2],
    level_handles[2],
    band,
    blank,
]
fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=4,
    fontsize=7,
    handletextpad=0.2,
    columnspacing=0.8,
    frameon=False,
)
save(fig, "fig_rater_agreement")
