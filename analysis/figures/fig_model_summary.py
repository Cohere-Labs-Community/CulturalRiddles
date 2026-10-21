"""Model accuracy, correct answers missed by exact match, and community spread.

Data come from ``analysis.tables.model_performance_summary.build_model_performance``
and its REASONING_LABELS, so strings and rounding match the table. The registered
figure uses stacked accuracy bars and half-violins of community accuracy.

Options:

* ``--all-communities``: include communities without human labels and add rank
  changes against the human-evaluated set.
* ``--panel bins``: replace half-violins with ordered community accuracy bins.
* ``--panel barcode``: draw community ticks behind the left accuracy panel.
* ``--left dumbbell``: connect exact-match and reviewed accuracy with dots.
* ``--out DIR``: save an alternative layout outside the registered outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
from scipy.stats import gaussian_kde

from analysis.figures._common import FULL_W, C, apply_style, new_fig, save
from analysis.figures._community_data import accuracy_inputs
from analysis.model_names import model_parameter_metadata
from analysis.model_results.human_eval_summary import COMMUNITY_BUCKET_COLUMNS
from analysis.style import ACCESS, MARKER_EDGE, OUTLINE_WIDTH, accuracy_colors
from analysis.tables.model_performance_summary import (
    REASONING_LABELS,
    build_model_performance,
)

apply_style()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--all-communities", action="store_true")
parser.add_argument("--panel", choices=("bins", "barcode", "violin"), default="violin")
parser.add_argument("--left", choices=("dumbbell", "stacked"), default="stacked")
parser.add_argument("--out", type=Path)
args = parser.parse_args()
if args.out is None and (args.all_communities or args.panel != "violin" or args.left != "stacked"):
    parser.error("Audit variants require --out DIR; figures/ holds registered outputs only.")
ALL = args.all_communities
BARCODE = args.panel == "barcode"
VIOLIN = args.panel == "violin"
STACKED = args.left == "stacked"
STEM = (
    "fig_model_summary"
    + ("_61" if ALL else "")
    + ("" if args.panel == "violin" else f"_{args.panel}")
    + ("" if STACKED else "_dumbbell")
)
# The two communities where no model exceeds 22%.
HARDEST = ("lumasaba_uganda", "mongondow_indonesia")


df = build_model_performance(all_communities=ALL)
meta = model_parameter_metadata().set_index("model_display")
df["access"] = df["model"].map(meta["access"])
df = df.sort_values("rank").reset_index(drop=True)
n = len(df)
if ALL:  # rank among the human-evaluated communities minus rank here
    reference = build_model_performance().set_index("model")["rank"]
    df["rank_change"] = df["model"].map(reference) - df["rank"]

assert set(df["access"]) <= set(ACCESS), set(df["access"])
print("access counts:", df["access"].value_counts().to_dict())

cmeta, cells = accuracy_inputs(all_communities=ALL)
cmeta = cmeta[cmeta["mean_human_correct"].notna()].reset_index(drop=True)
mat = cells.pivot(index="model_display", columns="community", values="human_correct").loc[
    df["model"], cmeta["community"]
]
N_COMM = mat.shape[1]
assert not mat.isna().any().any()
V = mat.to_numpy()
BINS = [(-1, 0), (0, 25), (25, 50), (50, 75), (75, 101)]
counts = np.stack(
    [
        ((V > lo) & (V <= hi)) if k == 0 else ((V >= lo) & (V < hi)) if k < 4 else (V >= 75)
        for k, (lo, hi) in enumerate(BINS)
    ],
    axis=1,
).sum(axis=2)
counts[:, 0] = (V == 0).sum(axis=1)
counts[:, 1] = ((V > 0) & (V < 25)).sum(axis=1)
assert (counts.sum(axis=1) == N_COMM).all()
assert (counts == df[list(COMMUNITY_BUCKET_COLUMNS)].to_numpy()).all()  # same as table bins
# Ordered light-to-dark fills shared by both publication sets.
BIN_COL = accuracy_colors()
BIN_TXT = [C["ink"], C["ink"], C["ink"], "white", "white"]
BIN_LAB = ["0", "(0,25)", "[25,50)", "[50,75)", "[75,100]"]

ROW = 0.118
body_h = n * ROW
y0, top_pad = 0.36, 0.44
W, H = FULL_W, y0 + body_h + top_pad

# x positions in inches
X_RANK, X_GLYPH, X_NAME = 0.14, 0.23, 0.31
X_PARAM_R, X_REAS = 2.30, 2.38
RANK_W = 0.34 if ALL else 0.0  # rank-change column at the right edge
MISS_W = 0.48 if STACKED else 0.0  # share-missed-by-exact-match column
LOW_W = 0.30 if VIOLIN else 0.0  # communities-below-50% column
AX_L, AX_R = 3.08, (W - 0.10 - RANK_W if BARCODE else 4.48) - MISS_W
CM_L, CM_R = 4.64, W - 0.10 - RANK_W - LOW_W
X_MISS, X_LOW = AX_R + MISS_W - 0.02, W - 0.12 - RANK_W

fig = new_fig(W, H, layout=None)
ax = fig.add_axes((AX_L / W, y0 / H, (AX_R - AX_L) / W, body_h / H))
ax.set_xlim(0, 100 if BARCODE else 88)
ax.set_ylim(-0.5, n - 0.5)
if not BARCODE:
    cx = fig.add_axes((CM_L / W, y0 / H, (CM_R - CM_L) / W, body_h / H))
    cx.set_xlim(0, 100)
    cx.set_ylim(-0.5, n - 0.5)
y = (n - 1) - np.arange(n)  # best on top
ft = blended_transform_factory(fig.transFigure, ax.transData)
SZ = 6.5
GAIN_ABOVE_MIN = 25.5  # pp; below this the connector is shorter than the label
INK = C["ink"]


def put(x_in, yy, s, ha="left", **kw):
    return ax.text(x_in / W, yy, s, transform=ft, ha=ha, va="center", fontsize=SZ, **kw)


EM_COL, REV_COL = C["blue"], C["sky"]
# orange: the only Okabe-Ito hue not already used for access or exact match
HARD_COL = C["orange"]
totals = cells.groupby("model_display")[["correct", "n_exact"]].sum().loc[df["model"]]
MISSED = 100 * (totals["correct"] - totals["n_exact"]) / totals["correct"]
print("share of correct answers missed by exact match:", MISSED.round(1).tolist())
GRID = np.linspace(0, 100, 401)
HARD_IDX = [cmeta["community"].tolist().index(c_) for c_ in HARDEST]
VIOLIN_H = 0.80  # rows; half-violin height above its baseline
VIOLIN_BASE = -0.38  # rows; baseline of the half-violin


def dumbbell_row(yi: float, r) -> None:
    """Draw the exact-match -> reviewed dumbbell with its values and gain."""
    ax.plot(
        [r.exact_match, r.human_accuracy],
        [yi, yi],
        color=C["grey"],
        lw=1.2,
        zorder=1,
        solid_capstyle="butt",
    )
    ax.plot(
        r.exact_match,
        yi,
        "o",
        mfc="white",
        mec=INK,
        mew=0.9,
        ms=3.6,
        zorder=3,
        clip_on=False,
    )
    ax.plot(r.human_accuracy, yi, "o", color=INK, ms=3.6, zorder=3, clip_on=False)
    # direct values: exact left of hollow dot, reviewed right of filled dot, gain above connector
    ax.annotate(
        f"{r.exact_match:.1f}",
        (r.exact_match, yi),
        xytext=(-3.6, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=SZ,
        color=INK,
    )
    ax.annotate(
        f"{r.human_accuracy:.1f}",
        (r.human_accuracy, yi),
        xytext=(3.6, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=SZ,
        color=INK,
    )
    if r.difference_pp >= GAIN_ABOVE_MIN:  # long connector: gain above its middle
        ax.annotate(
            f"+{r.difference_pp:.1f}",
            ((r.exact_match + r.human_accuracy) / 2, yi),
            xytext=(0, 1.7),
            textcoords="offset points",
            ha="center",
            va="baseline",
            fontsize=SZ,
            color=INK,
        )
    else:  # short connector: no room above it, so gain follows the reviewed value
        ax.annotate(
            f"+{r.difference_pp:.1f}",
            (r.human_accuracy, yi),
            xytext=(21, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=SZ,
            color=INK,
        )


def stacked_row(yi: float, r, missed: float) -> None:
    """Draw exact match, then the review-only gain, the total, and the share missed."""
    ax.barh(yi, r.exact_match, height=0.72, color=EM_COL, lw=0)
    ax.barh(
        yi,
        r.difference_pp,
        left=r.exact_match,
        height=0.72,
        color=REV_COL,
        edgecolor="white",
        lw=0.6,
    )
    ax.annotate(
        f"{r.human_accuracy:.1f}",
        (r.human_accuracy, yi),
        xytext=(2.5, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=SZ,
        color=INK,
    )
    put(X_MISS, yi, f"{missed:.0f}%", ha="right")


def violin_row(yi: float, values: np.ndarray) -> None:
    """Draw equally weighted community accuracies, highlighting two communities."""
    base = yi + VIOLIN_BASE
    density = gaussian_kde(values)(GRID)
    inside = (GRID >= values.min()) & (GRID <= values.max())
    top = base + VIOLIN_H * density / density.max()
    cx.fill_between(GRID[inside], base, top[inside], color=C["lightgrey"], lw=0, zorder=1)
    cx.plot(GRID[inside], top[inside], color=C["grey"], lw=0.5, zorder=2)
    cx.plot(
        values[HARD_IDX],
        [base] * len(HARD_IDX),
        "D",
        color=HARD_COL,
        mec=MARKER_EDGE,
        mew=0.4,
        ms=2.6,
        zorder=4,
        clip_on=False,
    )
    put(X_LOW, yi, str(int((values < 50).sum())), ha="right")


def bins_row(yi: float, row_counts: np.ndarray) -> None:
    """Draw the per-model share of communities in each accuracy bin."""
    left = 0.0
    for j, c_ in enumerate(row_counts):
        w_ = 100 * c_ / N_COMM
        cx.barh(yi, w_, left=left, height=0.82, color=BIN_COL[j], edgecolor="white", lw=0.3)
        if c_ >= 5:
            cx.text(
                left + w_ / 2,
                yi,
                str(int(c_)),
                ha="center",
                va="center",
                fontsize=SZ,
                color=BIN_TXT[j],
            )
        left += w_


if VIOLIN:
    cx.axvline(50, color=C["lightgrey"], lw=0.5, zorder=0)
for k_, (yi, (_, r)) in enumerate(zip(y, df.iterrows())):
    if STACKED:
        stacked_row(yi, r, MISSED.iloc[k_])
    else:
        dumbbell_row(yi, r)
    _, col, mk = ACCESS[r.access]
    ax.plot(
        X_GLYPH / W,
        yi,
        mk,
        color=col,
        mec=MARKER_EDGE,
        mew=OUTLINE_WIDTH,
        ms=3.6,
        transform=ft,
        clip_on=False,
        zorder=3,
    )
    put(X_RANK, yi, str(r["rank"]), ha="right")
    put(X_NAME, yi, r.model)
    put(X_PARAM_R, yi, r.parameters_b, ha="right")
    put(X_REAS, yi, REASONING_LABELS[str(r.reasoning)])
    if ALL and r.rank_change:
        put(
            W - 0.12,
            yi,
            f"{r.rank_change:+d}".replace("-", "−"),
            ha="right",
            color=C["blue"] if r.rank_change > 0 else C["vermillion"],
        )
    if BARCODE:  # one tick per community, behind the dumbbell
        ax.vlines(V[k_], yi - 0.36, yi + 0.36, color=C["grey"], lw=0.35, alpha=0.6, zorder=0)
    elif VIOLIN:
        violin_row(yi, V[k_])
    else:
        bins_row(yi, counts[k_])

for a_ in (ax,) if BARCODE else (ax, cx):
    a_.set_yticks([])
    a_.spines["left"].set_visible(False)
ax.set_xticks(range(0, 101 if BARCODE else 81, 20))
ax.set_xlabel("Accuracy (%)")
if not BARCODE:
    cx.set_xticks(range(0, 101, 25))
    cx.set_xlabel(
        f"Accuracy across {N_COMM} communities (%)"
        if VIOLIN
        else f"Share of {N_COMM} communities (%)"
    )

# ---- headers --------------------------------------------------------------
top = (y0 + body_h) / H
hdr1, hdr2 = top + 0.27 / H, top + 0.08 / H
dy = 0.045 / H


def head(x_in, s, row, ha="left"):
    fig.text(x_in / W, row, s, fontsize=SZ, va="bottom", ha=ha)


def glyph(x_in, row, **kw):
    kw.setdefault("ms", 3.6)
    fig.add_artist(Line2D([x_in / W], [row + dy], ls="", transform=fig.transFigure, **kw))


xk = X_GLYPH
for (lab, col, mk), step in zip(ACCESS.values(), [0.82, 0.78, 0.76]):
    glyph(xk, hdr1, marker=mk, color=col)
    head(xk + 0.08, lab, hdr1)
    xk += step
head(X_RANK, "#", hdr2, "right")
head(X_NAME, "Model", hdr2)
head(X_PARAM_R, "Params (B)", hdr2, "right")
head(X_REAS, "Reason.", hdr2)
if STACKED:
    glyph(AX_L + 0.03, hdr1, marker="s", color=EM_COL)
    head(AX_L + 0.11, "Exact match", hdr1)
    glyph(AX_L + 0.03, hdr2, marker="s", color=REV_COL)
    head(AX_L + 0.11, "Human + LLM jury", hdr2)
    head(X_MISS, "EM", hdr1, "right")
    head(X_MISS, "misses", hdr2, "right")
else:
    glyph(AX_L + 0.03, hdr1, marker="o", mfc="white", mec=INK, mew=0.9)
    head(AX_L + 0.11, "Exact match", hdr1)
    glyph(AX_L + 0.03, hdr2, marker="o", color=INK)
    head(AX_L + 0.11, "Human + LLM jury", hdr2)
if ALL:
    head(W - 0.12, "\u0394 rank", hdr2, "right")
if BARCODE:
    fig.add_artist(
        Line2D(
            [(AX_L + 1.85) / W] * 2,
            [hdr2 + 0.2 * dy, hdr2 + 1.8 * dy],
            color=C["grey"],
            lw=0.6,
            transform=fig.transFigure,
        )
    )
    head(AX_L + 1.90, f"Each of {N_COMM} communities", hdr2)
if VIOLIN:
    head(CM_L, "Community accuracy", hdr1)
    glyph(CM_L + 0.02, hdr2, marker="D", color=HARD_COL, ms=2.6)
    head(CM_L + 0.08, "Lumasaba, Mongondow", hdr2)
    head(X_LOW, "<50%", hdr2, "right")
xs = [CM_L, CM_L + 0.37, CM_L + 0.92]
xs2 = [CM_L, CM_L + 0.55]
for k, (xx, rowy) in enumerate(
    [] if BARCODE or VIOLIN else [(x_, hdr1) for x_ in xs] + [(x_, hdr2) for x_ in xs2]
):
    fig.add_artist(
        Rectangle(
            (xx / W, rowy + 0.005 / H),
            0.10 / W,
            0.085 / H,
            transform=fig.transFigure,
            facecolor=BIN_COL[k],
            edgecolor=C["grey"],
            lw=0.3,
        )
    )
    head(xx + 0.13, BIN_LAB[k], rowy)

if args.out:
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / f"{STEM}.pdf")
    fig.savefig(args.out / f"{STEM}.png", dpi=450)
else:
    save(fig, STEM)
