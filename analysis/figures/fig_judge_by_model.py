"""Redesign of tab:judge_by_model: judge leniency and accuracy shift per answering model.

Data come from the same CSVs and the same transformations as
``analysis.tables.judge_by_model`` (``false_accept_by_model.csv`` pivoted by judge,
``leaderboard_intervals.csv`` for the ``agree-or-human (zero)`` rule), so every
number equals the table. Row order is human accuracy (as in fig_model_summary).
Every plotted value is also printed in a column beside its panel, in the table's
order (DS, Gem., Both; then the accuracy shift with its 95% CI), so the figure can
be scanned like the table. Shifts whose CI excludes zero are bold, as in the table.
"""

from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from analysis.figures._common import FULL_W, C, apply_style, new_fig, panel_letter, save
from analysis.style import JUDGE, MARKER_EDGE, OVERVIEW
from analysis.tables.judge_by_model import judge_model_frame
from analysis.tables.model_performance_summary import build_model_performance

apply_style()

# Small vertical offsets keep near-identical judge rates from hiding one another.
DY = {"DeepSeek": 0.2, "Gemini": -0.2, "Both": 0.0}
POS, NEG, NS = C["blue"], C["vermillion"], C["grey"]
FS = 6.5
MUTED = OVERVIEW["subtle"]
COLUMN_ORDER = ("DeepSeek", "Gemini", "Both")  # the table's DS, Gem., Both
COLUMN_HEAD = {"DeepSeek": "DS", "Gemini": "Gem.", "Both": "Both"}

# ---- data: identical pipeline to the table generator ----------------------------
tab = judge_model_frame().to_pandas()
# Row order = human accuracy, the order used by the model table and figures.
acc = build_model_performance()[["model", "human_accuracy"]]
tab = tab.merge(acc, on="model", how="left", validate="one_to_one")
assert tab["human_accuracy"].notna().all(), tab[tab["human_accuracy"].isna()]
tab = tab.sort_values("human_accuracy", ascending=False).reset_index(drop=True)
n = len(tab)
print("max |shift| (pp):", round(100 * tab["shift"].abs().max(), 2))


def num(value: float, plus: bool = False) -> str:
    """Format a one-decimal number with a true minus sign and no negative zero."""
    rounded = round(value, 1) or 0.0
    return (f"{rounded:+.1f}" if plus else f"{rounded:.1f}").replace("-", "\u2212")


# ---- geometry (inches) ----------------------------------------------------------
ROW = 0.125
W = FULL_W
body_h = n * ROW
y0, top_pad = 0.58, 0.50
H = y0 + body_h + top_pad
x_names, x_n = 0.04, 1.86  # names left edge; right edge of the n_rej column
ax1_l, ax1_r = 1.98, 3.10
X_JUDGE = dict(zip(COLUMN_ORDER, (3.40, 3.68, 3.96), strict=True))  # right edges of rate columns
ax2_l, ax2_r = 4.10, 5.22
X_SHIFT = 5.30  # left edge of the shift (CI) column


def inch_axes(left, right):
    return fig.add_axes((left / W, y0 / H, (right - left) / W, body_h / H))


fig = new_fig(W, H, layout=None)
ax1, ax2 = inch_axes(ax1_l, ax1_r), inch_axes(ax2_l, ax2_r)
y = np.arange(n)[::-1]

for ax in (ax1, ax2):
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)

# ---- (a) leniency ---------------------------------------------------------------
ax1.set_xlim(0, 30)
for yi, r in zip(y, tab.itertuples()):
    v = {k: 100 * getattr(r, k) for k in JUDGE}
    ax1.plot(
        [min(v.values()), max(v.values())],
        [yi, yi],
        color=C["lightgrey"],
        lw=1.2,
        zorder=1,
        solid_capstyle="butt",
    )
    for k in ("Gemini", "DeepSeek", "Both"):  # focal drawn last
        _, col, mk, ms = JUDGE[k]
        ax1.plot(
            v[k],
            yi + DY[k],
            mk,
            color=col,
            ms=ms,
            mec=MARKER_EDGE,
            mew=0.4,
            zorder=3 if k != "Both" else 4,
            clip_on=False,
        )
ax1.set_xticks([0, 10, 20, 30])
ax1.set_xlabel("Human-rejected answers\njudged correct (%)")

# ---- (b) accuracy shift ---------------------------------------------------------
ax2.set_xlim(-1.7, 2.4)
ax2.axvline(0, color=C["ink"], lw=1.0, zorder=1)
for yi, r in zip(y, tab.itertuples()):
    s, lo, hi = 100 * r.shift, 100 * r.low, 100 * r.high
    sig = lo > 0 or hi < 0
    col = (POS if s > 0 else NEG) if sig else NS
    ax2.plot([lo, hi], [yi, yi], color=col, lw=1.1, zorder=2, solid_capstyle="butt")
    if sig:
        ax2.plot(
            s,
            yi,
            "^" if s > 0 else "v",
            color=col,
            ms=4.0,
            mec=MARKER_EDGE,
            mew=0.3,
            zorder=3,
            clip_on=False,
        )
    else:
        ax2.plot(s, yi, "o", mfc="white", mec=col, mew=1.0, ms=3.6, zorder=3, clip_on=False)
ax2.set_xticks([-1, 0, 1, 2])
ax2.set_xlabel("Accuracy, judges\nminus humans (pp)")

# ---- names and numeric columns (fixed inch positions, one text per cell) ----------
trans = blended_transform_factory(fig.transFigure, ax1.transData)


def cell(x_in, yi, text, ha, **kw):
    ax1.text(x_in / W, yi, text, transform=trans, va="center", ha=ha, fontsize=FS, **kw)


for yi, r in zip(y, tab.itertuples()):
    cell(x_names, yi, r.model, "left")
    cell(x_n, yi, f"{r.rejected_rows:,}", "right", color=MUTED)
    for k in COLUMN_ORDER:
        cell(X_JUDGE[k], yi, num(100 * getattr(r, k)), "right")
    sig = r.low > 0 or r.high < 0
    cell(
        X_SHIFT,
        yi,
        f"{num(100 * r.shift, plus=True)} [{num(100 * r.low)}, {num(100 * r.high)}]",
        "left",
        fontweight="bold" if sig else "normal",
    )

# ---- headers: direct keys, no boxes ---------------------------------------------
top = (y0 + body_h) / H
hdr_key, hdr_col = top + 0.20 / H, top + 0.04 / H
ftr = fig.transFigure
dy = 0.045 / H


def glyph(x_in, y_f, **kw):
    fig.add_artist(Line2D([x_in / W], [y_f + dy], ls="", transform=ftr, **kw))


def head(x_in, y_f, text, ha="left", **kw):
    fig.text(x_in / W, y_f, text, fontsize=FS, ha=ha, va="bottom", **kw)


head(x_names, hdr_col, "Answering model")
head(x_n, hdr_col, "$n_{\\mathrm{rej}}$", "right", color=MUTED)
for k in COLUMN_ORDER:
    head(X_JUDGE[k], hdr_col, COLUMN_HEAD[k], "right")
head(X_SHIFT, hdr_col, "$\\Delta$ (pp) [95% CI]")

xk = ax1_l + 0.14  # clear of the panel letter
for k, step in zip(("Both", "DeepSeek", "Gemini"), (0.50, 0.68, 0.60), strict=True):
    _, col, mk, ms = JUDGE[k]
    glyph(xk + 0.03, hdr_key, marker=mk, color=col, ms=ms + 0.3, mec=MARKER_EDGE, mew=0.4)
    head(xk + 0.11, hdr_key, k)
    xk += step
xk = ax2_l + 0.14
for (m, kw, lab), step in zip(
    [
        ("^", {"color": POS}, "CI > 0"),
        ("v", {"color": NEG}, "CI < 0"),
        ("o", {"mfc": "white", "mec": NS, "mew": 1.0}, "CI includes 0"),
    ],
    (0.55, 0.55, 0.9),
    strict=True,
):
    glyph(xk + 0.03, hdr_key, marker=m, ms=4.0 if m != "o" else 3.6, **kw)
    head(xk + 0.11, hdr_key, lab)
    xk += step

panel_letter(ax1, "a", x=-0.0, y=1.0 + 0.16 / (body_h))
panel_letter(ax2, "b", x=-0.0, y=1.0 + 0.16 / (body_h))
save(fig, "fig_judge_by_model")
