"""Forest plot of riddle-feature odds ratios (replaces tab:riddle-features)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.transforms as mt
import pandas as pd

from analysis.difficulty_decomposition.prepare import OUTPUT_DIR
from analysis.difficulty_decomposition.riddle_features import FEATURE_FRAME_PATH
from analysis.figures._common import FULL_W, C, apply_style, new_fig, save

apply_style()
eff = pd.read_csv(OUTPUT_DIR / "riddle_feature_effects.csv").set_index("term")
sample = pd.read_csv(
    FEATURE_FRAME_PATH,
    usecols=["riddle_id", "answer_kind", "riddle_form", "multiline", "multi_clue"],
).drop_duplicates("riddle_id")
answer_counts = sample["answer_kind"].value_counts()
form_counts = sample["riddle_form"].value_counts()

# group, colour, marker, [(label, term|None, n)] ; groups ordered by largest |log OR|,
# rows within group by OR.
GROUPS = [
    (
        "Answer",
        C["vermillion"],
        "o",
        [
            (
                "No common English name",
                "answer_kindlocal_term",
                f"{answer_counts['local_term']:,}",
            ),
            (
                "In between",
                "answer_kindmid_frequency",
                f"{answer_counts['mid_frequency']:,}",
            ),
            ("Common English word", None, f"{answer_counts['common_word']:,}"),
            ("Number", "answer_kindnumeric", f"{answer_counts['numeric']:,}"),
        ],
    ),
    (
        "Length",
        C["blue"],
        "D",
        [
            ("Per doubling of words", "log2_words_c", ""),
        ],
    ),
    (
        "Form",
        C["green"],
        "s",
        [
            ("Question", "riddle_formquestion", f"{form_counts['question']:,}"),
            (
                "Third-person description",
                None,
                f"{form_counts['third_person_description']:,}",
            ),
            (
                "First-person statement",
                "riddle_formfirst_person_statement",
                f"{form_counts['first_person_statement']:,}",
            ),
        ],
    ),
    (
        "Clues",
        C["purple"],
        "^",
        [
            (
                "Multiline (verse)",
                "multilineTRUE",
                f"{int(sample['multiline'].sum()):,}",
            ),
            ("Several clues", "multi_clueTRUE", f"{int(sample['multi_clue'].sum()):,}"),
        ],
    ),
]

fig = new_fig(FULL_W, 2.5, layout=None)
L, R, B, T = 0.345, 0.655, 0.17, 0.90
ax = fig.add_axes((L, B, R - L, T - B))
ax.set_xscale("log")
ax.set_xlim(0.3, 5)
ax.set_xticks([0.5, 1, 2, 4])
ax.set_xticklabels(["0.5", "1", "2", "4"])
ax.minorticks_off()
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0, labelleft=False)

# row positions (top to bottom) with gap between groups
rows, y = [], 0.0
for gi, (g, col, mk, items) in enumerate(GROUPS):
    for it in items:
        rows.append((y, g, col, mk, it))
        y += 1
    y += 0.6
ymax = y - 0.6
ax.set_ylim(ymax - 0.5 + 0.0, -0.9)
ax.axvline(1, color=C["ink"], lw=0.8, zorder=1)
ax.text(1, -0.95, "no effect (OR = 1)", ha="center", va="bottom", fontsize=7, color=C["ink"])

lab = mt.blended_transform_factory(fig.transFigure, ax.transData)
tr_axes = mt.blended_transform_factory(ax.transAxes, ax.transData)
xo, xn = 0.835, 0.99  # right-aligned OR(CI) and n columns (fig coords)
hy = -0.95
for xx, s in ((xo, "OR (95% CI)"), (xn, "Riddles")):
    fig.text(
        xx,
        ax.transData.transform((0, hy))[1] / fig.bbox.height,
        s,
        ha="right",
        va="bottom",
        fontsize=7,
        color=C["ink"] if s.startswith("OR") else C["grey"],
    )

for y, g, col, mk, (name, term, n) in rows:
    py = ax.transData.transform((0, y))[1] / fig.bbox.height
    fig.text(L - 0.012, py, name, ha="right", va="center", fontsize=7.5)
    if term is None:
        ax.plot(
            [1],
            [y],
            marker=mk,
            ms=5,
            mfc="white",
            mec=C["grey"],
            mew=1.0,
            ls="none",
            zorder=3,
        )
        ax.text(1.12, y, "ref.", fontsize=7, va="center", ha="left", color=C["grey"])
        orci = "ref."
    else:
        r = eff.loc[term]
        lo, hi, o = r.ci_low, r.ci_high, r.odds_ratio
        hollow = lo <= 1 <= hi
        ax.plot([lo, hi], [y, y], color=col, lw=1.2, zorder=2, solid_capstyle="butt")
        ax.plot(
            [o],
            [y],
            marker=mk,
            ms=5.5,
            mfc="white" if hollow else col,
            mec=col,
            mew=1.2,
            ls="none",
            zorder=3,
        )
        orci = f"{o:.2f} ({lo:.2f}\u2013{hi:.2f})"
    fig.text(xo, py, orci, ha="right", va="center", fontsize=7.5)
    fig.text(xn, py, n or "\u2013", ha="right", va="center", fontsize=7.5, color=C["grey"])

# group labels, set in the group's colour at the far left
for g, col, mk, items in GROUPS:
    ys = [r[0] for r in rows if r[1] == g]
    py = ax.transData.transform((0, ys[0]))[1] / fig.bbox.height
    fig.text(
        0.026,
        py,
        g,
        ha="left",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=C["ink"],
    )
    fig.add_artist(
        plt.Line2D(
            [0.012],
            [py],
            transform=fig.transFigure,
            marker=mk,
            ms=5,
            color=col,
            mfc=col,
            mec=col,
            ls="none",
        )
    )

ax.set_xlabel("Odds ratio of a correct answer (log scale; <1 = harder, >1 = easier)")
save(fig, "fig_riddle_features")
