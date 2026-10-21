"""Riddle surface forms: one strip row per form, one dot per community.

Message: a community's share of each riddle form ranges from ~0 to ~100%, far
from the pooled mix (black diamonds), and five named communities sit at the
extremes. Data: ``analysis.riddle_analysis.surface_forms`` (classification and
shares are reused unchanged).

Run from the repo root::

    uv run python -m analysis.figures.fig_surface_forms
"""

from __future__ import annotations

import numpy as np
import polars as pl

from analysis.figures._common import COL_W, C, apply_style, new_fig, save
from analysis.riddle_analysis.surface_forms import (
    EXEMPLAR_COMMUNITIES,
    FORMS,
    OVERALL,
    classify_forms,
    form_shares,
)
from analysis.style import FORMS as FORM_STYLE

STEM = "fig_surface_forms"
HEIGHT = 2.6

# Colour + marker per form; every row is also labelled.
STYLE = FORM_STYLE
ROW_LABEL = {
    "who_am_i": "\u201cWho am I?\u201d",
    "what_is_it": "\u201cWhat is it?\u201d",
    "other_question": "Other\nquestion",
    "first_person_statement": "First-person\nstatement",
    "third_person_description": "Third-person\ndescription",
}
# Exemplar community -> form in which it is (near-)exclusive.
FOCAL_FORM = {
    "luganda_uganda": "who_am_i",
    "japanese_japan": "what_is_it",
    "vietnamese_vietnam": "other_question",
    "arabic_morocco": "first_person_statement",
    "chinese_singapore": "third_person_description",
}
assert set(FOCAL_FORM) == set(EXEMPLAR_COMMUNITIES)

# Layout (points). Axes are placed manually so row geometry is exact.
LEFT_IN, RIGHT_IN, BOTTOM_IN, TOP_IN = 0.77, 0.10, 0.54, 0.03
DOT = 3.4  # swarm dot diameter (pt)
SWARM_HALF = 8.0  # max vertical swarm half-height (pt); denser piles overlap
SWARM_SHIFT = -2.0  # swarm centre below row centre, leaves a text lane above
TEXT_LANE = 8.0  # label baseline above row centre (pt)
XLIM = (-2.5, 103.0)


def swarm_offsets(x_pt: np.ndarray, first: int | None) -> np.ndarray:
    """Vertical offsets (pt) so dots at similar x do not overlap, capped at +-SWARM_HALF.

    ``first`` (index of a focal dot) is placed at offset 0. When a pile is too
    dense for the cap the remaining dots overlap (translucent) instead of
    growing the row.
    """
    order = ([first] if first is not None else []) + [
        i for i in np.argsort(x_pt, kind="stable") if i != first
    ]
    placed_x: list[float] = []
    placed_y: list[float] = []
    off = np.zeros(len(x_pt))
    cands = [0.0]
    for k in np.arange(0.5, SWARM_HALF + 0.01, 0.5):
        cands += [k, -k]
    for i in order:
        best, best_gap = 0.0, -1.0
        for y in cands:
            if not placed_x:
                best, best_gap = y, 99.0
                break
            gap = np.min(np.hypot(np.array(placed_x) - x_pt[i], np.array(placed_y) - y))
            if gap >= DOT:
                best, best_gap = y, gap
                break
            if gap > best_gap:
                best, best_gap = y, gap
        off[i] = best
        placed_x.append(x_pt[i])
        placed_y.append(best)
    return off


def main() -> None:
    apply_style()
    forms = classify_forms()
    shares = form_shares(forms)
    overall = shares.filter(pl.col("community") == OVERALL).row(0, named=True)
    comm = shares.filter(pl.col("community") != OVERALL)
    assert comm.height == 61 and int(overall["n"]) == forms.height

    rows = sorted(FORMS, key=lambda f: -overall[f])  # largest overall share on top
    n_rows = len(rows)

    fig = new_fig(COL_W, HEIGHT, layout=None)
    ax = fig.add_axes(
        (
            LEFT_IN / COL_W,
            BOTTOM_IN / HEIGHT,
            1 - (LEFT_IN + RIGHT_IN) / COL_W,
            1 - (BOTTOM_IN + TOP_IN) / HEIGHT,
        )
    )
    w_pt = ax.get_position().width * COL_W * 72
    h_pt = ax.get_position().height * HEIGHT * 72
    pt_per_pct = w_pt / (XLIM[1] - XLIM[0])
    pitch = h_pt / n_rows  # points per row
    to_row = 1.0 / pitch  # points -> row units

    ax.set_xlim(*XLIM)
    ax.set_ylim(n_rows - 0.5, -0.5)  # row 0 on top

    focal = {v: k for k, v in FOCAL_FORM.items()}
    codes = dict(zip(comm["community"], comm["code_label"]))

    for r, form in enumerate(rows):
        colour, marker = STYLE[form]
        vals = comm[form].to_numpy()
        names = comm["community"].to_list()
        fidx = names.index(focal[form]) if form in focal else None
        off = swarm_offsets(vals * pt_per_pct, fidx)
        y = r - (SWARM_SHIFT + off) * to_row  # data y grows downward; +off = up

        ctx = np.ones(len(vals), bool)
        if fidx is not None:
            ctx[fidx] = False
        ax.plot(
            vals[ctx],
            y[ctx],
            ls="",
            marker=marker,
            ms=DOT,
            mfc=colour,
            mec="none",
            alpha=0.55,
            zorder=2,
            clip_on=False,
        )
        if fidx is not None:
            ax.plot(
                vals[fidx],
                y[fidx],
                ls="",
                marker=marker,
                ms=DOT + 1.6,
                mfc=colour,
                mec=C["black"],
                mew=1.0,
                zorder=4,
                clip_on=False,
            )
            code = codes[names[fidx]].replace("(", "").replace(")", "")
            xv = vals[fidx]
            ha = "right" if xv > 80 else "center"
            ax.text(
                min(xv, 100.0) + (3.0 if ha == "right" else 0.0),
                r - TEXT_LANE * to_row,
                code,
                ha=ha,
                va="bottom",
                fontsize=7,
                color=C["ink"],
                zorder=5,
            )

        ov = overall[form]
        ax.plot(
            ov,
            r - SWARM_SHIFT * to_row,
            ls="",
            marker="D",
            ms=5.0,
            mfc=C["black"],
            mec="white",
            mew=0.7,
            zorder=6,
        )
        ax.text(
            ov,
            r - TEXT_LANE * to_row,
            f"{ov:.1f}" + (" overall" if r == 0 else ""),
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            color=C["black"],
            zorder=6,
        )

    ax.set_yticks(range(n_rows), [ROW_LABEL[f] for f in rows], linespacing=0.95)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of riddles (%)\nOne dot per community")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_bounds(0, 100)

    save(fig, STEM)


if __name__ == "__main__":
    main()
