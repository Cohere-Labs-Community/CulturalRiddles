"""Cultures figure: raincloud per community (8 easiest | 8 hardest), density coloured by language resource level,
dots = model configurations coded by human-review share (circle / triangle / x)."""

from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde

from analysis.figures._common import FULL_W, RESOURCE, STYLE, apply_style, new_fig, save
from analysis.figures._community_data import (
    HIGH_REVIEW_COVERAGE,
    LOW_REVIEW_COVERAGE,
    top_bottom_selection,
)

N = 8
GAP = 1.0
INK = "#222222"
MAX_W = 0.42  # half-density width at the tallest peak (x units)
DOT = 3.4

selected, values, coverage, _ = top_bottom_selection(N)
levels = selected["language_resourcedness"].tolist()
pos = np.arange(2 * N, dtype=float)
pos[N:] += GAP
labels = selected["language_code"] + "\n(" + selected["country_code"] + ")"

# --- Summary values for the selected communities ----------------------------
for i, (v, lv) in enumerate(zip(values, levels)):
    print(
        f"{selected.culture[i]:<28}{lv:<14} mean {v.mean():5.1f} "
        f">20: {(v > 20).sum():2d} <10: {(v < 10).sum():2d} max {v.max():5.1f}"
    )

# --- densities, area-normalised per community, common scale ------------------
grid = np.linspace(0, 100, 201)
dens = []
for v in values:
    d = gaussian_kde(v, bw_method=0.35)(grid)
    d = d / d.max()
    dens.append(
        np.where((grid <= v.max()) & (grid >= v.min()), d, 0.0)
    )  # cut at the data range  # shape only: every cloud has the same peak width
scale = MAX_W

apply_style()
fig = new_fig(FULL_W, 2.55, layout=None)
fig.subplots_adjust(left=0.075, right=0.995, bottom=0.125, top=0.90)
ax = fig.add_subplot()
ax.set_xlim(-0.7, pos[-1] + 0.7)
ax.set_ylim(0, 100)
fig.canvas.draw()
ax_h = ax.get_window_extent().height / fig.dpi
ax_w = ax.get_window_extent().width / fig.dpi
x_per_in = (ax.get_xlim()[1] - ax.get_xlim()[0]) / ax_w


def swarm(y, cap_x):
    """Deterministic beeswarm offsets (x data units), |offset| <= cap_x."""
    d_in = DOT * 1.05 / 72
    y_in = np.asarray(y) / 100 * ax_h
    cap = cap_x / x_per_in
    cands = [0.0] + [s * k * d_in / 2 for k in range(1, 60) for s in (1, -1)]
    cands = [c for c in cands if abs(c) <= cap]
    placed: list[tuple[float, float]] = []
    off = np.zeros(len(y_in))
    for i in np.argsort(y_in):
        best, best_gap = 0.0, -1.0
        for x in cands:
            g = min(
                [((x - px) ** 2 + (y_in[i] - py) ** 2) ** 0.5 for px, py in placed],
                default=9,
            )
            if g >= d_in:
                best = x
                break
            if g > best_gap:
                best, best_gap = x, g
        off[i] = best
        placed.append((best, y_in[i]))
    return off * x_per_in


# review-share classes: shape + hue (vermillion > reddish-purple > orange, also dark -> light)
CLASSES = {
    "high": {"marker": "o", "ms": DOT, "mfc": "#555555", "mec": "none", "mew": 0.0},
    "mid": {
        "marker": "^",
        "ms": DOT + 0.3,
        "mfc": "#9A9A9A",
        "mec": "none",
        "mew": 0.0,
    },
    "low": {
        "marker": "x",
        "ms": DOT - 0.2,
        "mfc": "none",
        "mec": "#808080",
        "mew": 0.8,
    },
}
MEAN_FILL = STYLE["mean"]


def mean_bar(x, y, w=0.4, axis=None, **kw):
    a = axis or ax
    a.plot([x - w / 2, x + w / 2], [y, y], color=INK, lw=4.0, solid_capstyle="butt", **kw)
    a.plot(
        [x - w / 2, x + w / 2],
        [y, y],
        color=MEAN_FILL,
        lw=2.6,
        solid_capstyle="butt",
        **kw,
    )


def review_class(c):
    return np.where(
        c >= HIGH_REVIEW_COVERAGE,
        "high",
        np.where(c >= LOW_REVIEW_COVERAGE, "mid", "low"),
    )


def darker(hexc, f=0.6):
    r, g, b = (int(hexc[i : i + 2], 16) for i in (1, 3, 5))
    return f"#{int(r * f):02x}{int(g * f):02x}{int(b * f):02x}"


for i, (v, cov, dd, lv) in enumerate(zip(values, coverage, dens, levels)):
    _, colour, _g = RESOURCE[lv]
    x0 = pos[i] + 0.05
    ax.fill_betweenx(grid, x0, x0 + dd * scale, facecolor=colour, edgecolor="none", zorder=1)
    ax.plot(
        np.where(dd > 0, x0 + dd * scale, np.nan),
        grid,
        color=darker(colour),
        lw=0.6,
        zorder=1.5,
    )
    off = swarm(v, 0.22)
    cls = review_class(cov)
    for k, sty in CLASSES.items():
        m = cls == k
        if m.any():
            ax.plot(
                pos[i] - 0.28 + off[m],
                v[m],
                linestyle="",
                zorder=3,
                clip_on=False,
                marker=sty["marker"],
                markersize=sty["ms"],
                markerfacecolor=sty["mfc"],
                markeredgecolor=sty["mec"],
                markeredgewidth=sty["mew"],
            )
    mean_bar(pos[i] + 0.06, v.mean(), zorder=5)

# group headers (ink) over a thin neutral rule
for sl, name in (
    (slice(0, N), "8 easiest communities"),
    (slice(N, 2 * N), "8 hardest communities"),
):
    x0, x1 = pos[sl][0] - 0.5, pos[sl][-1] + 0.6
    ax.plot(
        [x0, x1],
        [104, 104],
        color="#8A8A8A",
        lw=0.8,
        clip_on=False,
        solid_capstyle="butt",
    )
    ax.text(
        (x0 + x1) / 2,
        105.5,
        name,
        color=INK,
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        clip_on=False,
    )

ax.set_xticks(pos)
ax.set_xticklabels(labels, fontsize=7)
ax.tick_params(axis="x", length=0, pad=2)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("Accuracy (%)")
ax.spines[["top", "right"]].set_visible(False)

# --- keys: unboxed, in the empty upper area of the hardest-communities axes (xlim/slots unchanged) ----
fig.canvas.draw()
rend = fig.canvas.renderer  # ty: ignore[unresolved-attribute]
inv = ax.transData.inverted()


def width_units(txt, bold=False):
    t = ax.text(0, 0, txt, fontsize=6.5, fontweight="bold" if bold else "normal")
    w = t.get_window_extent(rend)
    t.remove()
    return inv.transform((w.x1, 0))[0] - inv.transform((w.x0, 0))[0]


KX = pos[N] - 0.45
ROWS = [99, 92, 85, 78]


def title(y, txt):
    ax.text(KX, y, txt, fontsize=6.5, fontweight="bold", color=INK, va="center", ha="left")


def items(y, entries):
    x = KX
    for draw, txt in entries:
        draw(x + 0.14, y)
        ax.text(x + 0.32, y, txt, fontsize=6.5, va="center", ha="left")
        x += 0.32 + width_units(txt) + 0.25


def dot_key(k):
    sty = CLASSES[k]
    return lambda x, y: ax.plot(
        x,
        y,
        linestyle="",
        marker=sty["marker"],
        markersize=sty["ms"],
        markerfacecolor=sty["mfc"],
        markeredgecolor=sty["mec"],
        markeredgewidth=sty["mew"],
    )


def swatch_key(k):
    c = RESOURCE[k][1]
    return lambda x, y: ax.plot(
        x,
        y,
        linestyle="",
        marker="s",
        markersize=6.5,
        markerfacecolor=c,
        markeredgecolor=darker(c),
        markeredgewidth=0.5,
    )


title(ROWS[0], "Human-reviewed share of answers")
items(
    ROWS[1],
    [
        (dot_key("high"), "≥50%"),
        (dot_key("mid"), "20–50%"),
        (dot_key("low"), "<20%"),
        (lambda x, y: mean_bar(x, y, w=0.28), "mean"),
    ],
)
title(ROWS[2], "Language resource level")
items(
    ROWS[3],
    [(swatch_key(k), RESOURCE[k][0]) for k in ("high", "medium", "low", "extremely-low")],
)

save(fig, "fig_cultures")
