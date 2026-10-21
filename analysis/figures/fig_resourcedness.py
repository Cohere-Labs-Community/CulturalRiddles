"""Redesign of fig:resourcedness: accuracy by language-resource level.

Message: pooled accuracy falls with resources (diamonds + connecting line), but
community means (dots) spread widely inside every level.

Every human-evaluated community is plotted and pooled (the "57" set in
``_resource_data``). Dots are a constant size: scored responses per community
only span 2.6-2.8k, so an area encoding would be invisible.
"""

from __future__ import annotations

import math

import numpy as np

from analysis.figures._common import COL_W, RESOURCE, C, apply_style, new_fig, save
from analysis.figures._resource_data import (
    ROW_ORDER,
    SUDANESE,
    SWAHILI,
    load_variant,
)

apply_style()

MS = 3.6  # community marker size (pt)
RADIUS_IN = MS / 2 / 72  # marker radius in inches (for collision tests)
XMAX = 70.0
POOLED_DY = -0.30  # pooled diamond sits below the swarm (row units)
SWARM_BASE = -0.10  # swarm baseline (row units)


def draw(variant: str, stem: str) -> None:
    community, summary = load_variant(variant)
    print(
        f"--- variant {variant}: {len(community)} points, {int((~community.hollow).sum())} pooled"
    )
    print(
        summary[
            [
                "communities",
                "scored_responses",
                "mean_community_accuracy",
                "median_community_accuracy",
                "pooled_accuracy",
                "community_ci_low",
                "community_ci_high",
                "between_community_sd",
            ]
        ]
        .round(2)
        .to_string()
    )

    W, H = COL_W, 2.45
    LEFT, RIGHT, TOP_M, BOT = 0.80, 0.08, 0.06, 0.40  # inches
    fig = new_fig(W, H, layout=None)
    ax = fig.add_axes((LEFT / W, BOT / H, (W - LEFT - RIGHT) / W, (H - BOT - TOP_M) / H))
    ax_w_in = W - LEFT - RIGHT
    n_rows = len(ROW_ORDER)
    row_in = (H - BOT - TOP_M) / n_rows
    rows_y = {g: n_rows - 1 - i for i, g in enumerate(ROW_ORDER)}  # high on top

    def swarm_up(values: np.ndarray) -> np.ndarray:
        """Collision-free one-sided beeswarm: offsets (row units) >= 0 above the baseline."""
        vin = values / XMAX * ax_w_in
        placed: list[tuple[float, float]] = []
        out = np.zeros(len(values))
        d = 2 * RADIUS_IN * 1.12
        for i in np.argsort(values):
            off = 0.0
            while not all(math.hypot(vin[i] - px, off - py) >= d for px, py in placed):
                off += 0.004
            placed.append((vin[i], off))
            out[i] = off
        return out / row_in

    swarm_pts: dict[str, tuple] = {}
    line_x, line_y = [], []
    for g in ROW_ORDER:
        _, color, marker = RESOURCE[g]
        y0 = rows_y[g]
        sub = community.loc[community["language_resourcedness"] == g]
        vals = sub["accuracy"].to_numpy()
        hollow = sub["hollow"].to_numpy()
        ys = y0 + SWARM_BASE + swarm_up(vals)
        swarm_pts[g] = (vals, ys, list(sub["community"]))
        ax.plot(
            vals[~hollow],
            ys[~hollow],
            linestyle="none",
            marker=marker,
            ms=MS,
            mfc=color,
            mec=C["ink"],
            mew=0.5,
            zorder=3,
            clip_on=False,
        )
        if hollow.any():
            ax.plot(
                vals[hollow],
                ys[hollow],
                linestyle="none",
                marker=marker,
                ms=MS,
                mfc="white",
                mec=C["ink"],
                mew=0.8,
                zorder=3,
                clip_on=False,
            )

        row = summary.loc[g]
        px, py = row["pooled_accuracy"], y0 + POOLED_DY
        line_x.append(px)
        line_y.append(py)
        ax.plot(
            [row["community_ci_low"], row["community_ci_high"]],
            [py, py],
            color="black",
            lw=1.3,
            solid_capstyle="butt",
            zorder=5,
        )
        for xe in (row["community_ci_low"], row["community_ci_high"]):
            ax.plot([xe, xe], [py - 0.07, py + 0.07], color="black", lw=1.0, zorder=5)
        ax.plot(
            px,
            py,
            marker="D",
            ms=7.0,
            mfc=color,
            mec="black",
            mew=1.1,
            linestyle="none",
            zorder=6,
        )
        ax.text(
            row["community_ci_high"] + 1.2,
            py,
            f"{px:.1f}",
            ha="left",
            va="center",
            fontsize=7,
            color="black",
            fontweight="bold",
            zorder=6,
        )

    ax.plot(line_x, line_y, color="black", lw=0.8, zorder=2)  # the decline

    def point(g, name):
        vals, ys, names = swarm_pts[g]
        j = names.index(name)
        return vals[j], ys[j]

    leader = {
        "arrowstyle": "-",
        "lw": 0.5,
        "color": C["ink"],
        "shrinkA": 0,
        "shrinkB": 2,
    }
    sx, sy = point("medium", SWAHILI)
    ax.annotate(
        "Swahili (Kenya)",
        xy=(sx, sy),
        xytext=(1.0, 2.40),
        fontsize=7,
        ha="left",
        va="center",
        color=C["ink"],
        zorder=7,
        arrowprops=leader,
    )
    ux, uy = point("high", SUDANESE)
    ax.annotate(
        "Sudanese Arabic",
        xy=(ux, uy),
        xytext=(ux + 3.0, uy + 0.38),
        fontsize=7,
        ha="left",
        va="center",
        color=C["ink"],
        zorder=7,
        arrowprops=leader,
    )

    ax.set_xlim(0, XMAX)
    ax.set_ylim(-0.55, n_rows - 0.35)
    ax.set_xticks(range(0, 71, 10))
    ax.set_xlabel("Accuracy (%)")
    ax.set_yticks([rows_y[g] for g in ROW_ORDER])
    ax.set_yticklabels(
        [
            f"{RESOURCE[g][0]}\n$n$={int(summary.loc[g, 'communities'])}, "
            f"$N$={summary.loc[g, 'scored_responses'] / 1000:.0f}k"
            for g in ROW_ORDER
        ],
        fontsize=7,
        linespacing=1.15,
    )
    ax.tick_params(axis="y", length=0, pad=4)
    ax.spines["left"].set_visible(False)
    ax.set_ylabel("")
    save(fig, stem)


draw("57", "fig_resourcedness")
