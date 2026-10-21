"""Variance decomposition of riddle difficulty (replaces e_difficulty_variance_components).

One panel, two 100% stacked bars joined by a zoom connector:
* top    = share of total variance (model, model x facet, residual | benchmark block);
* bottom = the benchmark block rescaled to 100% of benchmark variance.
Each component keeps one colour + hatch in both bars; labels are direct (no legend).

Rounding rule: percentages are rounded to integers with the largest-remainder method so
that each bar sums to exactly 100 (segment widths use the unrounded shares).

Run from the repo root::

    uv run python -m analysis.figures.fig_variance
"""

from __future__ import annotations

import polars as pl
from matplotlib.patches import Polygon, Rectangle

from analysis.difficulty_decomposition.difficulty_variance_components import load_shares
from analysis.figures._common import COL_W, apply_style, new_fig, save
from analysis.style import STYLE as TARGET_STYLE
from analysis.style import VARIANCE

HEIGHT = 1.75

# Component -> (face colour, hatch, hatch/ink colour), shared across targets.
STYLE = VARIANCE
BENCH = ["Riddle", "Language", "Community", "Topic"]
GRAY = ["Model", "Model $\\times$ facet", "Residual"]


def round_to_100(values: list[float]) -> list[int]:
    """Largest-remainder rounding of ``values`` (summing to 100) to integers."""
    floors = [int(v) for v in values]
    order = sorted(range(len(values)), key=lambda i: values[i] - floors[i], reverse=True)
    for i in order[: 100 - sum(floors)]:
        floors[i] += 1
    return floors


def main() -> None:
    apply_style()
    shares = load_shares()
    all_share = {r["label"]: r["share"] * 100 for r in shares.iter_rows(named=True)}
    bench_share = {
        r["label"]: r["benchmark_share"] * 100
        for r in shares.filter(pl.col("benchmark_share").is_not_null()).iter_rows(named=True)
    }
    top_order = BENCH + GRAY
    top = [all_share[k] for k in top_order]
    bot = [bench_share[k] for k in BENCH]
    top_r = dict(zip(top_order, round_to_100(top), strict=True))
    bot_r = dict(zip(BENCH, round_to_100(bot), strict=True))
    split = sum(all_share[k] for k in BENCH)  # benchmark block width in the top bar

    fig = new_fig(COL_W, HEIGHT, layout=None)
    left, right, bottom, topm = 0.72, 0.03, 0.02, 0.0
    ax = fig.add_axes(
        (
            left / COL_W,
            bottom / HEIGHT,
            1 - (left + right) / COL_W,
            1 - (bottom + topm) / HEIGHT,
        )
    )
    y_top, y_bot, bh = 3.1, 0.0, 1.25
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.7, 5.45)
    ax.axis("off")

    def bar(y: float, keys: list[str], widths: list[float]) -> dict[str, tuple[float, float]]:
        spans, x = {}, 0.0
        for k, w in zip(keys, widths, strict=True):
            face, hatch, ink = STYLE[k]
            ax.add_patch(
                Rectangle(
                    (x, y - bh / 2),
                    w,
                    bh,
                    facecolor=face,
                    edgecolor=ink,
                    hatch=hatch,
                    linewidth=0,
                    zorder=2,
                )
            )
            ax.add_patch(
                Rectangle(
                    (x, y - bh / 2),
                    w,
                    bh,
                    facecolor="none",
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=3,
                )
            )
            spans[k] = (x, x + w)
            x += w
        return spans

    top_sp = bar(y_top, top_order, top)
    bot_sp = bar(y_bot, BENCH, bot)

    # zoom connector: benchmark block of the top bar -> full bottom bar
    ax.add_patch(
        Polygon(
            [
                (0, y_top - bh / 2),
                (split, y_top - bh / 2),
                (100, y_bot + bh / 2),
                (0, y_bot + bh / 2),
            ],
            closed=True,
            facecolor=TARGET_STYLE["connector_fill"],
            edgecolor="none",
            zorder=0,
        )
    )
    for xs, xe in ((0, 0), (split, 100)):
        ax.plot(
            [xs, xe],
            [y_top - bh / 2, y_bot + bh / 2],
            color=TARGET_STYLE["connector_edge"],
            lw=0.7,
            zorder=1,
        )

    fs = 7

    def inside(bars, y, k, color):
        a, b = bars[k]
        ax.text(
            (a + b) / 2,
            y,
            f"{k}\n{(top_r if bars is top_sp else bot_r)[k]}",
            ha="center",
            va="center",
            fontsize=fs,
            color=color,
            zorder=5,
        )

    def out(bars, k, xt, yt, ha, y_from, rad=0.0):
        a, b = bars[k]
        xc = (a + b) / 2
        val = (top_r if bars is top_sp else bot_r)[k]
        name = k
        ax.annotate(
            f"{name} {val}",
            xy=(xc, y_from),
            xytext=(xt, yt),
            ha=ha,
            va="center",
            fontsize=fs,
            zorder=5,
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.5,
                "color": "#444444",
                "shrinkA": 1.5,
                "shrinkB": 0,
                "connectionstyle": f"arc3,rad={rad}",
            },
        )

    # --- top bar labels
    inside(top_sp, y_top, "Riddle", "white")
    inside(top_sp, y_top, "Model", "#1A1A1A")
    up = y_top + bh / 2
    out(top_sp, "Language", 36.0, 4.35, "right", up)
    out(top_sp, "Community", 46.0, 5.1, "right", up)
    out(top_sp, "Topic", 49.0, 5.1, "left", up)
    out(top_sp, "Model $\\times$ facet", 97.0, 4.35, "right", up)
    out(top_sp, "Residual", 100.0, 1.85, "right", y_top - bh / 2)

    # --- bottom bar labels
    inside(bot_sp, y_bot, "Riddle", "white")
    inside(bot_sp, y_bot, "Language", "#1A1A1A")
    dn = y_bot - bh / 2
    out(bot_sp, "Community", 84.0, -1.25, "right", dn)
    out(bot_sp, "Topic", 100.0, -1.25, "right", dn)

    # direct row labels
    ax.text(-1.5, y_top, "Total\nvariance\n(%)", ha="right", va="center", fontsize=8)
    ax.text(-1.5, y_bot, "Benchmark\nvariance\n(%)", ha="right", va="center", fontsize=8)

    save(fig, "fig_variance")
    print("top rounded", top_r, sum(top_r.values()))
    print("bench rounded", bot_r, sum(bot_r.values()))
    print("raw top", {k: round(v, 2) for k, v in all_share.items()})
    print(
        "raw bench",
        {k: round(v, 2) for k, v in bench_share.items()},
        "split",
        round(split, 2),
    )


if __name__ == "__main__":
    main()
