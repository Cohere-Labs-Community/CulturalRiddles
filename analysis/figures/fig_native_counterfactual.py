"""Without native speakers: accuracy if the LLM jury replaced every human label.

Reads ``analysis.llm_judge.native_counterfactual`` outputs.
Left: change in each model's accuracy (jury only minus current), models ranked
by current accuracy; the agreement rule with its 95% bootstrap interval, the
three other rules as small markers.
Right: change in each community's mean accuracy under the agreement rule
against its current accuracy, coloured and shaped by resource level.

Options: ``--out DIR`` writes ``<stem>.pdf/png`` to DIR instead of ``figures/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from analysis.corpus import load_corpus_metadata
from analysis.figures._common import (
    PREPRINT_W,
    RESOURCE,
    C,
    apply_style,
    new_fig,
    save,
)
from analysis.llm_judge.native_counterfactual import OUT_DIR
from analysis.style import MARKER_EDGE

apply_style()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path)
args = parser.parse_args()
STEM = "fig_native_counterfactual"
SZ = 7
INK = C["ink"]
# rule -> (label, colour, marker); the agreement rule is the focal series
RULES = {
    "agree": ("Both judges agree, else unscored", INK, "o"),
    "lower": ("Lower of the two scores", C["blue"], "s"),
    "gemini": ("Gemini judge alone", C["orange"], "^"),
    "deepseek": ("DeepSeek judge alone", C["purple"], "D"),
}
N_LABEL = 3  # communities labeled at each end of the shift

models = pl.read_csv(OUT_DIR / "model_accuracy.csv")
communities = pl.read_csv(OUT_DIR / "community_accuracy.csv").filter(pl.col("rule") == "agree")
order = models.filter(pl.col("rule") == "agree").sort("current", descending=True)["model"].to_list()
row = {m: len(order) - 1 - i for i, m in enumerate(order)}
metadata = load_corpus_metadata()
labels = dict(
    zip(
        metadata["community"],
        metadata["language_code"] + " (" + metadata["country_code"] + ")",
        strict=True,
    )
)

fig = new_fig(PREPRINT_W, 4.3)
left, right = fig.subplots(1, 2, width_ratios=(1.15, 1))

for rule, (label, colour, marker) in reversed(RULES.items()):
    frame = models.filter(pl.col("rule") == rule)
    y = [row[m] for m in frame["model"]]
    if rule == "agree":
        left.hlines(y, frame["diff_low"], frame["diff_high"], color=INK, lw=0.8, zorder=3)
        left.plot(frame["diff"], y, marker, color=INK, ms=3.4, ls="", zorder=4)
    else:
        left.plot(frame["diff"], y, marker, color=colour, ms=2.6, ls="", alpha=0.85, zorder=2)
left.axvline(0, color=C["grey"], lw=0.6, zorder=0)
left.set_yticks(range(len(order)), [f"{m}" for m in reversed(order)], fontsize=SZ)
left.tick_params(axis="y", length=0)
left.spines["left"].set_visible(False)
left.set_ylim(-0.6, len(order) - 0.4)
left.set_xlabel("Change in model accuracy (points)")
agree = models.filter(pl.col("rule") == "agree")
for m, d in zip(agree["model"], agree["diff"], strict=True):  # numbers for the black dots
    left.text(
        1.01,
        row[m],
        f"{d:+.1f}".replace("-", "\u2212"),
        transform=left.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=SZ - 0.5,
        color=INK,
    )
left.text(
    1.01,
    1.0,
    "Agree",
    transform=left.transAxes,
    ha="left",
    va="bottom",
    fontsize=SZ - 0.5,
    color=C["grey"],
)
handles = [
    left.plot([], [], m, color=c, ms=3.4 if k == "agree" else 2.6, ls="")[0]
    for k, (_, c, m) in RULES.items()
]
fig.legend(
    handles,
    [lab for lab, _, _ in RULES.values()],
    loc="outside lower center",
    ncol=2,
    fontsize=SZ,
    handletextpad=0.3,
)

for level, (level_label, colour, marker) in RESOURCE.items():
    frame = communities.filter(pl.col("resource") == level)
    right.vlines(
        frame["current"],
        frame["diff_low"],
        frame["diff_high"],
        color=colour,
        lw=0.6,
        alpha=0.7,
        zorder=2,
    )
    right.plot(
        frame["current"],
        frame["diff"],
        marker,
        color=colour,
        mec=MARKER_EDGE,
        mew=0.4,
        ms=4,
        ls="",
        label=level_label,
        zorder=3,
    )
right.axhline(0, color=C["grey"], lw=0.6, zorder=0)
ranked = communities.sort("diff")
LABEL_ROOM = 25  # accuracy points a label needs to the right of its marker
for frame in (ranked.head(N_LABEL), ranked.tail(N_LABEL)):
    points = frame.select("community", "current", "diff").rows()
    xs = [x for _, x, _ in points]
    for name, x, d in points:
        crowded = any(0 < other - x < LABEL_ROOM for other in xs)
        right.annotate(
            labels[name],
            (x, d),
            xytext=(-3 if crowded else 3, 0),
            textcoords="offset points",
            ha="right" if crowded else "left",
            fontsize=SZ - 0.5,
            va="center",
        )
right.set_xlabel("Current community accuracy (%)")
right.set_ylabel("Change, agreement rule (points)")
right.set_xlim(0, 70)
right.legend(
    title="Resource level",
    title_fontsize=SZ,
    fontsize=SZ,
    loc="upper left",
    handletextpad=0.2,
    borderaxespad=0.2,
)

if args.out:
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / f"{STEM}.pdf")
    fig.savefig(args.out / f"{STEM}.png", dpi=450)
else:
    save(fig, STEM)
