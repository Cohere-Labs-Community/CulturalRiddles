"""Redesign of tab:judge_by_community: per-community judge kappa, sorted dot plot.

Message: LLM-human agreement on correctness is high for most communities and dips
for low-resource ones; the jury's lowest community is Swahili and, for the single
judges, Lumasaba.

Data: the files the table generator reads (``agreement_by_community.csv`` for the
rows, ``agreement_by_resourcedness.csv`` / ``agreement_summary.csv`` for the pooled
resource-level rows and the overall reference), so every dot equals a table cell.
Rows are sorted by the table's sort key, "Both" = the jury's two-judge rule (judges'
score when they agree, else the human label; an upper bound), the same rule and
numbers used by the generated claims (analysis.tables.macros, ``jury*`` macros). The two
single judges are shown beside it on the same rows.
"""

from __future__ import annotations

import polars as pl
from matplotlib.transforms import blended_transform_factory

from analysis.corpus import load_corpus_metadata
from analysis.figures._common import (
    FULL_W,
    RESOURCE,
    STYLE,
    C,
    apply_style,
    min_font_pt,
    new_fig,
    save,
)
from analysis.style import OVERVIEW
from analysis.tables.common import JUDGE_STUDY_DIR

apply_style()

JURY = "agree-or-human (zero)"
JUDGES = [
    (JURY, "Jury (both judges)"),
    ("deepseek-v4.1-flash|zero", "DeepSeek"),
    ("gemini-3.5-flash|zero", "Gemini"),
]
LEVELS = ["high", "medium", "low", "extremely-low"]
EDGE = STYLE["resource_edge"]  # dark outline keeps light resource markers visible
NUM_FS = 6.5  # point size of the numeric columns; rows are 0.1 in apart
WHISK = C["grey"]
conds = [j for j, _ in JUDGES]

# ---- data ----------------------------------------------------------------------
meta = pl.from_pandas(
    load_corpus_metadata()[["community", "language", "country_code", "language_resourcedness"]]
)
bycom = pl.read_csv(JUDGE_STUDY_DIR / "agreement_by_community.csv").filter(
    pl.col("condition").is_in(conds)
)
bylev = pl.read_csv(JUDGE_STUDY_DIR / "agreement_by_resourcedness.csv").filter(
    pl.col("condition").is_in(conds)
)
summ = pl.read_csv(JUDGE_STUDY_DIR / "agreement_summary.csv").filter(
    pl.col("condition").is_in(conds) & pl.col("level").str.starts_with("binary (score 3")
)
overall = {r["condition"]: r["kappa"] for r in summ.iter_rows(named=True)}
assert set(overall) == set(conds)

jury_k = bycom.filter(pl.col("condition") == JURY).select("community", jk="kappa")
comm = (
    meta.join(jury_k, on="community", how="inner")
    .sort("jk", descending=True)  # same order as the table's "Both" column
    .with_columns(label=pl.col("language") + " (" + pl.col("country_code") + ")")
)
assert comm.height == bycom["community"].n_unique() == 57

lookup = {}
for cond in conds:
    for r in bycom.filter(pl.col("condition") == cond).iter_rows(named=True):
        lookup[(cond, "comm", r["community"])] = (
            r["kappa"],
            r["low"],
            r["high"],
            r["human_acc"],
        )
    for r in bylev.filter(pl.col("condition") == cond).iter_rows(named=True):
        lookup[(cond, "level", r["language_resourcedness"])] = (
            r["kappa"],
            r["low"],
            r["high"],
            r["human_acc"],
        )
res_of = dict(zip(comm["community"], comm["language_resourcedness"]))

# Communities to name: per-panel lowest and highest (rounded ties all named).
focal = set()
focal_by_cond = {}
for cond in conds:
    ks = {c: round(lookup[(cond, "comm", c)][0], 2) for c in comm["community"]}
    lo, hi = min(ks.values()), max(ks.values())
    focal_by_cond[cond] = {c for c, v in ks.items() if v in (lo, hi)}
    focal |= focal_by_cond[cond]
print({c: sorted(v) for c, v in focal_by_cond.items()})

# ---- rows (y grows downward; plotted inverted) -----------------------------------
rows = []  # (y, kind, key, label)
y = 0.0
HEAD1 = y
y += 1.0
for lv in LEVELS:
    rows.append((y, "level", lv, RESOURCE[lv][0]))
    y += 1.0
y += 0.6
HEAD2 = y
y += 1.0
for r in comm.iter_rows(named=True):
    rows.append((y, "comm", r["community"], r["label"]))
    y += 1.0
YMAX = y - 0.5

PITCH = 0.1  # inches per row
TOP_M, BOT_M = 0.36, 0.42
H = YMAX * PITCH + TOP_M + BOT_M
fig = new_fig(FULL_W, H, layout=None)

LEFT, GAP, RIGHT = 1.30, 0.30, 0.06
KW = 1.02  # width of each kappa panel (inches)
widths = [KW, KW, KW, FULL_W - LEFT - RIGHT - 3 * GAP - 3 * KW]
axes, x0 = [], LEFT
for w in widths:
    axes.append(fig.add_axes((x0 / FULL_W, BOT_M / H, w / FULL_W, (H - TOP_M - BOT_M) / H)))
    x0 += w + GAP


def draw_point(ax, yy, est, lo, hi, level, big, ci=True):
    _, col, mk = RESOURCE[level]
    if ci:
        ax.plot(
            [lo, hi],
            [yy, yy],
            color=WHISK,
            lw=1.1 if big else 1.0,
            solid_capstyle="butt",
            zorder=2,
        )
    ax.plot(
        [est],
        [yy],
        ls="",
        marker=mk,
        ms=5.2 if big else 3.6,
        mfc=col,
        mec=EDGE,
        mew=0.5,
        zorder=3,
    )


for ax, (cond, name) in zip(axes[:3], JUDGES):
    ax.axvline(overall[cond], color=OVERVIEW["muted"], lw=1.0, ls=(0, (3, 2)), zorder=1)
    for yy, kind, key, _ in rows:
        est, lo, hi, _ = lookup[(cond, kind, key)]
        level = key if kind == "level" else res_of[key]
        draw_point(ax, yy, est, lo, hi, level, big=(kind == "level"))
        ax.text(
            1.04,
            yy,
            f"{est:.2f}",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=NUM_FS,
            color=C["ink"],
            fontweight="bold" if kind == "comm" and key in focal_by_cond[cond] else "normal",
        )
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_xlabel("Cohen\u2019s $\\kappa$")
    ax.annotate(
        name,
        (0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0, 11),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=7,
        fontweight="bold",
    )
    ax.annotate(
        f"overall {overall[cond]:.2f}",
        (overall[cond], 1.0),
        xycoords=ax.get_xaxis_transform(),
        xytext=(0, 2),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7,
    )

# human-scored correct share: why the intervals widen
axh = axes[3]
for yy, kind, key, _ in rows:
    est = lookup[(JURY, kind, key)][3] * 100
    level = key if kind == "level" else res_of[key]
    draw_point(axh, yy, est, 0, 0, level, big=(kind == "level"), ci=False)
    axh.text(
        72,
        yy,
        f"{est:.0f}",
        ha="right",
        va="center",
        fontsize=NUM_FS,
        color=C["ink"],
        fontweight="bold" if kind == "comm" and key in focal else "normal",
    )
axh.set_xlim(0, 72)
axh.set_xticks([0, 25, 50])
axh.set_xlabel("Correct (%)")
axh.annotate(
    "Human-scored",
    (0.0, 1.0),
    xycoords="axes fraction",
    xytext=(0, 11),
    textcoords="offset points",
    ha="left",
    va="bottom",
    fontsize=7,
    fontweight="bold",
)

for ax in axes:
    ax.set_ylim(YMAX, -0.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)

# row labels (shared rows)
ax0 = axes[0]
ax0.set_yticks([r[0] for r in rows])
ax0.set_yticklabels([r[3] for r in rows], fontsize=6.5)
ax0.tick_params(axis="y", left=True, labelleft=True, length=0, pad=3)
for t, (yy, kind, key, _) in zip(ax0.get_yticklabels(), rows):
    t.set_color(C["ink"])
    if kind == "comm" and key in focal:
        t.set_fontweight("bold")
trans = blended_transform_factory(fig.transFigure, ax0.transData)
for yy, txt in [
    (HEAD1, "Pooled by resource level"),
    (HEAD2, "Communities, sorted by jury $\\kappa$"),
]:
    ax0.text(
        0.012,
        yy,
        txt,
        transform=trans,
        ha="left",
        va="center",
        fontsize=7,
        fontweight="bold",
    )

print("H", round(H, 2), "min font", min_font_pt(fig))
save(fig, "fig_judge_by_community")
