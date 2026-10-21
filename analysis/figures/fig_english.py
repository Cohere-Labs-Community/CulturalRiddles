"""Effect of translating riddles into English, for five hard communities.

One panel: original (hollow circle) -> English (filled square) accuracy per community,
with the paired gain and its 95% riddle-bootstrap CI printed in a column at the right.
Blue solid connectors mark gains whose CI excludes zero; vermillion dashed connectors
mark gains whose CI includes zero. Dark rows use every riddle; light rows drop the riddles
whose translation reveals the answer and are drawn only where the two sets differ.
"""

from __future__ import annotations

import matplotlib as mpl
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from analysis.corpus import load_corpus_metadata
from analysis.figures._common import COL_W, C, apply_style, new_fig, save
from analysis.llm_judge.english_condition import OUTPUT_DIR, high_resource_accuracy

apply_style()

COLUMNS = (
    "paired_original",
    "paired_english",
    "paired_delta",
    "paired_delta_low",
    "paired_delta_high",
)
LANGUAGES = dict(load_corpus_metadata()[["community", "language"]].itertuples(index=False)) | {
    "all": "All"
}


def results(name: str) -> tuple[dict[str, tuple[float, ...]], dict[str, int]]:
    """Read one comparison.

    Returns:
        Language -> (original, English, gain, gain low, gain high), and
        language -> scored responses.
    """
    frame = pl.read_csv(OUTPUT_DIR / name).with_columns(
        language=pl.col("community").replace_strict(LANGUAGES)
    )
    values = {
        row[0]: tuple(float(v) for v in row[1:])
        for row in frame.select("language", *COLUMNS).iter_rows()
    }
    return values, dict(frame.select("language", "responses").iter_rows())


ALL, ALL_N = results("accuracy_by_community.csv")
WITHOUT, WITHOUT_N = results("accuracy_by_community_without_answer_in_riddle.csv")
NOREV = {name: WITHOUT[name] for name in WITHOUT if WITHOUT_N[name] != ALL_N[name]}
N_RIDDLES = ALL_N["All"] // 5
HIGH_RES = high_resource_accuracy()

ROWS = ["Buginese", "Lumasaba", "Yoruba", "Mongondow", "Sudanese Arabic", "All"]
YPOS = {n: i for i, n in enumerate(ROWS)}
YPOS["All"] += 0.7  # separate the pooled row
DY = 0.2  # offset of the two row sets within a two-set row
FS = 6.5
INK, GREY = C["ink"], C["grey"]
POS, NEG = C["blue"], C["vermillion"]
SOLID, DASHED = "-", (0, (2.2, 1.6))

# ---- geometry (inches) ----------------------------------------------------------
W, H = COL_W, 2.62
AX_L, AX_W = 0.93, 1.00
AX_B, AX_H = 0.50, 1.55
X_NUM = AX_L + AX_W + 0.06  # left edge of the gain column
fig = new_fig(W, H, layout=None)
ax = fig.add_axes((AX_L / W, AX_B / H, AX_W / W, AX_H / H))
ax.set_xlim(0, 72)
ax.set_ylim(YPOS["All"] + 0.6, -1.0)
ax.set_yticks([YPOS[n] for n in ROWS], ROWS)
ax.get_yticklabels()[-1].set_fontweight("bold")
ax.set_xticks([0, 20, 40, 60])
ax.set_xlabel("Accuracy (%)")

# reference: accuracy of high-resource communities
ax.axvline(HIGH_RES, color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
ax.text(
    HIGH_RES - 1.5,
    -0.95,
    f"high-resource {HIGH_RES:.0f}%",
    fontsize=FS,
    color=GREY,
    ha="right",
    va="top",
)


def lighten(hex_colour, f=0.5):
    """Blend a colour toward white (opaque, so overlapping marks do not show through)."""
    rgb = mpl.colors.to_rgb(hex_colour)
    return tuple(f * c + (1 - f) for c in rgb)


def signed(value: float) -> str:
    """One decimal with an explicit sign and a true minus."""
    return f"{value:+.1f}".replace("-", "\u2212")


def draw(name, data, light):
    """Draw one dumbbell and print its gain with 95% CI."""
    orig, eng, gain, lo, hi = data
    significant = lo > 0
    col, ls = (POS, SOLID) if significant else (NEG, DASHED)
    if light:
        col = lighten(col)
    y = YPOS[name] + ((DY if light else -DY) if name in NOREV else 0)
    ms = 3.6 if light else 4.4
    ax.plot(
        [orig, eng],
        [y, y],
        color=col,
        lw=1.0 if light else 1.4,
        ls=ls,
        zorder=2,
        solid_capstyle="butt",
    )
    ax.plot(orig, y, "o", mfc="white", mec=col, mew=1.1, ms=ms, zorder=3)
    ax.plot(eng, y, "s", mfc=col, mec=col, mew=1.1, ms=ms, zorder=3)
    ax.text(
        X_NUM / W,
        y,
        f"{signed(gain)} [{signed(lo).lstrip('+')}, {signed(hi).lstrip('+')}]",
        transform=blended_transform_factory(fig.transFigure, ax.transData),
        fontsize=FS,
        color=GREY if light else INK,
        fontweight="bold" if significant and not light else "normal",
        ha="left",
        va="center",
    )


for n, d in ALL.items():
    draw(n, d, light=False)
for n, d in NOREV.items():
    draw(n, d, light=True)

# ---- headers: direct keys, no boxes ---------------------------------------------
top = (AX_B + AX_H) / H
row1, row2, row3 = top + 0.10 / H, top + 0.25 / H, top + 0.40 / H
dy = 0.04 / H


def glyph(x_in, row, **kw):
    fig.add_artist(Line2D([x_in / W], [row + dy], transform=fig.transFigure, **kw))


def segment(x_in, row, **kw):
    fig.add_artist(
        Line2D([x_in / W, (x_in + 0.18) / W], [row + dy] * 2, transform=fig.transFigure, **kw)
    )


def head(x_in, row, text, **kw):
    fig.text(x_in / W, row, text, fontsize=FS, ha="left", va="bottom", **kw)


# row 3: the two ends of each dumbbell
glyph(0.04, row3, ls="", marker="o", mfc="white", mec=INK, mew=1.1, ms=4.4)
head(0.12, row3, "original")
glyph(0.74, row3, ls="", marker="s", color=INK, ms=4.4)
head(0.82, row3, "English")
# row 2: the two riddle sets
segment(0.03, row2, color=INK, lw=1.4)
head(0.25, row2, f"all {N_RIDDLES} riddles")
segment(1.08, row2, color=GREY, lw=1.0)
head(1.30, row2, "w/o revealing", color=GREY)
# row 1: significance of the gain
segment(0.03, row1, color=POS, lw=1.4, ls=SOLID)
head(0.25, row1, "gain CI > 0")
segment(1.08, row1, color=NEG, lw=1.4, ls=DASHED)
head(1.30, row1, "CI includes 0")
# gain column header
head(X_NUM, row2, "Gain (pts)")
head(X_NUM, row1, "[95% CI]")

save(fig, "fig_english")
