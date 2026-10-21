"""Scoring pipeline at 6.50 in text width: routes and final scores.

The column version (``fig_pipeline``) at full width, with every count
and share labeled. Ribbons are coloured by route on the left (rules, human,
jury) and by outcome on the right (``OUTCOME`` in ``_common``).
"""

from __future__ import annotations

from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MPath

from analysis.figures._common import OUTCOME, PREPRINT_W, C, apply_style, new_fig, save
from analysis.figures._pipeline_data import (
    N,
    flow,
    n_human,
    n_jury,
    n_models,
    n_riddles,
    n_rules,
    n_unsc,
    pct,
    tot,
)
from analysis.style import ROUTE

# ----------------------------------------------------------------------------- style
apply_style()
# Drawn in the slide's 12.3 x 4.4 coordinate frame, scaled to the text width.
FIG_W, FIG_H = 12.3, 4.4
SCALE = PREPRINT_W / FIG_W
FS = 14 * SCALE
fig = new_fig(PREPRINT_W, FIG_H * SCALE, layout=None)
ax = fig.add_axes((0, 0, 1, 1))
ax.set_xlim(0, FIG_W)
ax.set_ylim(FIG_H, 0)  # inches, y down
ax.axis("off")

H = 2.55  # inches for N responses
k = H / N

SCORE = [  # key, label, colour, hatch
    (3.0, "3  correct", OUTCOME["correct"][1], OUTCOME["correct"][2]),
    (2.0, "2  partially correct", OUTCOME["partial"][1], OUTCOME["partial"][2]),
    (1.0, "1  admits not knowing", OUTCOME["declined"][1], OUTCOME["declined"][2]),
    (0.0, "0  incorrect", OUTCOME["incorrect"][1], OUTCOME["incorrect"][2]),
]
UNSC = (C["lightgrey"], "---")

XA, XB, XD = 0.05, 4.70, 7.40
WA, WB, WD = 0.25, 0.32, 0.32
Y0 = 0.90


def ribbon(x0, y0a, y0b, x1, y1a, y1b, **kw):
    dx = (x1 - x0) * 0.5
    verts = [
        (x0, y0a),
        (x0 + dx, y0a),
        (x1 - dx, y1a),
        (x1, y1a),
        (x1, y1b),
        (x1 - dx, y1b),
        (x0 + dx, y0b),
        (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        MPath.MOVETO,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.LINETO,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MPath(verts, codes), lw=0, **kw))


def node(x, y, w, h, fc, hatch="", hc=None, lw=0.5):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=fc,
            edgecolor=hc or C["ink"],
            hatch=hatch,
            lw=1.0 * SCALE,
            zorder=4,
            joinstyle="miter",
        )
    )
    # crisp outline in ink so thin / pale bars stay visible
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor="none", edgecolor=C["ink"], lw=1.2 * SCALE, zorder=5)
    )


# ----------------------------------------------------------------------------- column A
yA = Y0 + 0.10
node(XA, yA, WA, N * k, C["ink"])
ax.text(
    XA,
    yA - 0.075 - 0.25,
    f"All model responses · {N:,}",
    fontsize=FS + 0.4,
    fontweight="bold",
    ha="left",
    va="bottom",
)
ax.text(
    XA,
    yA - 0.075,
    f"{n_models} models × {n_riddles:,} riddles",
    fontsize=FS,
    ha="left",
    va="bottom",
)

# ----------------------------------------------------------------------------- column B (routes)
routes = [("rules", n_rules), ("human", n_human), ("jury", n_jury)]
GAP_B = 0.10
yB, yAcur, y = {}, {}, Y0
ya = yA
for key, n in routes:
    yB[key], yAcur[key] = y, ya
    y += n * k + GAP_B
    ya += n * k
for key, n in routes:
    col, hatch, hc, al = ROUTE[key]
    ribbon(
        XA + WA,
        yAcur[key],
        yAcur[key] + n * k,
        XB,
        yB[key],
        yB[key] + n * k,
        facecolor=col,
        alpha=al,
        zorder=1,
    )
    node(XB, yB[key], WB, n * k, col, hatch, hc)

# ribbon-internal labels (ink text on tinted ribbons)
LH = 0.112  # line height, in


def block(key, lines, x=0.45, anchor="mid"):
    n = dict(routes)[key]
    ymid = (yAcur[key] + yB[key]) / 2 + n * k / 2
    ytop = (yAcur[key] + yB[key]) / 2
    if anchor == "mid":
        y_first = ymid - LH * (len(lines) - 1) / 2
    else:
        y_first = ytop + 0.07
    for i, (txt, bold) in enumerate(lines):
        ax.text(
            x,
            y_first + i * LH,
            txt,
            fontsize=FS,
            fontweight="bold" if bold else "normal",
            ha="left",
            va="center",
            zorder=3,
            color=C["ink"],
        )


block("rules", [(f"Rules · {n_rules:,} · {pct(n_rules)}", True)])
block("human", [(f"Native speakers · {n_human:,} · {pct(n_human)}", True)])
block("jury", [(f"LLM jury · {n_jury:,} · {pct(n_jury)}", True)])

# ----------------------------------------------------------------------------- column D (scores)
GAP_D = 0.21
dnodes = {}  # key -> (y_top, height)
y = Y0
for s, lab, col, hatch in SCORE:
    h = tot[s] * k
    dnodes[s] = (y, h)
    y += h + GAP_D
dnodes["unscored"] = (y, n_unsc * k)

# B -> D ribbons, outcome colours
dcursor = {key: dnodes[key][0] for key in dnodes}
src_cursor = dict(yB)
for key, n in routes:
    for s, lab, col, hatch in SCORE:
        v = int(flow.loc[key, s]) if key in flow.index else 0
        if v == 0:
            continue
        h = v * k
        ribbon(
            XB + WB,
            src_cursor[key],
            src_cursor[key] + h,
            XD,
            dcursor[s],
            dcursor[s] + h,
            facecolor=col,
            alpha=0.60,
            zorder=2,
        )
        src_cursor[key] += h
        dcursor[s] += h
    if key == "jury":
        h = n_unsc * k
        ribbon(
            XB + WB,
            src_cursor[key],
            src_cursor[key] + h,
            XD,
            dcursor["unscored"],
            dcursor["unscored"] + h,
            facecolor=UNSC[0],
            alpha=0.9,
            zorder=2,
        )

for s, lab, col, hatch in SCORE:
    yt, h = dnodes[s]
    node(XD, yt, WD, h, col, hatch, C["ink"] if hatch else None)
yt, h = dnodes["unscored"]
node(XD, yt, WD, h, "white", UNSC[1], C["grey"])

# D labels: name | count | % of all responses
X_NAME, X_CNT, X_PCT = XD + WD + 0.12, 10.75, 12.2
for s, lab, col, hatch in SCORE:
    yt, h = dnodes[s]
    yc = yt + h / 2
    ax.text(X_NAME, yc, lab, fontsize=FS, va="center", ha="left")
    ax.text(X_CNT, yc, f"{tot[s]:,}", fontsize=FS, va="center", ha="right")
    ax.text(X_PCT, yc, pct(tot[s]), fontsize=FS, va="center", ha="right")
yt, h = dnodes["unscored"]
yc = yt + h / 2
ax.text(X_NAME, yc, "unscored", fontsize=FS, va="center", ha="left")
ax.text(X_CNT, yc, f"{n_unsc:,}", fontsize=FS, va="center", ha="right")
ax.text(X_PCT, yc, pct(n_unsc), fontsize=FS, va="center", ha="right")

save(fig, "fig_pipeline_wide")
