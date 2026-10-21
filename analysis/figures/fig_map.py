"""Proportional-symbol map of languages per covered country.

``fig_map`` fits one column; ``fig_map_wide`` fills the 6.50 in text width.

Circle area AND colour (sequential ramp) both encode the number of distinct
languages per country (1..10). Data: ``country_language_coverage`` on
``data/canonical/corpus_metadata.csv``; shapes: vendored Natural Earth 110m GeoJSON.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Polygon

from analysis.corpus import load_corpus_metadata
from analysis.figures._common import COL_W, PREPRINT_W, C, apply_style, new_fig, save
from analysis.figures._map_data import (
    GEOMETRY_COUNTRY_ALIASES,
    SMALL_COUNTRY_POINTS,
    country_language_coverage,
    geometry_rings,
    load_country_features,
)
from analysis.style import MAP_RAMP

LON0, LON1 = -170.0, 180.0


@dataclass(frozen=True)
class Layout:
    """Canvas, symbol, and label geometry of one map variant.

    ``labels`` maps a country to the (lon, lat) of its direct label;
    ``key`` is the (lon, lat) of the size key and its x offsets.
    """

    stem: str
    width: float
    lat: tuple[float, float]
    d1_pt: float
    font: float
    line: float
    pad: float
    labels: dict[str, tuple[float, float]]
    key: tuple[float, float, tuple[float, float, float]]

    @property
    def height(self) -> float:
        """Equirectangular height: one degree of longitude equals one of latitude."""
        return self.width * (self.lat[1] - self.lat[0]) / (LON1 - LON0)

    def radius_deg(self, n: np.ndarray) -> np.ndarray:
        """Circle radius in map degrees; area ∝ n."""
        return self.d1_pt * np.sqrt(n) / 72 / 2 * (LON1 - LON0) / self.width


SCALE_WIDE = PREPRINT_W / 12.3
LAYOUTS = (
    Layout(
        "fig_map",
        COL_W,
        (-46.0, 74.0),
        3.6,
        6.5,
        0.35,
        0.5,
        {
            "India": (70, 9),
            "Uganda": (24, -22),
            "Luxembourg": (-20, 66),
            "Cuba": (-92, 8),
        },
        (-164.0, -36.0, (0, 26, 56)),
    ),
    Layout(
        "fig_map_wide",
        PREPRINT_W,
        (-50.0, 78.0),
        12.0 * SCALE_WIDE,
        14.0 * SCALE_WIDE,
        1.0 * SCALE_WIDE,
        0.4,
        {
            "India": (58, 4),
            "Uganda": (26, -24),
            "Luxembourg": (-14, 70),
            "Cuba": (-92, 6),
        },
        (-160.0, -40.0, (0, 26, 58)),
    ),
)
RAMP = MAP_RAMP
LAND = "#E6E6E6"

# Metropolitan / main-body anchor points where the Natural Earth label point is
# misleading (overseas territories) or too crowded. (lon, lat)
ANCHOR = {
    "France": (2.5, 46.6),
    "Spain": (-3.7, 40.0),
    "Portugal": (-8.0, 39.6),
    "Netherlands": (5.5, 52.2),
    "Norway": None,
    "India": (78.5, 21.5),
    "Pakistan": (69.5, 30.0),
    "Nepal": (84.0, 28.3),
    "Bangladesh": (90.2, 23.8),
    "Sri Lanka": (80.7, 7.6),
    "Canada": (-100.0, 58.0),
    "United States": (-98.0, 39.0),
}


def relax(pos: np.ndarray, rad: np.ndarray, home: np.ndarray, pad: float = 0.5) -> np.ndarray:
    """Deterministic repulsion so circles do not overlap; weak pull to home."""
    pos = pos.copy()
    for _ in range(400):
        move = np.zeros_like(pos)
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = pos[i] - pos[j]
                dist = np.hypot(*d)
                need = rad[i] + rad[j] + pad
                if dist < need:
                    u = d / dist if dist > 1e-9 else np.array([1.0, 0.0])
                    push = (need - dist) / 2
                    move[i] += u * push
                    move[j] -= u * push
        move += 0.02 * (home - pos)
        pos += move * 0.5
    return pos


def draw(layout: Layout) -> None:
    """Draw and save one map variant."""
    cov = country_language_coverage(load_corpus_metadata()).reset_index(drop=True)
    langs = dict(zip(cov["country"], cov["languages"], strict=True))

    # --- geometry -----------------------------------------------------------
    patches, anchors = [], {}
    for feat in load_country_features():
        props = feat["properties"]
        name = GEOMETRY_COUNTRY_ALIASES.get(props["ADMIN"], props["ADMIN"])
        if name == "Antarctica":
            continue
        if name in langs:
            anchors[name] = (float(props["LABEL_X"]), float(props["LABEL_Y"]))
        for ring in geometry_rings(feat["geometry"]):
            xy = np.asarray(ring, dtype=float)[:, :2]
            if len(xy) >= 3:
                patches.append(Polygon(xy, closed=True))
    for name, pt in SMALL_COUNTRY_POINTS.items():
        if name in langs:
            anchors[name] = pt
    anchors.update({k: v for k, v in ANCHOR.items() if v and k in langs})
    missing = set(langs) - set(anchors)
    assert not missing, missing

    names = sorted(langs, key=lambda c: (-langs[c], c))  # big first
    n = np.array([langs[c] for c in names])
    home = np.array([anchors[c] for c in names])
    rad = layout.radius_deg(n)
    pos = relax(home, rad, home, layout.pad)

    # --- draw -----------------------------------------------------------------
    fig = new_fig(layout.width, layout.height, layout=None)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(LON0, LON1)
    ax.set_ylim(*layout.lat)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.add_collection(PatchCollection(patches, facecolor=LAND, edgecolor="none", zorder=1))

    def color(v):
        return RAMP((np.asarray(v) - 1) / 9)

    for nm, (x, y), r, v in zip(names, pos, rad, n, strict=True):
        ax.add_patch(
            Circle(
                (x, y),
                r,
                facecolor=color(v),
                edgecolor=C["ink"],
                linewidth=layout.line,
                zorder=3,
            )
        )

    # direct labels (+ thin leaders) for the countries named in the text
    def label(name, xy, text, ha="center", va="center"):
        i = names.index(name)
        ax.annotate(
            text,
            pos[i],
            xytext=xy,
            fontsize=layout.font,
            ha=ha,
            va=va,
            color=C["ink"],
            zorder=5,
            arrowprops={
                "arrowstyle": "-",
                "lw": layout.line,
                "color": C["ink"],
                "shrinkA": 1,
                "shrinkB": r_pt(name),
            },
        )

    def r_pt(name):
        return layout.d1_pt * np.sqrt(langs[name]) / 2

    label("India", layout.labels["India"], "India", ha="right")
    i = names.index("India")
    ax.text(
        pos[i][0],
        pos[i][1],
        "10",
        fontsize=layout.font,
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        zorder=6,
    )
    for name in ("Uganda", "Luxembourg", "Cuba"):
        label(name, layout.labels[name], name)

    # inline size key: same area + colour mapping, bottom-left (empty Pacific)
    kx, ky, offsets = layout.key
    for v, dx in zip((1, 5, 10), offsets, strict=True):
        x = kx + dx
        ax.add_patch(
            Circle(
                (x, ky),
                float(layout.radius_deg(np.asarray(v))),
                facecolor=color(v),
                edgecolor=C["ink"],
                linewidth=layout.line,
                zorder=3,
            )
        )
        ax.text(
            x + float(layout.radius_deg(np.asarray(v))) + 2.5,
            ky,
            str(v),
            fontsize=layout.font,
            ha="left",
            va="center",
            color=C["ink"],
        )

    print(
        f"{len(names)} countries; max {n.max()} ({names[0]}); "
        f"multi-language countries: {sorted(c for c in names if langs[c] > 1)}"
    )
    moved = np.hypot(*(pos - home).T) * 1.0
    print(
        "max nudge (deg):",
        round(float(moved.max()), 2),
        "median:",
        round(float(np.median(moved)), 2),
    )
    save(fig, layout.stem)


def main() -> None:
    """Draw both map variants."""
    apply_style()
    for layout in LAYOUTS:
        draw(layout)


if __name__ == "__main__":
    main()
