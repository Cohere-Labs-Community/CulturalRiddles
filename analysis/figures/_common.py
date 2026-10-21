"""Shared style and fixed-size output helpers for the analysis figures.

Hard defaults enforced here:
* canvas sized to column width (7.7 cm) or text width (16.0 cm);
* shared green/blue/coral colour with redundant marker, hatch, and line encodings;
* every condition carries colour + a second channel (marker, hatch, line style);
* no titles, captions, grids, boxed legends, or top/right spines;
* vector PDF with live text (fonttype 42) plus a review PNG.

Usage from the repo root::

    uv run python -m analysis.figures.<script>

Each script writes only the selected target (default ``preprint``). Select
the paper output folder with ``CULTURALRIDDLES_FIGURE_TARGET=paper``.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import Text

from analysis.constants import PAPER_FIGURES_DIR, PREPRINT_FIGURES_DIR
from analysis.figure_registry import FIGURES, PREPRINT_FIGURES
from analysis.style import (
    BOTTOM as BOTTOM,
)
from analysis.style import (
    OUTCOME as OUTCOME,
)
from analysis.style import (
    RESOURCE as RESOURCE,
)
from analysis.style import (
    STYLE,
    TARGET,
    C,
)
from analysis.style import (
    TOP as TOP,
)

# Two-column layout: textwidth 16.0 cm, columnsep 0.6 cm -> column 7.7 cm.
COL_W = 7.7 / 2.54  # 3.031 in
FULL_W = 16.0 / 2.54  # 6.299 in
# Wide layout: textwidth 469.76 pt.
PREPRINT_W = 469.76 / 72.27  # 6.5 in


def apply_style() -> None:
    """Apply rcParams: sans text, thin neutral axes, no grid, vector export."""
    mpl.rcParams.update(
        {
            **STYLE["typography"],
            "axes.unicode_minus": False,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "axes.labelcolor": C["ink"],
            "text.color": C["ink"],
            "lines.linewidth": 1.4,
            "lines.markersize": 4.5,
            "hatch.linewidth": 0.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def new_fig(width: float, height: float, **kw) -> plt.Figure:
    """Create a figure at exact final size (no later scaling, no tight crop)."""
    return plt.figure(figsize=(width, height), layout=kw.pop("layout", "constrained"), **kw)


def min_font_pt(fig: plt.Figure) -> float:
    """Return the smallest rendered text size (pt) in ``fig`` for a legibility audit."""
    fig.canvas.draw()
    sizes = [
        t.get_fontsize() for t in fig.findobj(Text) if t.get_text().strip() and t.get_visible()
    ]
    return min(sizes) if sizes else float("nan")


def save(fig: plt.Figure, stem: str) -> None:
    """Write ``<stem>.pdf`` and a natively rendered ``<stem>.png``.

    Only the selected target is written. Stems outside that target are closed
    without rendering (the map generator contains both width variants).
    Column figures get 675 dpi (about 2040 px wide); full-width figures get 450 dpi.
    """
    if stem not in FIGURES and stem not in PREPRINT_FIGURES:
        raise ValueError(f"Unregistered figure: {stem}")
    if TARGET == "paper" and stem not in FIGURES:
        plt.close(fig)
        return
    folder = PAPER_FIGURES_DIR if TARGET == "paper" else PREPRINT_FIGURES_DIR
    w, h = fig.get_size_inches()
    smallest = min_font_pt(fig)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.pdf", metadata={"CreationDate": None})
    fig.savefig(folder / f"{stem}.png", dpi=675 if w < 4 else 450)
    print(f"{TARGET}/{stem}: {w:.3f} x {h:.3f} in, smallest text {smallest:.1f} pt")
    plt.close(fig)


def panel_letter(ax: plt.Axes, letter: str, x: float = -0.02, y: float = 1.02) -> None:
    """Add a bold panel letter at the axes' top-left in axes coordinates."""
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="right",
        va="bottom",
    )
