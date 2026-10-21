"""Shared green, blue, and coral figure tokens for every publication target.

Targets select output locations and registered stems, never visual encodings.
"""

from __future__ import annotations

import os

from matplotlib.colors import LinearSegmentedColormap

TARGET_ENV = "CULTURALRIDDLES_FIGURE_TARGET"


def figure_target() -> str:
    """Return the validated output target; individual generators default to preprint."""
    target = os.environ.get(TARGET_ENV, "preprint")
    if target not in {"paper", "preprint"}:
        raise ValueError(f"{TARGET_ENV} must be 'paper' or 'preprint', got {target!r}")
    return target


# Base palette, taken from the benchmark overview artwork.
OVERVIEW = {
    "blue": "#2D4CB9",
    "green": "#5BBF8A",
    "red": "#E0533A",
    "pale_green": "#CDE8B4",
    "peach": "#F0A08C",
    "blue_tint": "#8DA2F2",
    "blue_pale": "#E6EBFC",
    "ink": "#1A1A1A",
    "text": "#111111",
    "white": "#FFFFFF",
    "context": "#C4C7CC",
    "panel": "#F5F7FA",
    "panel_edge": "#E3E6EB",
    "muted": "#555555",
    "tag_edge": "#999999",
    "rule": "#E5E7EB",
    "subtle": "#666666",
    "pill_edge": "#CFD3D9",
    "pill": "#F8F9FB",
    "axis": "#444444",
    "arrow": "#8A8D91",
    "card_ink": "#222222",
    "baseline": "#CFD2D6",
    "rules": "#ECECEC",
    "declined": "#CFCFCF",
}
C = {
    "black": OVERVIEW["ink"],
    "orange": OVERVIEW["peach"],
    "sky": OVERVIEW["blue_tint"],
    "green": OVERVIEW["green"],
    "yellow": OVERVIEW["pale_green"],
    "blue": OVERVIEW["blue"],
    "vermillion": OVERVIEW["red"],
    "purple": OVERVIEW["blue_tint"],
    "grey": "#8A8A8A",
    "lightgrey": OVERVIEW["context"],
    "ink": OVERVIEW["ink"],
}
# Keep the established final-size typography and canvas layouts.
TYPOGRAPHY = {
    "font.family": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.titlesize": 8,
}
STYLE = {
    "resource": tuple(OVERVIEW[k] for k in ("green", "pale_green", "peach", "red")),
    "map_ramp": tuple(OVERVIEW[k] for k in ("blue_pale", "blue_tint", "blue")),
    "accuracy_ramp": tuple(OVERVIEW[k] for k in ("white", "blue_pale", "blue_tint", "blue")),
    "mean": OVERVIEW["blue"],
    "resource_edge": OVERVIEW["ink"],
    "community_hatch": OVERVIEW["ink"],
    "connector_fill": OVERVIEW["blue_pale"],
    "connector_edge": OVERVIEW["blue_tint"],
    "typography": TYPOGRAPHY,
}
TARGET = figure_target()
OUTCOME = {
    "correct": ("Correct", C["green"], ""),
    "partial": ("Partially correct", C["orange"], "////"),
    "halluc": ("Hallucinated", C["vermillion"], ""),
    "incorrect": ("Other incorrect", C["purple"], "\\\\\\\\"),
    "declined": ("Declined to answer", C["blue"], "xxxx"),
    "noanswer": ("Placeholder / empty", C["grey"], "...."),
}
RESOURCE = {
    key: (label, color, marker)
    for key, label, color, marker in zip(
        ("high", "medium", "low", "extremely-low"),
        ("High", "Medium", "Low", "Extremely low"),
        STYLE["resource"],
        ("o", "s", "^", "D"),
        strict=True,
    )
}
ACCESS = {
    "proprietary": ("Proprietary", C["vermillion"], "s"),
    "open-weight": ("Open-weight", C["green"], "^"),
    "open-source": ("Open-source", C["purple"], "o"),
}
JUDGE = {
    "DeepSeek": ("DeepSeek judge", C["purple"], "^", 3.8),
    "Gemini": ("Gemini judge", C["orange"], "s", 3.4),
    "Both": ("Both judges", C["ink"], "o", 3.8),
}
FORMS = {
    "who_am_i": (C["blue"], "o"),
    "what_is_it": (C["vermillion"], "s"),
    "other_question": (C["orange"], "v"),
    "first_person_statement": (C["sky"], "^"),
    "third_person_description": (C["green"], "P"),
}
VARIANCE = {
    "Riddle": (C["blue"], "", C["blue"]),
    "Language": (C["sky"], "", C["sky"]),
    "Community": (C["orange"], "////", STYLE["community_hatch"]),
    "Topic": (C["green"], "", C["green"]),
    "Model": (C["lightgrey"], "", C["lightgrey"]),
    "Model $\\times$ facet": (C["lightgrey"], "////", "#555555"),
    "Residual": (C["lightgrey"], "....", "#555555"),
}
TOP = (C["blue"], "o")
BOTTOM = (C["vermillion"], "s")
MARKER_EDGE = C["ink"]
OUTLINE_WIDTH = 0.45
MAP_RAMP = LinearSegmentedColormap.from_list("languages", STYLE["map_ramp"])


def accuracy_colors() -> list:
    """Return five ordered light-to-dark blue accuracy-bin fills."""
    ramp = LinearSegmentedColormap.from_list("accuracy", STYLE["accuracy_ramp"])
    return [ramp(v) for v in (0.0, 0.25, 0.50, 0.75, 1.0)]


ROUTE = {
    "rules": (OVERVIEW["rules"], "+++", C["ink"], 0.55),
    "human": (OVERVIEW["blue"], "", C["ink"], 0.50),
    "jury": (OVERVIEW["blue_pale"], "|||", C["ink"], 0.30),
}
