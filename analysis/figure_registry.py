"""Map every figure and generated table to the one script that writes it.

Figures go to ``figures/paper/`` and ``figures/preprint/``. Tables go to
``data/derived/tables/``; macros and machine-readable claims go to
``data/derived/claims/``. ``analysis.figures.main`` draws every figure and
``analysis.tables.build`` writes every table and the claims.

Run a generator with ``uv run python -m <module>`` from the repository root.
"""

from __future__ import annotations

from analysis.constants import PAPER_FIGURES_DIR, PREPRINT_FIGURES_DIR


def _figures(*stems: str) -> dict[str, str]:
    """Map figure stems to their ``analysis.figures`` module."""
    return {stem: f"analysis.figures.{stem}" for stem in stems}


# Figures rendered at column or text width in both output sets.
FIGURES: dict[str, str] = _figures(
    "fig_map",
    "fig_pipeline",
    "fig_model_summary",
    "fig_cultures",
    "fig_resourcedness",
    "fig_variance",
    "fig_errors",
    "fig_riddle_features",
    "fig_surface_forms",
)

# Additional figures and wide variants.
PREPRINT_FIGURES: dict[str, str] = {
    "fig_map_wide": "analysis.figures.fig_map",
    **_figures(
        "fig_pipeline_wide",
        "fig_rater_agreement",
        "fig_english",
        "fig_judge_by_model",
        "fig_judge_by_community",
        "fig_native_counterfactual",
    ),
}

TABLES: dict[str, str] = {
    name: f"analysis.tables.{name}"
    for name in (
        "coverage",
        "english",
        "human_error_types",
        "model_performance_summary",
        "judge_agreement",
        "judge_prompt",
        "judge_by_community",
        "judge_by_model",
        "riddle_features",
    )
}

CLAIMS: dict[str, str] = {name: "analysis.tables.macros" for name in ("macros", "claims", "counts")}


def figure_modules() -> list[str]:
    """Return every figure module once, shared figures first."""
    return list(dict.fromkeys([*FIGURES.values(), *PREPRINT_FIGURES.values()]))


def unregistered_figures() -> set[str]:
    """Return figure stems on disk that have no registered generator.

    Returns:
        Stems in ``figures/paper/`` that are not paper figures, and stems in
        ``figures/preprint/`` that are neither paper nor preprint figures.
    """
    paper = {path.stem for path in PAPER_FIGURES_DIR.glob("*.pdf")}
    preprint = {path.stem for path in PREPRINT_FIGURES_DIR.glob("*.pdf")}
    return (paper - FIGURES.keys()) | (preprint - FIGURES.keys() - PREPRINT_FIGURES.keys())
