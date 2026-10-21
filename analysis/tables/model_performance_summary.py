"""Render model performance to ``data/derived/tables/model_performance_summary.tex``.

``build_model_performance`` keeps values numeric; LaTeX formatting is added
only by :func:`render_model_performance_latex`."""

from __future__ import annotations

import pandas as pd

from analysis.model_names import model_parameter_metadata, reasoning_category
from analysis.model_results.human_eval_summary import (
    COMMUNITY_BUCKET_COLUMNS,
    build_summary,
)
from analysis.tables.common import latex_escape, write_table

ACCESS_ABBREVIATIONS = {
    "proprietary": "P",
    "open-weight": "OW",
    "open-source": "OS",
}


REASONING_LABELS = {
    "N/A": "N/A",
    "default": "Default",
    "none": "None",
    "medium": "Medium",
    "enabled": "Enabled",
    "adaptive": "Adapt.",
    "high": "High",
}


MODEL_PERFORMANCE_CAPTION = (
    r"Performance of \nummodels model configurations on "
    r"\numhumanscoredcommunities cultures with human evaluation, sorted by "
    r"human-correct accuracy in descending order. \emph{Access} abbreviations "
    r"are P (proprietary), OW (open-weight: weights, code, and a technical "
    r"report released), and OS (open-source: additionally the training data "
    r"and full training recipes). "
    r"\emph{Parameters} are in billions: active/total for "
    r"mixture-of-experts models, total for dense models, and ``--'' when "
    r"undisclosed. \emph{Total} denotes \% of score 3 responses, \emph{exact} "
    r"accuracy contributed by exact match, and \emph{review} by annotators and "
    r"LLM judges. \emph{Reasoning} is the effort set in the request. "
    r"\emph{Default} means that no effort was set, so the provider default applied, "
    r"and N/A means that the model has no reasoning mode. "
    r"Extrema across these accuracy columns are \textbf{bolded}. "
    r"The final five columns show the number of cultures in fixed intervals of "
    r"accuracy, with the largest count in each row \textbf{bolded}."
)


def _format_parameter_count(value: float | None) -> str:
    """Format a parameter count without unnecessary trailing zeros."""
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):g}"


def _format_parameters(active: float | None, total: float | None) -> str:
    """Format dense and mixture-of-experts parameter metadata."""
    if pd.isna(total):
        return "--"
    if pd.isna(active):
        return _format_parameter_count(total)
    return f"{_format_parameter_count(active)}/{_format_parameter_count(total)}"


def build_model_performance(all_communities: bool = False) -> pd.DataFrame:
    """Build the model-performance summary table.

    Args:
        all_communities: Include the benchmarked communities without human
            labels (see :func:`build_summary`).

    Returns:
        One row per model with matched score-3/exact percentages, metadata, and
        counts of communities in five fixed score-3-accuracy intervals.

    Raises:
        ValueError: If a model lacks metadata or the community buckets do not
            sum to the evaluated total.
    """
    summary = build_summary(all_communities=all_communities)
    model_metadata = model_parameter_metadata().set_index("model_display")
    missing_models = set(summary["model"]) - set(model_metadata.index)
    if missing_models:
        raise ValueError(
            f"Model summary contains models without metadata: {sorted(missing_models)}"
        )
    models = summary["model"]
    table = pd.DataFrame(
        {
            "rank": range(1, len(summary) + 1),
            "model": models,
            "access_abbreviation": models.map(model_metadata["access"].map(ACCESS_ABBREVIATIONS)),
            "parameters_b": [
                _format_parameters(active, total)
                for active, total in zip(
                    models.map(model_metadata["active_parameter_count_b"]),
                    models.map(model_metadata["model_size_b"]),
                    strict=True,
                )
            ],
            "reasoning": [
                reasoning_category(default, actual)
                for default, actual in zip(
                    models.map(model_metadata["default_reasoning"]),
                    models.map(model_metadata["actual_reasoning_effort"]),
                    strict=True,
                )
            ],
            "exact_match": summary["exact_match"],
            "human_accuracy": summary["human_correct"],
            "difference_pp": summary["difference_pp"],
            **{column: summary[column] for column in COMMUNITY_BUCKET_COLUMNS},
            "n_evaluated_communities": summary["n_evaluated_communities"],
        }
    )
    if table[["access_abbreviation", "reasoning"]].isna().any().any():
        raise ValueError("Model performance table contains incomplete model metadata")
    bucket_total = table[list(COMMUNITY_BUCKET_COLUMNS)].sum(axis=1)
    if not bucket_total.eq(table["n_evaluated_communities"]).all():
        raise ValueError("Community accuracy buckets do not sum to the evaluated total")
    return table


def _accuracy_cell(value: float, column: pd.Series, macro: str, bold_macro: str) -> str:
    """Render one aggregate accuracy cell, bolding column extrema."""
    command = bold_macro if value in {column.min(), column.max()} else macro
    return rf"\{command}{{{value:.1f}}}"


def _shade_macros(name: str, low: float, high: float) -> list[str]:
    """Return the plain and bold blue-shaded cell macros for one column."""
    shade = (
        rf"\pgfmathtruncatemacro{{\shade}}{{8+24*(#1-{low:.1f})/({high:.1f}-{low:.1f})}}\shadeblue"
    )
    return [
        rf"  \newcommand{{\{name}cell}}[1]{{{shade}#1}}",
        rf"  \newcommand{{\bold{name}cell}}[1]{{{shade}\textbf{{#1}}}}",
    ]


def render_model_performance_latex(frame: pd.DataFrame) -> str:
    """Render the model-performance DataFrame as a LaTeX table.

    Args:
        frame: Output from :func:`build_model_performance`.

    Returns:
        The complete ``table*`` environment.
    """
    accuracy_columns = (
        ("human", "human_accuracy"),
        ("exact", "exact_match"),
        ("delta", "difference_pp"),
    )
    count_max = int(frame[list(COMMUNITY_BUCKET_COLUMNS)].to_numpy().max())
    rows: list[str] = []
    for row in frame.to_dict(orient="records"):
        counts = [int(row[column]) for column in COMMUNITY_BUCKET_COLUMNS]
        fields = [
            str(row["rank"]),
            latex_escape(row["model"]),
            str(row["access_abbreviation"]),
            str(row["parameters_b"]),
            REASONING_LABELS[str(row["reasoning"])],
            *(
                _accuracy_cell(float(row[column]), frame[column], name + "cell", f"bold{name}cell")
                for name, column in accuracy_columns
            ),
            *(
                rf"\{'boldcountcell' if value == max(counts) else 'countcell'}{{{value}}}"
                for value in counts
            ),
        ]
        rows.append("    " + " & ".join(fields) + r" \\")

    return "\n".join(
        [
            r"\begin{table*}[ht]",
            r"  \centering",
            r"  \footnotesize",
            r"  \setlength{\tabcolsep}{2.6pt}",
            r"  \renewcommand{\arraystretch}{1.12}",
            r"  \definecolor{tableblue}{HTML}{0072B2}",
            r"  \definecolor{tableamber}{HTML}{E69F00}",
            r"  \newcommand{\shadeblue}{\edef\cellshade{\noexpand\cellcolor{tableblue!\shade}}\cellshade}",
            r"  \newcommand{\shadeamber}{\edef\cellshade{\noexpand\cellcolor{tableamber!\shade}}\cellshade}",
            *(
                line
                for name, column in accuracy_columns
                for line in _shade_macros(name, *frame[column].agg(["min", "max"]))
            ),
            rf"  \newcommand{{\countcell}}[1]{{\pgfmathtruncatemacro{{\shade}}{{8+42*(#1)/({count_max})}}\shadeamber#1}}",
            rf"  \newcommand{{\boldcountcell}}[1]{{\pgfmathtruncatemacro{{\shade}}{{8+42*(#1)/({count_max})}}\shadeamber\textbf{{#1}}}}",
            "",
            r"  \begin{tabular}{@{}r l c c c r r r r r r r r@{}}",
            r"    \toprule",
            r"    & & & & & \multicolumn{3}{c}{Accuracy (\%)} &",
            r"    \multicolumn{5}{c}{Communities by accuracy (\%)} \\",
            r"    \cmidrule(lr){6-8} \cmidrule(l){9-13}",
            r"    \# & Model & Access & Params (B) & Reason. & Total & Exact & Review & $0$ & $(0,25)$ & $[25,50)$ & $[50,75)$ & $[75,100]$ \\",
            r"    \midrule",
            *rows,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \moveup",
            rf"  \caption{{{MODEL_PERFORMANCE_CAPTION}}}",
            r"  \vspace{-1em}",
            r"  \label{tab:model-summary}",
            r"\end{table*}",
            "",
        ]
    )


def main() -> None:
    """Write the model-performance table."""
    write_table(
        "model_performance_summary",
        render_model_performance_latex(build_model_performance()),
    )


if __name__ == "__main__":
    main()
