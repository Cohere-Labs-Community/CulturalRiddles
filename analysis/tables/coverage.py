"""Render community coverage to ``data/derived/tables/coverage.tex``."""

from __future__ import annotations

import pandas as pd

from analysis.constants import HUMAN_ANNOTATIONS_PATH
from analysis.corpus import load_corpus_metadata
from analysis.tables.common import latex_escape, write_table

COVERAGE_CAPTION = (
    r"\textbf{Coverage inventory for cultures in \dataset}, characterized by language, "
    r"country, ISO 639-3 language code, ISO alpha-3 country code, and UN M49 "
    r"subregion. \# writers is the number of riddle writers, and \# evaluators "
    r"is the number of annotators for manually scored responses. \cmark{} "
    r"indicates that a subset of model responses (typically 100 responses, or "
    r"10\%) for that cultural community were annotated by a second, different "
    r"annotator (H2)."
)


def load_iaa_communities(metadata: pd.DataFrame) -> set[str]:
    """Return communities with second annotations, checked against metadata."""
    labels = pd.read_parquet(HUMAN_ANNOTATIONS_PATH, columns=["community", "annotation"])
    iaa_communities = set(labels.loc[labels["annotation"].eq("second"), "community"])
    unknown_iaa_communities = iaa_communities.difference(metadata["community"])
    if unknown_iaa_communities:
        raise ValueError(
            "Second annotations name communities absent from metadata: "
            f"{sorted(unknown_iaa_communities)}"
        )
    return iaa_communities


def build_coverage_metrics(metadata: pd.DataFrame) -> pd.DataFrame:
    """Attach evaluator and IAA coverage to each community's writer count."""
    labels = pd.read_parquet(
        HUMAN_ANNOTATIONS_PATH,
        columns=["community", "annotation", "annotator_id", "benchmark_status"],
    )
    first = labels.loc[labels["annotation"].eq("first") & labels["benchmark_status"].eq("valid")]
    evaluators = (
        first.groupby("community")["annotator_id"]
        .agg(lambda ids: len(set(ids.dropna().str.split(";").explode())))
        .rename("evaluators")
    )
    coverage = metadata.set_index("community").join(evaluators).reset_index()
    coverage["evaluators"] = coverage["evaluators"].fillna(0).astype(int)
    coverage["has_iaa"] = coverage["community"].isin(load_iaa_communities(metadata))
    return coverage


def render_coverage_latex(coverage: pd.DataFrame) -> str:
    """Render the benchmark coverage appendix table with source-derived metrics."""
    writer_min, writer_max = coverage["riddle_writers"].agg(["min", "max"])
    evaluator_min, evaluator_max = coverage["evaluators"].agg(["min", "max"])
    writer_cell = (
        r"\newcommand{\coveragewritercell}[1]{\cellcolor{coveragewriter!24}#1}"
        if writer_min == writer_max
        else rf"\newcommand{{\coveragewritercell}}[1]{{\pgfmathtruncatemacro{{\coveragewritershade}}{{8+32*(#1-{writer_min})/({writer_max}-{writer_min})}}\edef\coveragecellshade{{\noexpand\cellcolor{{coveragewriter!\coveragewritershade}}}}\coveragecellshade#1}}"
    )
    evaluator_cell = (
        r"\newcommand{\coverageevaluatorcell}[1]{\cellcolor{coverageevaluator!24}#1}"
        if evaluator_min == evaluator_max
        else rf"\newcommand{{\coverageevaluatorcell}}[1]{{\pgfmathtruncatemacro{{\coverageevaluatorshade}}{{8+32*(#1-{evaluator_min})/({evaluator_max}-{evaluator_min})}}\edef\coveragecellshade{{\noexpand\cellcolor{{coverageevaluator!\coverageevaluatorshade}}}}\coveragecellshade#1}}"
    )
    total_regions = int(coverage["un_m49_subregion"].nunique())
    rows = []
    for index, row in enumerate(coverage.to_dict("records"), start=1):
        values = [
            str(index),
            *(
                latex_escape(row[column])
                for column in (
                    "language",
                    "country",
                    "language_code",
                    "country_code",
                    "un_m49_subregion",
                )
            ),
            rf"\coveragewritercell{{{row['riddle_writers']}}}",
            rf"\coverageevaluatorcell{{{row['evaluators']}}}" if row["evaluators"] else "",
            r"\cmark" if row["has_iaa"] else "",
        ]
        rows.append("        " + " & ".join(values) + r" \\")
    return "\n".join(
        [
            r"\begin{table*}[pt]",
            r"    \centering",
            r"    \footnotesize",
            r"    \renewcommand{\arraystretch}{0.87}",
            r"    \setlength{\tabcolsep}{2.2pt}",
            r"    \definecolor{coveragewriter}{HTML}{0072B2}",
            r"    \definecolor{coverageevaluator}{HTML}{E69F00}",
            "    " + writer_cell,
            "    " + evaluator_cell,
            r"    \begin{tabular}{@{}rllcclrrc@{}}",
            r"        \toprule",
            r"        No. & Language & Country & Lang. code & Country code & Region & \# writers & \# evaluators & H2 \\",
            r"        \midrule",
            *rows,
            r"        \midrule",
            rf"        \multicolumn{{3}}{{r}}{{\textbf{{Total}}}} & \numlang & \numcountries & {total_regions} & \numwriters & \numreviewers & \numiaacommunities \\",
            r"        \bottomrule",
            r"    \end{tabular}",
            "    \\caption{" + COVERAGE_CAPTION + "}",
            r"    \label{tab:coverage}",
            r"\end{table*}",
            "",
        ]
    )


def main() -> None:
    """Write the coverage table."""
    write_table(
        "coverage",
        render_coverage_latex(build_coverage_metrics(load_corpus_metadata())),
    )


if __name__ == "__main__":
    main()
