"""Render setting-B community agreement to ``data/derived/tables/judge_by_community.tex``."""

from __future__ import annotations

import polars as pl

from analysis.corpus import load_corpus_metadata
from analysis.tables.common import (
    JUDGE_STUDY_DIR,
    STUDY_JUDGES,
    float_table,
    format_stat,
    latex_escape,
    write_table,
)

COMMUNITY_BLOCKS = 2


COMMUNITY_HEADER = r"Community & Cor.~(\%) & DS & Gem. & Both"


def render_judge_community_latex() -> str:
    """Render per-community correctness kappa from the stratified study."""
    metadata = pl.from_pandas(load_corpus_metadata()[["community", "language", "country_code"]])
    frame = (
        pl.read_csv(JUDGE_STUDY_DIR / "agreement_by_community.csv")
        .filter(pl.col("condition").is_in(list(STUDY_JUDGES)))
        .with_columns(pl.col("condition").replace_strict(STUDY_JUDGES))
    )
    wide = (
        frame.pivot(
            on="condition",
            index="community",
            values=["kappa", "pairs", "human_acc"],
            aggregate_function="first",
        )
        .join(metadata, on="community", how="left")
        .sort("kappa_Both", nulls_last=True)
    )
    cells = [
        f"{latex_escape(row['language'])} ({row['country_code']}) & "
        f"{format_stat(row['human_acc_Both'], 100, '.0f')} & {format_stat(row['kappa_DeepSeek'])} & "
        f"{format_stat(row['kappa_Gemini'])} & {format_stat(row['kappa_Both'])}"
        for row in wide.iter_rows(named=True)
    ]
    height = -(-len(cells) // COMMUNITY_BLOCKS)
    blank = " & " * (len(COMMUNITY_HEADER.split("&")) - 1)
    rows = [
        " & ".join(
            cells[index] if index < len(cells) else blank
            for index in range(row, row + height * COMMUNITY_BLOCKS, height)
        )
        + r" \\"
        for row in range(height)
    ]
    caption = (
        r"\textbf{Correctness $\kappa$ between the benchmark label and each single "
        r"judge (DS, Gem.) or the LLM jury (Both) per community} in the stratified study "
        rf"(setting~B; {wide['pairs_Both'].min()}--{wide['pairs_Both'].max()} "
        r"pairs each), sorted by Both. Both is the jury's two-judge rule: the "
        r"judges' score when they agree, else the human label, so it is an upper "
        r"bound. Cor.~(\%): share of answers that humans scored correct; few "
        r"correct answers mean wide "
        r"intervals. Countries are given as ISO 3166-1 alpha-3 codes."
    )
    columns = r" @{\hspace{2em}} ".join(["l r r r r"] * COMMUNITY_BLOCKS)
    return float_table(
        f"@{{}}{columns}@{{}}",
        " & ".join([COMMUNITY_HEADER] * COMMUNITY_BLOCKS),
        rows,
        caption,
        "tab:judge_by_community",
        "3.5pt",
    )


def main() -> None:
    """Write the per-community judge table."""
    write_table("judge_by_community", render_judge_community_latex())


if __name__ == "__main__":
    main()
