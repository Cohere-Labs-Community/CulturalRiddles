"""Render English-translation results to ``data/derived/tables/english.tex``."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from analysis.corpus import load_corpus_metadata
from analysis.llm_judge.english_condition import (
    ENGLISH_MODELS,
    OUTPUT_DIR,
    answer_in_riddle,
    english_rows,
    high_resource_accuracy,
    load_export,
)
from analysis.tables.common import format_count, write_table

ALL_RIDDLES_CSV = OUTPUT_DIR / "accuracy_by_community.csv"
WITHOUT_REVEALING_CSV = OUTPUT_DIR / "accuracy_by_community_without_answer_in_riddle.csv"
NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
ROW_ORDER = (
    "yoruba_nigeria",
    "arabic_sudan",
    "lumasaba_uganda",
    "bugisnese_indonesia",
    "mongondow_indonesia",
    "all",
)


@dataclass(frozen=True)
class EnglishSummary:
    """Every number the English-condition table and text quote."""

    all_riddles: pl.DataFrame
    without_revealing: pl.DataFrame
    riddles: int
    revealing_by_community: dict[str, int]
    revealing_english_accuracy: float
    high_resource_accuracy: float

    @property
    def revealing(self) -> int:
        """Riddles whose English translation contains the answer."""
        return sum(self.revealing_by_community.values())


def english_summary() -> EnglishSummary:
    """Collect the English-condition results written by ``english_condition``."""
    export = load_export()
    revealing = answer_in_riddle(export)
    revealing_ids = revealing["riddle_id"].unique()
    scored = english_rows(export).filter(
        pl.col("riddle_id").is_in(revealing_ids.implode()) & pl.col("is_scored")
    )
    return EnglishSummary(
        all_riddles=pl.read_csv(ALL_RIDDLES_CSV),
        without_revealing=pl.read_csv(WITHOUT_REVEALING_CSV),
        riddles=export["riddle_id"].n_unique(),
        revealing_by_community=dict(
            export.filter(pl.col("riddle_id").is_in(revealing_ids.implode()))
            .unique("riddle_id")
            .group_by("community_id")
            .len()
            .sort("len", descending=True)
            .iter_rows()
        ),
        revealing_english_accuracy=scored.select(
            pl.col("is_correct").cast(pl.Float64).mean() * 100
        ).item(),
        high_resource_accuracy=high_resource_accuracy(),
    )


def signed(value: float) -> str:
    """Format a signed number with a typographic minus."""
    return f"$+${value:.1f}" if value >= 0 else f"$-${abs(value):.1f}"


def rows(frame: pl.DataFrame, communities: tuple[str, ...]) -> list[str]:
    """Render one block of community rows."""
    names = dict(load_corpus_metadata()[["community", "language"]].itertuples(index=False)) | {
        "all": "All"
    }
    lookup = {row["community"]: row for row in frame.iter_rows(named=True)}
    return [
        f"{names[c]} & {lookup[c]['paired_original']:.1f} & "
        f"{lookup[c]['paired_english']:.1f} & {signed(lookup[c]['paired_delta'])} "
        f"({signed(lookup[c]['paired_delta_low']).replace('$+$', '')}--"
        f"{signed(lookup[c]['paired_delta_high']).replace('$+$', '')}) \\\\"
        for c in communities
    ]


def render_english_latex(summary: EnglishSummary) -> str:
    """Render the original-versus-English accuracy table."""
    changed = tuple(c for c in ROW_ORDER if c in summary.revealing_by_community or c == "all")
    models = len(ENGLISH_MODELS)

    def scored(frame: pl.DataFrame) -> str:
        total = frame.filter(pl.col("community") == "all").row(0, named=True)
        return f"{format_count(total['paired_responses'])} of {format_count(total['responses'])}"

    languages = dict(load_corpus_metadata()[["community", "language"]].itertuples(index=False))
    clean = [languages[c] for c in ROW_ORDER[:-1] if c not in summary.revealing_by_community]
    no_revealing = (
        ", ".join(clean[:-1]) + f", and {clean[-1]}" if len(clean) > 2 else " and ".join(clean)
    )
    caption = (
        r"\textbf{Translating into English raises accuracy by about 10\,pp, "
        r"with or without riddles that reveal their answer.} Pooled accuracy (\%) "
        f"of {NUMBER_WORDS.get(models, models)} models on the original and English riddles, over answers "
        f"scored in both conditions ({scored(summary.all_riddles)} with all "
        f"riddles; {scored(summary.without_revealing)} without the revealing "
        r"ones). Gain is the paired difference in pp with a 95\% riddle-bootstrap CI; it "
        r"can differ from English minus original by 0.1\,pp due to rounding. "
        f"{no_revealing} have no revealing riddles. Most revealing riddles are "
        "loose glosses of a local term, and models answer only "
        f"{summary.revealing_english_accuracy:.1f}\\% of them in English."
    )
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l r r l@{}}",
        r"\toprule",
        r"Community & Orig. & Engl. & Gain (95\% CI) \\",
        r"\midrule",
        rf"\multicolumn{{4}}{{@{{}}l}}{{\emph{{All {summary.riddles} riddles}}}} \\",
        *rows(summary.all_riddles, ROW_ORDER),
        r"\midrule",
        (
            rf"\multicolumn{{4}}{{@{{}}l}}{{\emph{{Without the {summary.revealing} "
            r"revealing riddles}} \\"
        ),
        *rows(summary.without_revealing, changed),
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        r"\label{tab:english}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Write ``data/derived/tables/english.tex``."""
    write_table("english", render_english_latex(english_summary()))


if __name__ == "__main__":
    main()
