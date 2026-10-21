"""Render setting-B judge leniency to ``data/derived/tables/judge_by_model.tex``."""

from __future__ import annotations

import polars as pl

from analysis.model_names import model_display_map
from analysis.tables.common import (
    JUDGE_STUDY_DIR,
    STUDY_JUDGES,
    float_table,
    format_stat,
    latex_escape,
    write_table,
)

MODEL_HEADER = r"Model & $n_{\text{rej}}$ & DS & Gem. & Both & $\Delta$ (pp)"


def judge_model_frame() -> pl.DataFrame:
    """Return each answering model's judge false-accept rates and accuracy shift.

    Returns:
        One row per model with ``model`` (display name), ``rejected_rows``,
        the ``DeepSeek``, ``Gemini``, and ``Both`` false-accept rates, and the
        agreement-with-human-fallback rule's accuracy ``shift`` and interval.
    """
    names = model_display_map()
    rates = (
        pl.read_csv(JUDGE_STUDY_DIR / "false_accept_by_model.csv")
        .filter(pl.col("condition") != "ensemble-min (zero)")
        .with_columns(pl.col("condition").replace_strict(STUDY_JUDGES))
    )
    # A judge without a verdict on a pair sees one fewer rejected row.
    accept = rates.pivot(on="condition", index="model_name", values="false_accept_rate").join(
        rates.group_by("model_name").agg(pl.col("rejected_rows").max()),
        on="model_name",
    )
    shift = (
        pl.read_csv(JUDGE_STUDY_DIR / "leaderboard_intervals.csv")
        .filter(pl.col("condition") == "agree-or-human (zero)")
        .select("model_name", "shift", "low", "high")
    )
    return accept.join(shift, on="model_name", how="left").with_columns(
        model=pl.col("model_name").replace_strict(names)
    )


def render_judge_model_latex() -> str:
    """Render judge leniency and accuracy shift for each answering model."""
    table = judge_model_frame().sort("Gemini", descending=True)
    rows = [
        f"{latex_escape(row['model'])} & {row['rejected_rows']:,} & "
        f"{format_stat(row['DeepSeek'], 100, '.1f')} & {format_stat(row['Gemini'], 100, '.1f')} & "
        f"{format_stat(row['Both'], 100, '.1f')} & "
        + (
            rf"\textbf{{{format_stat(row['shift'], 100, '.1f')}}}"
            if row["low"] > 0 or row["high"] < 0
            else format_stat(row["shift"], 100, ".1f")
        )
        + rf" {{\scriptsize[{format_stat(row['low'], 100, '.1f')}, "
        + rf"{format_stat(row['high'], 100, '.1f')}]}} \\"
        for row in table.iter_rows(named=True)
    ]
    max_shift = max(map(abs, table["shift"].to_list())) * 100
    caption = (
        r"\textbf{Judge leniency by answering model} (setting~B): $n_{\text{rej}}$: number "
        r"of human-rejected responses (scores 0--2) of each answering model; DS, "
        r"Gem., Both: share (\%) of them that DeepSeek, Gemini, or both judges "
        r"score correct; $\Delta$: accuracy shift in pp when jury consensus "
        r"replaces a human score and disagreements retain the human score "
        r"(95\% paired bootstrap interval; bold intervals exclude zero). "
        r"Under this human-fallback rule, no model's accuracy "
        rf"moves by more than {max_shift:.1f}\,pp."
    )
    half = -(-len(rows) // 2)
    cells = [row.removesuffix(" \\\\") for row in rows]
    right = cells[half:] + ["& & & & &"] * (2 * half - len(cells))
    return float_table(
        r"@{}l r r r r r@{\hspace{0.6em}}l r r r r r@{}",
        f"{MODEL_HEADER} & {MODEL_HEADER}",
        [f"{left} & {other} \\\\" for left, other in zip(cells[:half], right)],
        caption,
        "tab:judge_by_model",
        "1.6pt",
    )


def main() -> None:
    """Write the per-model judge table."""
    write_table("judge_by_model", render_judge_model_latex())


if __name__ == "__main__":
    main()
