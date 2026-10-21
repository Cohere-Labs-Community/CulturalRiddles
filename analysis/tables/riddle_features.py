"""Render riddle and answer features and their odds ratios as a table.

Counts come from the answer-level frame that
``difficulty_decomposition.riddle_features`` writes, and odds ratios from that
module's fit (``fit_riddle_features.R``), so both describe the same riddles.
"""

from __future__ import annotations

import polars as pl

from analysis.difficulty_decomposition.prepare import OUTPUT_DIR
from analysis.difficulty_decomposition.riddle_features import (
    COMMON_WORD_MIN_ZIPF,
    FEATURE_FRAME_PATH,
    LOCAL_TERM_MAX_ZIPF,
)
from analysis.tables.common import write_table

EFFECTS_PATH = OUTPUT_DIR / "riddle_feature_effects.csv"
COLUMNS = (
    "riddle_id",
    "answer_kind",
    "riddle_form",
    "multi_clue",
    "multiline",
    "log2_words",
)
CAPTION = (
    r"\caption{{\textbf{{Answers without a common English name halve the odds of a "
    r"correct answer.}} Features are computed automatically from the English "
    r"translation of the {riddles:,} riddles in the model. Answer groups use the "
    r"\texttt{{wordfreq}} Zipf frequency~\citep{{speer2022wordfreq}} of the rarest "
    r"word in the English reference answer. Odds ratios (OR) with 95\% CIs come "
    r"from the random-effects logistic model of a correct answer, which also "
    r"controls for model and resource level; ref.\ marks reference levels.}}"
)
FEATURE_ROWS: tuple[tuple[str, str, str, pl.Expr, str | None], ...] = (
    (
        "Answer",
        "No common English name",
        (
            rf"rarest word below Zipf {LOCAL_TERM_MAX_ZIPF:g} "
            r"(\emph{bebinca})"
        ),
        pl.col("answer_kind") == "local_term",
        "answer_kindlocal_term",
    ),
    (
        "",
        "In between",
        (
            rf"Zipf {LOCAL_TERM_MAX_ZIPF:g} to {COMMON_WORD_MIN_ZIPF:g} "
            r"(\emph{snowmobile})"
        ),
        pl.col("answer_kind") == "mid_frequency",
        "answer_kindmid_frequency",
    ),
    (
        "",
        "Common English word",
        rf"Zipf {COMMON_WORD_MIN_ZIPF:g} or above (\emph{{drum}}, \emph{{cooking}})",
        pl.col("answer_kind") == "common_word",
        None,
    ),
    (
        "",
        "Number",
        "no letters (years, numbers)",
        pl.col("answer_kind") == "numeric",
        "answer_kindnumeric",
    ),
    (
        "Form",
        "Question",
        r"contains ``?'' or opens with a question word",
        pl.col("riddle_form") == "question",
        "riddle_formquestion",
    ),
    (
        "",
        "First-person statement",
        "first-person pronoun, not a question",
        pl.col("riddle_form") == "first_person_statement",
        "riddle_formfirst_person_statement",
    ),
    (
        "",
        "Third-person description",
        "all other riddles",
        pl.col("riddle_form") == "third_person_description",
        None,
    ),
    (
        "Clues",
        "Several clues",
        r"$\geq$2 sentences or a contrast word (\emph{but}, \emph{never})",
        pl.col("multi_clue"),
        "multi_clueTRUE",
    ),
    (
        "",
        "Multiline",
        "original text has a line break (verse)",
        pl.col("multiline"),
        "multilineTRUE",
    ),
)


def load_answers() -> pl.DataFrame:
    """Read the answers and riddle features used by the riddle-feature model.

    Returns:
        One row per scored answer with its riddle's features.
    """
    return pl.read_csv(FEATURE_FRAME_PATH, columns=list(COLUMNS))


def load_odds_ratios() -> dict[str, str]:
    """Read the feature model's odds ratios, formatted for the table.

    Returns:
        ``OR (low--high)`` strings keyed by model term.
    """
    effects = pl.read_csv(EFFECTS_PATH)
    return {
        row["term"]: (f"{row['odds_ratio']:.2f} ({row['ci_low']:.2f}--{row['ci_high']:.2f})")
        for row in effects.iter_rows(named=True)
    }


def render_riddle_features_latex(answers: pl.DataFrame) -> str:
    """Render each feature's rule, riddle count, and odds ratio.

    Args:
        answers: Output of :func:`load_answers`.

    Returns:
        A complete LaTeX ``table*`` environment.
    """
    riddles = answers.unique("riddle_id")
    counts = riddles.select(
        *(row[3].sum().alias(str(index)) for index, row in enumerate(FEATURE_ROWS))
    ).row(0)
    odds = load_odds_ratios()
    body = []
    for (group, level, rule, _, term), count in zip(FEATURE_ROWS, counts, strict=True):
        if group and body:
            body.append(r"\midrule")
        effect = odds[term] if term else "ref."
        body.append(f"{group} & {level} & {rule} & {count:,} & {effect} \\\\")
    median_words = riddles.select(2 ** pl.col("log2_words").median()).item()
    body += [
        r"\midrule",
        (
            f"Length & Words & per doubling (median {median_words:.0f} words) & -- & "
            f"{odds['log2_words_c']} \\\\"
        ),
    ]
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            CAPTION.format(riddles=riddles.height),
            r"\label{tab:riddle-features}",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{@{}l l l r l@{}}",
            r"\toprule",
            r"Feature & Level & Rule (examples) & Riddles & OR (95\% CI) \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )


def main() -> None:
    """Write the riddle-feature table from the current fit."""
    write_table("riddle_features", render_riddle_features_latex(load_answers()))


if __name__ == "__main__":
    main()
