"""Render the error-type table to ``data/derived/tables/human_error_types.tex``.

Fine-grained error types are counted over rule-scored and human-scored
responses only; LLM-jury and unscored (pending) responses never enter.
"""

from __future__ import annotations

import pandas as pd

from analysis.model_results.evaluation_data import (
    human_evaluated_communities,
    load_flat_frame,
)
from analysis.tables.common import (
    format_count,
    format_percent,
    latex_escape,
    write_table,
)

ERROR_TYPE_DETAILS: dict[str, tuple[str, str]] = {
    "0.1": ("Too broad", "The answer is too generic or a blanket answer."),
    "0.2": (
        "Right category, wrong entity",
        "The category is correct, but the specific entity is incorrect (e.g., model answer is a different volleyball player)",
    ),
    "0.3": ("Hallucinated", "The answer is made up, or completely irrelevant."),
    "0.4": ("Other (add note)", "Other incorrect answer; requires annotator's note."),
    "2.1": (
        "Closely related culture or region",
        "The answer confuses related cultures or regions (e.g., national vs. regional).",
    ),
    "2.2": (
        "Wrong level of detail",
        "The answer is close to the reference answer but misses some details (e.g., book title instead of author).",
    ),
    "2.3": (
        "Surface pattern match",
        "The answer reuses a clue keyword without solving the riddle (e.g., ``cycle'' for an ``hour cycle'' riddle).",
    ),
    "2.4": (
        "Other (add note)",
        "Other partially correct answer; needs annotator's note.",
    ),
    "3.1": (
        "Different valid answer",
        "The answer is valid but differs from the reference answer (e.g., another accepted name for the same game).",
    ),
    "3.2": (
        "Correct (in English)",
        "The valid answer is in English, not riddle's language.",
    ),
    "3.3": ("Other correct", "Correct answer, requires a note."),
}


def human_evaluation_frame() -> pd.DataFrame:
    """Load the human-scored responses of every human-evaluated community.

    Pending answers with a valid first label in the doubly annotated sample
    count as human-scored. Labels are flat: one row per model response.
    """
    frame = load_flat_frame(with_judges=False)
    included = human_evaluated_communities(frame)
    return frame.loc[frame["is_human"] & frame["community"].isin(included)].copy()


def error_type_frame() -> pd.DataFrame:
    """Return the population behind every fine-grained error-type share.

    Rule-scored rows (exact match, empty output, answer placeholder, explicit
    "I don't know") plus human-scored rows of the 28 reported models. LLM-jury
    and unscored (pending) rows are excluded: the jury is not reliable enough
    for error types, so it never contributes to them.
    """
    frame = load_flat_frame(with_judges=False)
    return frame.loc[(frame["is_human"] | frame["is_auto"]) & frame["score"].notna()].copy()


def _error_type_code(value: object) -> str:
    """Normalize a numeric error-type value to its published code."""
    return f"{float(str(value)):.1f}"


def build_human_error_types() -> pd.DataFrame:
    """Summarize the error-type taxonomy for the generated table.

    Rows cover every rule- or human-scored response, so the counts sum to the
    table total: score-0 and score-2 subtypes, score-1 abstentions, and score-3
    responses (rule-based exact matches plus human-labelled correct variants).
    """
    scored = error_type_frame()
    human = scored.loc[scored["is_human"]]
    auto = scored.loc[scored["is_auto"]]
    method = auto["scoring_method"]
    typed = human.loc[human["error_type"].notna()].copy()
    typed["code"] = typed["error_type"].map(_error_type_code)
    counts = typed.groupby("code").size().to_dict()
    unknown_codes = sorted(set(counts).difference(ERROR_TYPE_DETAILS), key=float)
    if unknown_codes:
        raise ValueError(f"Human evaluation contains unrecognized error types: {unknown_codes}")

    def row(code: str, label: str, definition: str, responses: int, group: str) -> dict:
        return {
            "code": code,
            "error_type": label,
            "definition": definition,
            "responses": int(responses),
            "score_group": group,
        }

    rows: list[dict[str, object]] = []
    for code, (label, definition) in ERROR_TYPE_DETAILS.items():
        if code.startswith("3.") or code not in counts:
            continue
        rows.append(row(code, label, definition, counts[code], code.split(".")[0]))

    untyped_scores = human.loc[human["error_type"].isna(), "score"]
    abstentions = int(untyped_scores.eq(1).sum()) + int(method.eq("auto_idk").sum())
    for score, label in ((0, "Incorrect, no type"), (2, "Partial, no type")):
        untyped = int(untyped_scores.eq(score).sum())
        if untyped:
            rows.append(
                row(
                    f"{score}.x",
                    label,
                    "The annotator gave the score without a type.",
                    untyped,
                    str(score),
                )
            )
    rows.append(
        row(
            "r0e",
            "Empty output",
            "The model returned no text (rule-based).",
            method.eq("auto_empty").sum(),
            "0",
        )
    )
    rows.append(
        row(
            "r0p",
            "Placeholder",
            "The model returned the prompt's answer template token (rule-based).",
            method.eq("auto_placeholder").sum(),
            "0",
        )
    )
    if abstentions:
        rows.append(
            row(
                "1",
                "Admits not knowing",
                "The model says it does not know the answer (rule-based or human).",
                abstentions,
                "1",
            )
        )
    rows.append(
        row(
            "r3",
            "Exact match",
            "Case-insensitive exact match with the reference (rule-based).",
            method.eq("auto_exact_match").sum(),
            "3",
        )
    )
    correct_variants = int(untyped_scores.eq(3).sum())
    if correct_variants:
        rows.append(
            row(
                "3",
                "Correct, no type",
                "Human judged the answer to be correct without any sublabel being recorded",
                correct_variants,
                "3",
            )
        )
    for code, (label, definition) in ERROR_TYPE_DETAILS.items():
        if code.startswith("3.") and code in counts:
            rows.append(row(code, label, definition, counts[code], "3"))

    table = pd.DataFrame(rows).sort_values("score_group", kind="stable")
    unaccounted = len(scored) - int(table["responses"].sum())
    if unaccounted:
        raise ValueError(f"{unaccounted} scored responses fall outside the taxonomy")
    table = table.reset_index(drop=True)
    table["percent"] = table["responses"] / len(scored) * 100
    table["is_dominant"] = False
    for score_group in ("0", "2", "3"):
        candidates = table.index[table["score_group"].eq(score_group)]
        if len(candidates):
            table.loc[table.loc[candidates, "responses"].idxmax(), "is_dominant"] = True
    return table


def render_human_error_types_latex(frame: pd.DataFrame) -> str:
    """Render the human error-type summary as LaTeX."""
    rows: list[str] = []
    previous_score_group: str | None = None
    for row in frame.to_dict(orient="records"):
        score_group = str(row["score_group"])
        if previous_score_group is not None:
            separator = (
                r"        \midrule"
                if score_group != previous_score_group
                else r"        \addlinespace[2pt]"
            )
            rows.append(separator)
        if row["is_dominant"]:
            rows.append(r"        \dominantrow")
        display_code = (
            ""
            if row["code"] in {"0.x", "1", "2.x", "3"} or row["code"].startswith("r")
            else latex_escape(row["code"])
        )
        rows.append(
            "        "
            + " & ".join(
                [
                    display_code,
                    latex_escape(row["error_type"]),
                    latex_escape(row["definition"]),
                    format_count(int(row["responses"])),
                    format_percent(float(row["percent"])) + r"\%",
                ]
            )
            + r" \\"
        )
        previous_score_group = score_group
    return "\n".join(
        [
            r"\begin{table*}[!htbp]",
            r"    \centering",
            r"    \small",
            r"    \begin{tabularx}{\textwidth}{@{}r p{0.205\textwidth} X r r@{}}",
            r"        \toprule",
            r"        & \textbf{Error type} & \textbf{Definition and example} & \textbf{Responses} & \textbf{\%} \\",
            r"        \midrule",
            *rows,
            r"        \midrule",
            r"        \multicolumn{3}{r}{Total rule- and human-scored responses} & \numerrortyperesponses & 100.00\% \\",
            r"        \bottomrule",
            r"    \end{tabularx}",
            r"    \caption{\textbf{Error types and their counts} over the \numerrortyperesponses responses scored by rules or by human annotators (\numhumanresponses human-labeled, covering \numuniquehumanresponses unique answers; the rest scored by exact-match, empty-output, placeholder and ``I don't know'' rules). Responses scored only by the LLM jury or not yet scored are excluded, since the jury is not reliable for error types. The dominant type within each score is highlighted in blue.}",
            r"    \label{tab:errortypes}",
            r"\end{table*}",
            "",
        ]
    )


def main() -> None:
    """Write the human error-type table."""
    write_table("human_error_types", render_human_error_types_latex(build_human_error_types()))


if __name__ == "__main__":
    main()
