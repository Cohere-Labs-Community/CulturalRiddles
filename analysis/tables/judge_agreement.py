"""Render the table that compares LLM judges with human annotators.

Reports one set of metrics for three settings: the doubly annotated sample (two
annotators, two judges), the 20,000-pair stratified sample (one annotator, two
judges), and the answers without human labels (two judges only). Inputs are the committed
aggregates of ``llm_judge.rater_agreement`` and ``llm_judge.pending_labels``.
Writes ``data/derived/tables/judge_agreement.tex``."""

from __future__ import annotations

import polars as pl

from analysis.constants import DERIVED_DIR
from analysis.tables.common import write_table

RATER_DIR = DERIVED_DIR / "llm_judge" / "rater_agreement"


LEVELS = ("binary", "score", "fine_tree")


RATER_LABELS = {
    "H1": "H1",
    "H2": "H2",
    "DeepSeek": "DeepSeek",
    "Gemini": "Gemini",
    "Ensemble": "Jury",
}


DOUBLY_PAIRS = (
    "H1~H2",
    "H1~DeepSeek",
    "H1~Gemini",
    "H1~Ensemble",
    "H2~DeepSeek",
    "H2~Gemini",
    "H2~Ensemble",
    "DeepSeek~Gemini",
)


SETTING_PAIRS = {
    "stratified": ("H1~DeepSeek", "H1~Gemini", "H1~Ensemble", "DeepSeek~Gemini"),
    "pending": ("DeepSeek~Gemini",),
}


SETTING_TITLES = {
    "stratified": "B. Stratified sample, {units:,} pairs",
    "pending": "C. No human label, {units:,} pairs",
}


HUMANS = ("H1", "H2")


JUDGE_PAIR = "DeepSeek~Gemini"


JUDGE_PAIR_LABEL = "LLM Judges"


GROUP_RULE = r"\arrayrulecolor{black!25}\midrule[0.3pt]\arrayrulecolor{black}"


CAPTION = (
    r"\textbf{Judge agreement depends on setting and label granularity.} "
    r"A--C follow \S~\ref{sec:evaluation-pipeline}. "
    r"\textit{Accuracy}: $\kappa$ for score~3 versus scores~0--2; "
    r"\textit{Score 0--3}: quadratic-weighted $\kappa$; "
    r"\textit{Type}: weighted $\kappa$ over 13 response types "
    r"(\Cref{tab:errortypes}), with half penalties for disagreements. "
    r"$\pm$: half-width of 95\% confidence intervals from 1,000 bootstrap "
    r"resamples. \textbf{Bold}: highest agreement with human labels per setting. "
    r"$\Delta$\,acc.: second minus first rater's accuracy (pp). "
    r"The table's jury takes the lower judge score and that judge's type "
    r"(DeepSeek's on a tie), without using human labels; correctness requires "
    r"both judges to assign score~3. Disagreements are included here, "
    r"rather than left unscored as in official scoring."
)


def kappa_rows(frame: pl.DataFrame, pairs: tuple[str, ...]) -> dict[str, dict]:
    """Return kappa estimates and intervals by rater pair and level."""
    kappa = frame.filter((pl.col("statistic") == "cohen_kappa") & pl.col("raters").is_in(pairs))
    table: dict[str, dict] = {pair: {} for pair in pairs}
    for row in kappa.iter_rows(named=True):
        table[row["raters"]][row["level"]] = (row["estimate"], row["low"], row["high"])
    return table


def _rates(frame: pl.DataFrame) -> dict[str, float]:
    """Return each rater's correct rate."""
    rates = frame.filter(pl.col("statistic") == "correct_rate")
    return dict(zip(rates["rater"], rates["estimate"], strict=False))


def _best(kappa: dict[str, dict]) -> dict[str, float]:
    """Highest rounded kappa per level among pairs that include a human."""
    human_pairs = [pair for pair in kappa if any(rater in HUMANS for rater in pair.split("~"))]
    return {
        level: max(
            (round(kappa[pair][level][0], 2) for pair in human_pairs),
            default=float("nan"),
        )
        for level in LEVELS
    }


def _cell(value: tuple[float, float, float], bold: bool, spread: bool) -> str:
    """Format a kappa as ``point`` or ``point $\\pm$ half-width``, maybe bold."""
    point, low, high = value
    text = f"{point:.2f}"
    if bold:
        text = rf"\textbf{{{text}}}"
    if not spread:
        return text
    return rf"{text}\textcolor{{black!55}}{{\,$\pm$\,{(high - low) / 2:.2f}}}"


def _format_row(pair: str, kappa: dict, rates: dict[str, float], best: dict[str, float]) -> str:
    """Format one rater pair as a LaTeX row."""
    first, second = pair.split("~")
    label = (
        JUDGE_PAIR_LABEL if pair == JUDGE_PAIR else f"{RATER_LABELS[first]}--{RATER_LABELS[second]}"
    )
    has_human = first in HUMANS or second in HUMANS
    cells = [
        _cell(
            kappa[level],
            bold=has_human and round(kappa[level][0], 2) == best[level],
            spread=level != "fine_tree",
        )
        for level in LEVELS
    ]
    delta = f"${(rates[second] - rates[first]) * 100:+.1f}$"
    return f"{label} & {' & '.join(cells)} & {delta} \\\\"


def _heading(text: str) -> str:
    return rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{text}}}}} \\"


def _group(pair: str) -> str:
    """Row group of a rater pair: both humans, one human, or both judges."""
    humans = [rater for rater in pair.split("~") if rater in HUMANS]
    return "humans" if len(humans) == 2 else humans[0] if humans else "judges"


def _base_rate(rates: dict[str, float]) -> str:
    """Accuracy of H1, or the judges' mean accuracy without H1."""
    if "H1" in rates:
        return rf"H1 accuracy {rates['H1'] * 100:.0f}\%"
    mean = sum(rates[judge] for judge in JUDGE_PAIR.split("~")) / 2
    return rf"judge accuracy {mean * 100:.0f}\%"


def _block(title: str, kappa: dict[str, dict], rates: dict[str, float]) -> list[str]:
    """Heading with the base rate, then one row per rater pair.

    Light rules separate human--human, each human against the judges, and
    judge--judge rows.
    """
    best = _best(kappa)
    lines = [_heading(f"{title}; {_base_rate(rates)}")]
    for index, pair in enumerate(kappa):
        if index and _group(pair) != _group(list(kappa)[index - 1]):
            lines.append(GROUP_RULE)
        lines.append(_format_row(pair, kappa[pair], rates, best))
    return lines


def doubly_annotated_block() -> list[str]:
    """Rows for the four-rater sample, including the judge ensemble."""
    pooled = (pl.col("sample") == "primary") & (pl.col("scope") == "Pooled")
    agreement = pl.read_csv(RATER_DIR / "agreement.csv").filter(pooled)
    rates = _rates(pl.read_csv(RATER_DIR / "marginals.csv").filter(pooled))
    units = agreement["units"].max()
    title = f"A. Doubly annotated, {units:,} pairs"
    return _block(title, kappa_rows(agreement, DOUBLY_PAIRS), rates)


def setting_block(setting: str) -> list[str]:
    """Rows for a judge-reliability setting."""
    frame = pl.read_csv(RATER_DIR / "judge_reliability.csv").filter(pl.col("setting") == setting)
    title = SETTING_TITLES[setting].format(units=frame["units"].max())
    return _block(title, kappa_rows(frame, SETTING_PAIRS[setting]), _rates(frame))


def render_judge_agreement_latex() -> str:
    """Render the three-setting agreement table as a LaTeX float."""
    body = [
        *doubly_annotated_block(),
        r"\midrule",
        *setting_block("stratified"),
        r"\midrule",
        *setting_block("pending"),
    ]
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"  \centering",
            r"  \footnotesize",
            r"  \setlength{\tabcolsep}{3pt}",
            r"  \setlength{\aboverulesep}{0pt}",
            r"  \setlength{\belowrulesep}{0pt}",
            r"  \renewcommand{\arraystretch}{1.15}",
            r"  \begin{tabular}{@{}l|ccc|r@{}}",
            r"    \toprule",
            r"    & \multicolumn{3}{c|}{Cohen's $\kappa$} & \\",
            r"    Raters & Accuracy & Score 0--3 & Type & $\Delta$\,acc. \\",
            r"    \midrule",
            *(f"    {line}" for line in body),
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \moveups",
            rf"  \caption{{{CAPTION}}}",
            r"  \vspace{-2em}",
            r"  \label{tab:judge-agreement}",
            r"\end{table}",
            "",
        ]
    )


def main() -> None:
    """Write the judge-agreement table."""
    write_table("judge_agreement", render_judge_agreement_latex())


if __name__ == "__main__":
    main()
