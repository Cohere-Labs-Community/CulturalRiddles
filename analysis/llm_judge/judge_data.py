"""Read the released judge-validation pairs, verdicts, and run summaries.

Every setting lives in ``data/canonical/judge/<setting>/``:

- ``setting_a``: doubly annotated pairs (two humans and two judges).
- ``setting_b``: a stratified sample of human-labeled test pairs.
- ``setting_c``: every answer without a valid human label.

The English-translation condition keeps its verdicts in
``data/canonical/english_condition/``. Verdict tables hold one row per judge, example
condition (``zero`` or ``few10`` shots), and (riddle, normalized answer) pair.
"""

from __future__ import annotations

from pathlib import Path

import orjson
import polars as pl

from analysis.constants import ENGLISH_CONDITION_DIR, JUDGE_DATA_DIR

JUDGES = {"DeepSeek": "deepseek-v4.1-flash", "Gemini": "gemini-3.5-flash"}
NA_LABELS = ("N/A", "NA")
SETTINGS = ("setting_a", "setting_b", "setting_c")


def normalize(column: str) -> pl.Expr:
    """Strip and lowercase an answer, the key every label and verdict uses."""
    return pl.col(column).fill_null("").str.strip_chars().str.to_lowercase()


def verdict_path(setting: str) -> Path:
    """Verdict table of a judge setting or of the English condition."""
    if setting == "english":
        return ENGLISH_CONDITION_DIR / "verdicts.parquet"
    return JUDGE_DATA_DIR / setting / "verdicts.parquet"


def load_pairs(setting: str) -> pl.DataFrame:
    """Load the frozen pairs of one setting (text columns, as the sampler wrote)."""
    return pl.read_parquet(JUDGE_DATA_DIR / setting / "pairs.parquet")


def load_verdicts(setting: str, judge: str, shots: str = "zero") -> pl.DataFrame:
    """Load one judge's verdicts in one setting and example condition.

    Args:
        setting: ``setting_a``, ``setting_b``, ``setting_c``, or ``english``.
        judge: Key of ``JUDGES``.
        shots: ``zero`` or ``few10``.

    Returns:
        One row per (riddle, normalized answer) with the verdict fields and
        attempt counts.
    """
    return pl.read_parquet(verdict_path(setting)).filter(
        (pl.col("judge") == judge) & (pl.col("shots") == shots)
    )


def judge_labels(setting: str, judge: str, shots: str = "zero") -> pl.DataFrame:
    """Return one judge's ``score_<judge>`` and ``fine_<judge>`` labels.

    A verdict without an error category (``N/A``) takes its bare score as its
    fine label.
    """
    return load_verdicts(setting, judge, shots).select(
        "riddle_id",
        "answer_norm",
        pl.col("score").alias(f"score_{judge}"),
        pl.when(pl.col("error_category").is_in(NA_LABELS) | pl.col("error_category").is_null())
        .then(pl.col("score").cast(pl.String))
        .otherwise(pl.col("error_category"))
        .alias(f"fine_{judge}"),
    )


def has_condition(setting: str, shots: str) -> bool:
    """Whether a setting was judged under an example condition."""
    shots_present = pl.read_parquet(verdict_path(setting), columns=["shots"])
    return shots_present.filter(pl.col("shots") == shots).height > 0


def run_summaries() -> list[dict[str, object]]:
    """Settings, cost, and token use of every judge run (``runs.json``)."""
    return orjson.loads((JUDGE_DATA_DIR / "runs.json").read_bytes())
