"""Load the approved aggregate writing-survey tables."""

from __future__ import annotations

import pandas as pd

from analysis.constants import (
    RIDDLE_WRITING_OUTPUT_DIR,
    WRITING_SURVEY_AGGREGATES_DIR,
)

DATA_DIR = WRITING_SURVEY_AGGREGATES_DIR
OUTPUT_DIR = RIDDLE_WRITING_OUTPUT_DIR
AGGREGATE_COLUMNS: dict[str, tuple[str, ...]] = {
    "survey_audit.csv": (
        "responses",
        "communities",
        "single_respondent_communities",
        "confidence_ratings_4_or_5",
        "minimum_topic_responses",
        "maximum_topic_responses",
    ),
    "topic_difficulty.csv": ("topic", "mean_difficulty", "responses"),
    "specificity_strategy.csv": ("strategy", "respondents"),
    "resources_used.csv": ("resource", "respondents"),
    "ai_use_functions.csv": ("use", "respondents"),
}


def load_aggregates() -> dict[str, pd.DataFrame]:
    """Read every approved source table; missing inputs are errors.

    The approved aggregate tables are the pipeline's only survey inputs.
    """
    tables = {}
    for name, columns in AGGREGATE_COLUMNS.items():
        table = pd.read_csv(DATA_DIR / name)
        if tuple(table.columns) != columns:
            raise ValueError(f"Unexpected aggregate columns in {name}: {list(table.columns)}")
        tables[name] = table
    return tables
