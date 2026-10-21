"""Audit and visualize the complete v0--v2 riddle corpus.

This module reads the canonical JSONL release directly rather than a cached
spreadsheet export, so every result reflects the current 6,100-riddle corpus.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.constants import RIDDLE_ANALYSIS_OUTPUT_DIR, RIDDLES_PATH
from analysis.corpus import add_corpus_groups, load_corpus_metadata

DATASET_PATH = RIDDLES_PATH
TABLE_DIR = RIDDLE_ANALYSIS_OUTPUT_DIR

# Deterministic topic-level rules, deliberately expressed at the concept level
# instead of as a closed list of every label presently in the corpus.
#
# Ordering is a fixed global priority, not a per-string "primary referent"
# scan: when a topic matches more than one theme's pattern, the theme listed
# earliest in this tuple wins, regardless of where in the topic string that
# theme's keyword appears (e.g. "Festivals & Folk Culture" resolves to
# "Religion & Folklore" because that theme precedes "Society" here, not
# because "Festival" appears before "Culture" in the string).
#
# A few patterns include literal, non-generalizable fallbacks for specific
# existing corpus topics (proper nouns, non-English script, or idiosyncratic
# phrasing with no reusable concept keyword). These are marked "# exception:"
# and won't match differently-worded future topics on the same theme; the
# general concept keywords alongside them are what future topics should hit.
THEME_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "Food",
        re.compile(
            r"\b(?:food|cuisin\w*|gastronom\w*|gasronom\w*|kitchen|agricultur\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Nature",
        re.compile(r"\b(?:animal\w*|nature|plant\w*|insect\w*)\b", re.IGNORECASE),
    ),
    (
        "Language",
        re.compile(
            r"\b(?:cantonese|expression\w*|colloquial|idiom\w*|language|proverb\w*|"
            r"pun\w*|singlish|slang|spoonerism|dialect)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Media",
        re.compile(
            r"\b(?:cinema|hollywood|literature|movie\w*|music|pop(?:ular)?\s*culture|"
            r"tv\s*show\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Religion & Folklore",
        re.compile(
            r"\b(?:belief\w*|festival\w*|myth\w*|folklore|religion)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Health",
        re.compile(r"\b(?:anatomy|human body|body)\b", re.IGNORECASE),
    ),
    (
        "Science",
        re.compile(r"\b(?:infrastructure|science|technology|tech)\b", re.IGNORECASE),
    ),
    (
        "Material Culture",
        re.compile(
            r"\b(?:clothing|objects?|things?|instrument\w*|craft\w*|artifact\w*|"
            r"attire|etiquette|konbini|sento|onsen)\b",  # exception: konbini/sento/onsen
            re.IGNORECASE,
        ),
    ),
    (
        "Economy",
        re.compile(r"\b(?:econom(?:y|ics?)|work life)\b", re.IGNORECASE),
    ),
    (
        "Places",
        re.compile(
            r"\b(?:city|cities|capital\w*|landmark\w*|geography|travel|maps?|"
            r"moscow)\b",  # exception: moscow
            re.IGNORECASE,
        ),
    ),
    (
        "Heritage",
        re.compile(
            r"\b(?:heritage|tradition\w*|custom\w*|ceremon\w*|ancestr\w*)\b|"
            r"(?:boat[-\s]?folk|indus|manchhar|شادي|درياء)",  # exception: Sindhi-specific
            re.IGNORECASE,
        ),
    ),
    (
        # Bare "culture" resolves here, not to Food/Media/Religion & Folklore/
        # Material Culture/Heritage, because those themes above claim a
        # specific sense of "culture" (cuisine, pop culture, folk culture,
        # objects, tradition) first; an unqualified "culture" topic is this
        # theme's general/residual sense, which is why it sits alongside
        # identity- and social-group topics (Gen-Z, caste, stereotypes).
        "Society & Identity",
        re.compile(
            r"\b(?:culture|gen[-\s]?z|humanities|caste|relationship\w*|identity|"
            r"stereotype\w*|clever|big in)\b",  # exception: clever/"big in"
            re.IGNORECASE,
        ),
    ),
)
UNCLASSIFIED_THEME = "Unclassified"


def load_corpus(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the riddle corpus and check it covers exactly the metadata communities."""
    rows: list[dict[str, Any]] = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
    ]
    df = pd.DataFrame(rows)
    if df["riddle_id"].duplicated().any():
        raise ValueError("Canonical corpus contains duplicate riddle IDs")

    metadata = load_corpus_metadata()
    expected_communities = set(metadata["community"])
    observed_communities = set(df["community"])
    missing = expected_communities - observed_communities
    unexpected = observed_communities - expected_communities
    if missing or unexpected:
        raise ValueError(
            "Corpus and metadata communities differ; "
            f"missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r}"
        )
    return add_corpus_groups(df)


def wildcard_choices(df: pd.DataFrame) -> pd.DataFrame:
    """Return exactly two distinct wildcard selections per metadata community."""
    metadata = load_corpus_metadata()
    choices = df.loc[df["is_wildcard"], ["community", "topic"]].drop_duplicates()
    expected_communities = set(metadata["community"])
    observed_communities = set(choices["community"])
    missing = expected_communities - observed_communities
    unexpected = observed_communities - expected_communities
    if missing or unexpected:
        raise ValueError(
            "Wildcard selections and metadata communities differ; "
            f"missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r}"
        )
    selections_per_community = (
        choices.groupby("community").size().reindex(metadata["community"], fill_value=0)
    )
    if not selections_per_community.eq(2).all():
        invalid = selections_per_community[~selections_per_community.eq(2)].to_dict()
        raise ValueError(
            f"Every metadata community must contribute exactly two wildcard selections: {invalid!r}"
        )
    expected_selection_count = len(metadata) * 2
    if len(choices) != expected_selection_count:
        raise ValueError(
            "Wildcard selection count does not match metadata coverage: "
            f"expected={expected_selection_count}, observed={len(choices)}"
        )
    return choices.copy()


def classify_wildcard_theme(topic: str) -> tuple[str, str | None]:
    """Classify a wildcard topic with ordered semantic rules.

    New labels automatically receive the first matching broad theme. Labels
    outside the current rule vocabulary remain explicitly ``Unclassified`` for
    review instead of causing the analysis to fail.
    """
    for theme, rule in THEME_RULES:
        if rule.search(topic):
            return theme, rule.pattern
    return UNCLASSIFIED_THEME, None


def wildcard_theme(topic: str) -> str:
    """Return the adaptive broad theme for one wildcard topic."""
    return classify_wildcard_theme(topic)[0]
