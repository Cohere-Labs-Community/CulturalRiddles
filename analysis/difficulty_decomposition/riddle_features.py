"""Derive riddle and answer features and fit their effect on a correct answer.

Features come from each riddle's English translation (form, clue structure,
length) and its English reference answer (how common its rarest word is in
English). This module writes the answer-level frame for the 57 human-evaluated
communities and, with ``--fit``, runs ``fit_riddle_features.R``, whose odds
ratios feed the paper's riddle-feature table.

Run from the repository root after ``prepare`` has written the analysis frame::

    uv run python -m analysis.difficulty_decomposition.riddle_features
    uv run python -m analysis.difficulty_decomposition.riddle_features --fit
"""

from __future__ import annotations

import argparse
import re
import subprocess

import numpy as np
import pandas as pd
from wordfreq import zipf_frequency

from analysis.constants import RIDDLES_PATH
from analysis.difficulty_decomposition.fit_provenance import (
    record_fit_provenance,
    validate_fit_provenance,
)
from analysis.difficulty_decomposition.prepare import (
    FRAME_PATH,
    OUTPUT_DIR,
    PACKAGE_DIR,
)
from analysis.riddle_analysis.surface_forms import QUESTION_FORMS, classify_form
from analysis.utils import write_csv

FEATURE_FRAME_PATH = FRAME_PATH.with_name("riddle_feature_frame.csv")
R_SCRIPT = PACKAGE_DIR / "fit_riddle_features.R"

# Zipf bands on the answer's rarest English word: below 1.5 is almost always a
# transliterated local term or name ("Oyilattam", "Pliska"); 3 and above is a
# common word ("violin", "hide and seek"); the band between is mixed.
LOCAL_TERM_MAX_ZIPF = 1.5
COMMON_WORD_MIN_ZIPF = 3.0

CONTRAST = re.compile(
    r"\b(?:but|yet|though|although|without|not|never|no|neither|nor)\b", re.IGNORECASE
)
SENTENCE_BREAK = re.compile(r"[.?!;\n]+")
PARENTHETICAL = re.compile(r"\(.*?\)|\[.*?\]")
WORD = re.compile(r"[^\W\d_]+")
LEADING_ARTICLE = re.compile(r"^(?:a|an|the)\s+")
PLURAL_ENDING = re.compile(r"(?:es|s)$")


def answer_zipf(answer: str) -> float:
    """Score how familiar an English reference answer is to English readers.

    Args:
        answer: English reference answer, possibly with parentheticals.

    Returns:
        Zipf frequency of the answer's rarest word; 0 for transliterated local
        terms and NaN when the answer has no alphabetic words (years, numbers).
    """
    words = WORD.findall(PARENTHETICAL.sub(" ", answer).lower())
    if not words:
        return np.nan
    return min(zipf_frequency(word, "en") for word in words)


def answer_concept(answer: str) -> str:
    """Normalize an English reference answer so that shared answers match.

    Args:
        answer: English reference answer.

    Returns:
        Lowercase answer without parentheticals, a leading article, trailing
        punctuation, or a plural ending ("The kites." becomes "kite").
    """
    text = PARENTHETICAL.sub("", answer).lower().strip(" .!")
    return PLURAL_ENDING.sub("", LEADING_ARTICLE.sub("", text))


def riddle_features() -> pd.DataFrame:
    """Derive riddle-pattern and answer-type features for every riddle.

    Form is the coarse grouping of the five surface forms of
    ``riddle_analysis.surface_forms`` (read from the English translation because
    the originals span 40+ scripts); line breaks are read from the original,
    where verse layout lives.

    Returns:
        One row per riddle with form, clue-structure, length, and answer-type
        features, the categorical ``riddle_form`` and ``answer_kind``, and the
        normalized ``answer_concept``.
    """
    corpus = pd.read_json(RIDDLES_PATH, lines=True, dtype={"reference_answer_english": str})[
        ["riddle_id", "riddle_original", "riddle_english", "reference_answer_english"]
    ].dropna(subset=["riddle_english", "reference_answer_english"])
    text = corpus["riddle_english"].str.strip()
    surface_form = text.map(classify_form)
    is_question = surface_form.isin(QUESTION_FORMS)
    features = pd.DataFrame(
        {
            "riddle_id": corpus["riddle_id"],
            "surface_form": surface_form,
            "n_clauses": text.map(
                lambda riddle: sum(bool(s.strip()) for s in SENTENCE_BREAK.split(riddle))
            ),
            "has_contrast": text.str.contains(CONTRAST),
            "multiline": corpus["riddle_original"].str.strip().str.contains("\n"),
            "log2_words": np.log2(text.str.split().str.len()),
            "answer_zipf": corpus["reference_answer_english"].map(answer_zipf),
            "answer_concept": corpus["reference_answer_english"].map(answer_concept),
        }
    )
    features["multi_clue"] = features["n_clauses"].ge(2) | features["has_contrast"]
    features["riddle_form"] = np.select(
        [is_question, surface_form.eq("first_person_statement")],
        ["question", "first_person_statement"],
        default="third_person_description",
    )
    features["answer_kind"] = np.select(
        [
            features["answer_zipf"].isna(),
            features["answer_zipf"].lt(LOCAL_TERM_MAX_ZIPF),
            features["answer_zipf"].ge(COMMON_WORD_MIN_ZIPF),
        ],
        ["numeric", "local_term", "common_word"],
        default="mid_frequency",
    )
    return features


def main(fit: bool = False) -> None:
    """Write the feature frame, then fit or validate the cached R models.

    Args:
        fit: Run the R mixed-model fits (about 10--20 minutes).

    Raises:
        subprocess.CalledProcessError: If the R fitting script fails.
        RuntimeError: If replayed fit summaries have missing or stale provenance.
    """
    frame = pd.read_csv(FRAME_PATH, low_memory=False)
    sample = frame[frame["complete_block"] & frame["is_scored"] & ~frame["grader_flag"]].merge(
        riddle_features(), on="riddle_id", how="inner", validate="m:1"
    )
    write_csv(sample, FEATURE_FRAME_PATH)
    if fit:
        subprocess.run(
            ["Rscript", str(R_SCRIPT), str(FEATURE_FRAME_PATH), str(OUTPUT_DIR)],
            check=True,
        )
        record_fit_provenance(
            "riddle_features",
            frame_path=FEATURE_FRAME_PATH,
            script_path=R_SCRIPT,
            output_dir=OUTPUT_DIR,
        )
    else:
        validate_fit_provenance(
            "riddle_features",
            frame_path=FEATURE_FRAME_PATH,
            script_path=R_SCRIPT,
            output_dir=OUTPUT_DIR,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write the riddle-feature frame and optionally fit the R models."
    )
    parser.add_argument("--fit", action="store_true", help="Also run the R fits.")
    main(fit=parser.parse_args().fit)
