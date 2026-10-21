"""Classify the surface form of each benchmarked riddle and plot the mix per community.

This module owns the riddle surface-form definitions. Population: the riddles
that were benchmarked (``numbenchmarkedriddles``) and have an
English text: the English translation, or the original for the English-language
communities, whose riddles have no translation. Untranslated verse riddles are
left out. All patterns read that English text, because the originals span many
scripts, so a form is a property of the translation (translators may add a
closing "?").

Each riddle gets exactly one form, the first rule that matches. Question forms
are read from the riddle's closing sentence only, so a "?" or a question word
earlier in the riddle does not make it a question:

1. *Who am I?* -- a question whose closing question is in the first person.
2. *What is it?* -- a question whose closing question only asks to name the
   thing described ("What is it?", "Who is he?", "Which fair is this?").
3. *Other question* -- any other riddle whose closing sentence is a question.
4. *First-person statement* -- not a question, but speaks as "I" or "we"
   anywhere in the riddle (the voice is not confined to one sentence).
5. *Third-person description* -- everything else.

The closing sentence is the last sentence of the normalized text (Unicode NFKC,
invisible characters removed, "؟" read as "?"). Trailing answer-format notes
("Answer in three words.", "Guess an idiom.", "(only four words)") are skipped
because they are not part of the riddle's own form. A line break ends a
sentence unless the previous line ends in a comma, semicolon, colon or dash or
the next line starts in lower case, so verse questions that wrap over lines
stay in one piece. The closing sentence is a *question* if it contains a "?",
opens with a question word (and is not an exclamation), or is a bare
"Who am I" / "What is it" style phrase. Forms 1-3 refine the difficulty
model's *question*, and forms 4-5 are its other two forms
(``difficulty_decomposition.riddle_features`` calls :func:`classify_form`).

The classification is plain regular expressions over the text: no embeddings,
no LLM, no sampling. The same text always gets the same form, and the
per-riddle labels are committed (``surface_forms_by_riddle.csv``) so any change
to the rules shows up as a diff.

Run from the repository root::

    uv run python -m analysis.riddle_analysis.surface_forms
"""

from __future__ import annotations

import re
import unicodedata

import polars as pl

from analysis.constants import RIDDLE_ANALYSIS_OUTPUT_DIR
from analysis.corpus import load_corpus_metadata
from analysis.figures._common import C
from analysis.model_results.evaluation_data import load_flat_frame
from analysis.riddle_analysis.corpus_data import load_corpus
from analysis.utils import write_csv

QUESTION_WORD = r"(?:what|who|whom|whose|which|where|why|how|in which)"
QUESTION_START = re.compile(rf"^\W*{QUESTION_WORD}\b", re.IGNORECASE)
# "us" is case-sensitive so the country "US" does not count, and "I" after
# "War" is the Roman numeral ("World War I").
FIRST_PERSON = re.compile(
    r"(?<!War )\b(?:i|i'm|i’m|i am|me|my|mine|myself|we|our|ours|ourselves)\b"
    r"|(?-i:\bus\b)",
    re.IGNORECASE,
)
INVISIBLE = re.compile("[\u200b-\u200d\u2060\ufeff]")
ARABIC_QUESTION_MARK = str.maketrans("؟", "?")
# A line break continues the sentence when the line before ends in these.
LINE_CONTINUES = re.compile(r"[,;:—–-]$")
# End of sentence: terminal punctuation, then whitespace. After closing quotes
# or brackets the sentence continues if the next word is lower case
# (``“Why did I cry?” in the middle of the sea``).
CLOSERS = "\"”’')]»"
SENTENCE_END = re.compile(rf"(?<=[.?!…])(?:\s+|[{re.escape(CLOSERS)}]+\s+(?![a-z]))")
CLAUSE_BREAK = re.compile(r"\s*[,;:—–]\s*|\s+-\s+")
# Sentences after the riddle proper: answer-format instructions and notes.
ANSWER_NOTE = re.compile(
    r"^\W*(?:answer|please|enter|hint|bonus|note|tip)\b|^\W*guess\s+(?:a|an|the)\b",
    re.IGNORECASE,
)
PARENTHETICAL = re.compile(r"[(\[][^()\[\]]*[)\]]\W*")
# A closing question that only asks to name the described thing: a question
# word, up to three words ("what sport"), "is/are/was/were", a pronoun ("it",
# "this", "he", ...) and up to two more words ("called", "place"); or a bare
# "Who?"; or "What"/"Which" with up to two noun-like words ("What old
# profession?"). Words that are verbs or pronouns ("What is Kakar?", "Which one
# survived?") do not qualify, so the boundary does not depend on word count.
NAME_IT = re.compile(
    r"^\W*(?:(?:and|so|then)\s+)?(?:what|who|which|whose)\b(?:\W+[\w'’-]+){0,3}?"
    r"\W+(?:is|are|was|were)\W+(?:it|this|that|he|she|they|these|those|you)\b"
    r"(?:\W+[\w'’-]+){0,2}\W*$"
    r"|^\W*(?:what|who|which)\W*$"
    r"|^\W*(?:what|which)(?:\W+(?!(?:is|are|was|were|do|does|did|has|have|had|can"
    r"|will|would|it|this|that|he|she|they|you|i|we)\b)[\w'’-]+(?<!ed)){1,2}\W*$",
    re.IGNORECASE,
)
WHO_AM_I = re.compile(r"^\W*(?:who|what)\W+(?:am\W+i|are\W+we)\W*$", re.IGNORECASE)
FORMS: dict[str, tuple[str, str]] = {
    "who_am_i": ("Who am I?", C["blue"]),
    "what_is_it": ("What is it?", C["vermillion"]),
    "other_question": ("Other question", C["orange"]),
    "first_person_statement": ("First-person statement", C["sky"]),
    "third_person_description": ("Third-person description", C["lightgrey"]),
}
QUESTION_FORMS = ("who_am_i", "what_is_it", "other_question")
# One community dominated by each form, for the compact figure.
EXEMPLAR_COMMUNITIES = (
    "luganda_uganda",
    "japanese_japan",
    "vietnamese_vietnam",
    "arabic_morocco",
    "chinese_singapore",
)
OVERALL = "Overall"
LABEL_MIN_SHARE = 8.0
SEGMENT_FONT_SIZE = 8.5
TICK_FONT_SIZE = 9.0
LEGEND_FONT_SIZE = 8.5
SHARES_PATH = RIDDLE_ANALYSIS_OUTPUT_DIR / "surface_forms_by_community.csv"
LABELS_PATH = RIDDLE_ANALYSIS_OUTPUT_DIR / "surface_forms_by_riddle.csv"
ENGLISH_LANGUAGE_CODE = "eng"


def normalize_text(text: str) -> str:
    """Make riddle text comparable: NFKC, no invisible characters, one newline style.

    Args:
        text: English riddle text.

    Returns:
        The text with fullwidth and Arabic question marks read as "?",
        no-break spaces as spaces, zero-width characters and carriage returns
        removed, and surrounding whitespace stripped.
    """
    text = unicodedata.normalize("NFKC", text).translate(ARABIC_QUESTION_MARK)
    return INVISIBLE.sub("", text).replace("\r", "").strip()


def split_sentences(text: str) -> list[str]:
    """Split normalized riddle text into sentences.

    Args:
        text: Output of :func:`normalize_text`.

    Returns:
        The non-empty sentences in order. Line breaks end a sentence unless the
        line before ends in a comma, semicolon, colon or dash, or the next line
        starts in lower case.
    """
    lines: list[str] = []
    for line in (raw.strip() for raw in text.split("\n")):
        if not line:
            continue
        if lines and (LINE_CONTINUES.search(lines[-1]) or line[0].islower()):
            lines[-1] += " " + line
        else:
            lines.append(line)
    return [
        sentence.strip()
        for line in lines
        for sentence in SENTENCE_END.split(line)
        if sentence.strip()
    ]


def closing_sentence(text: str) -> str:
    """Return the sentence that closes a riddle.

    It is the last sentence once trailing answer-format notes ("Answer in
    three words.", "Guess an idiom.", a parenthetical) are dropped.

    Args:
        text: English riddle text.

    Returns:
        The closing sentence, or the normalized text if it has no sentence.
    """
    text = normalize_text(text)
    sentences = split_sentences(text)
    while len(sentences) > 1 and (
        ANSWER_NOTE.match(sentences[-1]) or PARENTHETICAL.fullmatch(sentences[-1])
    ):
        sentences.pop()
    return sentences[-1] if sentences else text


def closing_question(text: str) -> str:
    """Return the question that closes a riddle.

    It is the closing sentence, cut to the clause after the last comma or dash
    when that clause opens with a question word ("..., who am I?" gives
    "who am I?").

    Args:
        text: English riddle text.

    Returns:
        The closing question, without surrounding whitespace.
    """
    sentence = closing_sentence(text)
    clause = CLAUSE_BREAK.split(sentence)[-1]
    return clause if QUESTION_START.match(clause) else sentence


def classify_form(text: str) -> str:
    """Assign one English riddle text to one surface form.

    Args:
        text: English riddle text.

    Returns:
        A key of :data:`FORMS`.
    """
    text = normalize_text(text)
    sentence = closing_sentence(text)
    closing = closing_question(text)
    is_question = (
        "?" in sentence
        or (bool(QUESTION_START.match(sentence)) and not sentence.rstrip(CLOSERS).endswith("!"))
        or bool(NAME_IT.match(closing))
        or bool(WHO_AM_I.match(closing))
    )
    if is_question and FIRST_PERSON.search(closing):
        return "who_am_i"
    if is_question and NAME_IT.match(closing):
        return "what_is_it"
    if is_question:
        return "other_question"
    if FIRST_PERSON.search(text):
        return "first_person_statement"
    return "third_person_description"


def benchmarked_riddle_texts() -> pl.DataFrame:
    """Read the English text of every benchmarked riddle that has one.

    Returns:
        One row per riddle with ``riddle_id``, ``community`` and ``text``: the
        English translation, or the original in the English-language
        communities. Benchmarked riddles with no English text (untranslated
        verse) are dropped.
    """
    corpus = load_corpus()
    benchmarked = set(load_flat_frame(with_judges=False)["riddle_id"])
    corpus = corpus[corpus["riddle_id"].isin(benchmarked)]
    english = corpus["language_code"].eq(ENGLISH_LANGUAGE_CODE)
    text = corpus["riddle_english"].where(~english, corpus["riddle_original"]).str.strip()
    return pl.from_pandas(corpus.assign(text=text)[["riddle_id", "community", "text"]]).filter(
        pl.col("text").is_not_null() & (pl.col("text") != "")
    )


def classify_forms() -> pl.DataFrame:
    """Assign each benchmarked riddle with an English text to one surface form.

    Returns:
        One row per riddle with ``riddle_id``, ``community``, ``text``,
        ``form`` (a key of ``FORMS``) and ``closing``, the closing question the
        form was read from.
    """
    texts = benchmarked_riddle_texts()
    return texts.with_columns(
        pl.col("text").map_elements(classify_form, return_dtype=pl.String).alias("form"),
        pl.col("text").map_elements(closing_question, return_dtype=pl.String).alias("closing"),
    )


def form_shares(forms: pl.DataFrame) -> pl.DataFrame:
    """Compute each community's percentage of riddles in every surface form.

    Args:
        forms: Output of :func:`classify_forms`.

    Returns:
        One row per community plus an ``Overall`` row (all riddles pooled),
        with a ``label``, the ``code_label`` such as ``lug (UGA)``, the riddle
        count ``n``, and one percentage column per form. Communities are sorted
        by their share of first-person riddles, highest first.
    """
    metadata = pl.from_pandas(
        load_corpus_metadata()[
            ["community", "language", "country", "language_code", "country_code"]
        ]
    )
    overall = forms.with_columns(pl.lit(OVERALL).alias("community"))
    counts = (
        pl.concat([forms.select("community", "form"), overall.select("community", "form")])
        .group_by("community", "form")
        .len()
        .pivot(on="form", index="community", values="len")
        .fill_null(0)
        .select("community", *FORMS)
    )
    total = pl.sum_horizontal(*FORMS)
    return (
        counts.with_columns(total.alias("n"), *(pl.col(form) / total * 100 for form in FORMS))
        .join(metadata, on="community", how="left")
        .with_columns(
            pl.when(pl.col("community") == OVERALL)
            .then(pl.lit(OVERALL))
            .otherwise(pl.format("{} ({})", "language", "country"))
            .alias("label"),
            pl.when(pl.col("community") == OVERALL)
            .then(pl.lit(OVERALL))
            .otherwise(pl.format("{} ({})", "language_code", "country_code"))
            .alias("code_label"),
            (pl.col("community") == OVERALL).alias("is_overall"),
            (pl.col("who_am_i") + pl.col("first_person_statement")).alias("first_person"),
        )
        .sort(["is_overall", "first_person", "label"], descending=[True, True, False])
        .select("community", "label", "code_label", "n", *FORMS)
    )


def main() -> None:
    """Write per-riddle labels and per-community form shares."""
    forms = classify_forms()
    write_csv(
        forms.select("riddle_id", "community", "form", "closing").sort("riddle_id").to_pandas(),
        LABELS_PATH,
    )
    shares = form_shares(forms)
    write_csv(shares.to_pandas().round(2), SHARES_PATH)


if __name__ == "__main__":
    main()
