"""Build the public Hugging Face dataset tables and card from the repository data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ID = "CohereLabsCommunity/CulturalRiddles"
REPO_URL = "https://github.com/Cohere-Labs-Community/CulturalRiddles"
DASHBOARD_URL = "https://huggingface.co/spaces/CohereLabsCommunity/CulturalRiddles"
PLAY_URL = "https://culturalriddles.ai/"

# Published table -> file inside the bundle. The riddles table stays the default.
TABLES = {
    "riddles": "data/riddles.parquet",
    "model_answers": "data/model_answers.parquet",
    "human_annotations": "data/human_annotations.parquet",
    "models": "data/models.parquet",
}

ANSWER_COLUMNS = {
    "riddle_id": "riddle_id",
    "community": "community",
    "topic": "topic",
    "model": "model",
    "model_display": "model_display",
    "model_answer": "model_answer",
    "answer_norm": "answer_norm",
    "score": "score",
    "scorer": "scorer",
    "routing": "routing",
    "error_type": "error_type",
    "error_type_label": "error_type_label",
    "annotator_id": "annotator_id",
    "validator_note": "annotator_note",
    "score_DeepSeek": "judge_score_deepseek",
    "score_Gemini": "judge_score_gemini",
}

ANNOTATION_COLUMNS = {
    "riddle_id": "riddle_id",
    "community": "community",
    "topic": "topic",
    "answer": "answer",
    "answer_norm": "answer_norm",
    "annotation": "annotation",
    "annotator_id": "annotator_id",
    "score": "score",
    "error_type": "error_type",
    "error_type_label": "error_type_label",
    "note": "annotator_note",
    "benchmark_status": "benchmark_status",
    "iaa_row": "iaa_row",
    "self_review": "self_review",
}

MODEL_COLUMNS = {
    "source_name": "model",
    "display_name": "model_display",
    "default_reasoning": "default_reasoning",
    "actual_reasoning_effort": "actual_reasoning_effort",
    "total_parameter_b": "total_parameter_b",
    "active_parameter_count_b": "active_parameter_count_b",
    "access": "access",
    "info_url": "info_url",
    "cutoff_date": "cutoff_date",
}

RIDDLE_FIELDS = [
    ("riddle_id", "string", "Stable unique ID, `{community}_{topic}_{number}`."),
    (
        "community",
        "string",
        "Community key, usually `{language}_{country}` (for example `tamil_india`).",
    ),
    ("language", "string", "ISO 639-3 code of the riddle's language."),
    ("topic", "string", "Topic the riddle belongs to."),
    ("is_wildcard", "bool", "True for a topic the community chose itself."),
    ("riddle_original", "string", "The riddle in its original language."),
    ("riddle_english", "string", "English translation, for human reference only. May be null."),
    ("riddle_hint", "string", "Optional hint. May be null."),
    (
        "riddle_note",
        "string",
        "Optional note from the writer; riddles flagged in the content review also carry a content note here. May be null.",
    ),
    ("reference_answer_original", "string", "Reference answer in the original language."),
    ("reference_answer_english", "string", "English translation of the answer. May be null."),
]

ANSWER_FIELDS = [
    ("riddle_id", "string", "Joins to `riddles`."),
    ("community", "string", "Community key."),
    ("topic", "string", "Topic of the riddle."),
    ("model", "string", "Model configuration key; joins to `models`."),
    ("model_display", "string", "Readable configuration name."),
    ("model_answer", "string", "The answer the model gave."),
    (
        "answer_norm",
        "string",
        "`model_answer` stripped and lowercased; joins to `human_annotations`.",
    ),
    (
        "score",
        "int",
        "0 incorrect, 1 admits not knowing, 2 partially correct, 3 correct. Null when no score was reached.",
    ),
    (
        "scorer",
        "string",
        "`rule`, `human` (native speaker), `llm_judges` (two agreeing judges), or `none`.",
    ),
    (
        "routing",
        "string",
        "How the answer was routed: rule-based categories, human evaluation, or pending and invalid human labels.",
    ),
    (
        "error_type",
        "string",
        "Answer type code, for example `0.3` hallucinated or `3.1` different valid answer.",
    ),
    ("error_type_label", "string", "Readable answer type."),
    (
        "annotator_id",
        "string",
        "Anonymous native-speaker ID, such as `annotator_k2p9`. Stable across tables.",
    ),
    ("annotator_note", "string", "The annotator's note, as written. May be null."),
    ("judge_score_deepseek", "int", "Score from the DeepSeek judge, where one ran. May be null."),
    ("judge_score_gemini", "int", "Score from the Gemini judge, where one ran. May be null."),
]

ANNOTATION_FIELDS = [
    ("riddle_id", "string", "Joins to `riddles`."),
    ("community", "string", "Community key."),
    ("topic", "string", "Topic of the riddle."),
    ("answer", "string", "The answer the annotator labeled."),
    (
        "answer_norm",
        "string",
        "`answer` stripped and lowercased; one label covers every model that gave it.",
    ),
    ("annotation", "string", "`first` label, or `second` label from an independent annotator."),
    ("annotator_id", "string", "Anonymous native-speaker ID; several are separated by `;`."),
    ("score", "int", "0 to 3, as in `model_answers`."),
    ("error_type", "string", "Answer type code."),
    ("error_type_label", "string", "Readable answer type."),
    ("annotator_note", "string", "The annotator's note, as written. May be null."),
    (
        "benchmark_status",
        "string",
        "`valid` or `invalid` for first labels; invalid labels are not used in scores.",
    ),
    ("iaa_row", "float", "Set for rows in the inter-annotator agreement sample."),
    ("self_review", "bool", "True when an annotator labeled the same pair again."),
]

MODEL_FIELDS = [
    ("model", "string", "Configuration key used in `model_answers`."),
    ("model_display", "string", "Readable configuration name."),
    ("default_reasoning", "string", "The provider's documented default reasoning setting."),
    (
        "actual_reasoning_effort",
        "string",
        "The setting recorded for the run. `unspecified` means the request did not set it, not that reasoning was off.",
    ),
    (
        "total_parameter_b",
        "float",
        "Published total parameters, in billions. Null when not published.",
    ),
    (
        "active_parameter_count_b",
        "float",
        "Published active parameters for mixture-of-experts models, in billions.",
    ),
    ("access", "string", "`proprietary`, `open-weight`, or `open-source`."),
    ("info_url", "string", "Model card or documentation."),
    ("cutoff_date", "string", "Published training-data cutoff, where known."),
]


def _table(rows: list[tuple[str, str, str]]) -> str:
    lines = ["| Field | Type | Description |", "|---|---|---|"]
    lines += [f"| `{name}` | {kind} | {text} |" for name, kind, text in rows]
    return "\n".join(lines)


def build_tables(root: Path, riddles: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return every public table, keyed like TABLES."""
    scored = root / "data/derived/scoring/responses_scored_consolidated.parquet"
    if not scored.is_file():
        raise FileNotFoundError(
            f"Scored answers are missing: {scored}. Run `uv run python -m analysis.run_all` first."
        )
    answers = pd.read_parquet(scored)[list(ANSWER_COLUMNS)].rename(columns=ANSWER_COLUMNS)
    for column in ("score", "judge_score_deepseek", "judge_score_gemini"):
        answers[column] = answers[column].astype("Int8")
    annotations = pd.read_parquet(root / "data/canonical/human_eval/annotations.parquet")
    annotations = annotations[list(ANNOTATION_COLUMNS)].rename(columns=ANNOTATION_COLUMNS)
    models = pd.read_csv(root / "data/canonical/model_mapping.csv")
    models = models[list(MODEL_COLUMNS)].rename(columns=MODEL_COLUMNS)
    unknown = set(answers["riddle_id"]) - set(riddles["riddle_id"])
    if unknown:
        raise ValueError(
            f"Model answers refer to riddles outside the release: {sorted(unknown)[:5]}"
        )
    unknown = set(annotations["riddle_id"]) - set(riddles["riddle_id"])
    if unknown:
        raise ValueError(f"Annotations refer to riddles outside the release: {sorted(unknown)[:5]}")
    if set(answers["model"]) != set(models["model"]):
        raise ValueError("Model answers and the model table list different configurations.")
    return {
        "riddles": riddles,
        "model_answers": answers,
        "human_annotations": annotations,
        "models": models,
    }


def contributor_credits(root: Path, metadata: pd.DataFrame, communities: list[str]) -> str:
    """List writers who consented to be credited, by community.

    ``docs/contributors.csv`` has one row per (community, name), using the name each writer
    chose, and an optional note such as a published source the riddles draw on. Writers who
    asked not to be named, or have not answered yet, are not in the file.
    """
    credits = pd.read_csv(root / "docs/contributors.csv", dtype=str, keep_default_na=False)
    if list(credits.columns) != ["community", "name", "note"]:
        raise ValueError("docs/contributors.csv must have the columns community,name,note.")
    unknown = sorted(set(credits["community"]) - set(communities))
    if unknown:
        raise ValueError(f"docs/contributors.csv names communities outside the release: {unknown}")
    labels = (metadata["language"] + " (" + metadata["country"] + ")").set_axis(
        metadata["community"]
    )
    lines = []
    for community, group in sorted(
        credits.groupby("community"), key=lambda item: labels.get(item[0], item[0])
    ):
        names = " · ".join(
            name.strip() + (f" ({note.strip()})" if note.strip() else "")
            for name, note in sorted(zip(group["name"], group["note"]))
        )
        lines.append(f"- **{labels.get(community, community)}:** {names}")
    return "\n".join(lines)


def render_card(root: Path, tables: dict[str, pd.DataFrame]) -> str:
    riddles, answers = tables["riddles"], tables["model_answers"]
    annotations, models = tables["human_annotations"], tables["models"]
    metadata = pd.read_csv(root / "data/canonical/corpus_metadata.csv")
    people = pd.read_csv(root / "data/canonical/contributor_counts.csv").set_index("contribution")[
        "people"
    ]
    communities = sorted(riddles["community"].unique())
    metadata = metadata[metadata["community"].isin(communities)]
    languages = sorted(riddles["language"].unique())
    countries = metadata["country"].nunique()
    wildcard = int(riddles["is_wildcard"].sum())
    answered = answers["riddle_id"].nunique()
    answered_communities = answers["community"].nunique()
    by_scorer = answers["scorer"].value_counts()
    first = annotations[annotations["annotation"].eq("first")]
    second = annotations[annotations["annotation"].eq("second")]
    writers, annotators = int(people["riddle_writers"]), int(people["annotators"])
    language_yaml = "\n".join(f"- {code}" for code in languages)
    credits = contributor_credits(root, metadata, communities)
    configs = "\n".join(
        f"- config_name: {name}\n"
        + ("  default: true\n" if name == "riddles" else "")
        + f"  data_files:\n  - split: test\n    path: {path}"
        for name, path in TABLES.items()
    )
    return f"""---
license: other
license_name: culturalriddles-evaluation-only
license_link: LICENSE
pretty_name: Cultural Riddles
language:
{language_yaml}
task_categories:
- question-answering
multilinguality:
- multilingual
annotations_creators:
- expert-generated
language_creators:
- expert-generated
source_datasets:
- original
tags:
- riddles
- cultural-reasoning
- cultural-knowledge
- multilingual
- multicultural
- community-curated
- low-resource-languages
- human-evaluation
configs:
{configs}
---

# Cultural Riddles

**Explore the results:** [dashboard]({DASHBOARD_URL}) · **Play:** [try the riddles yourself]({PLAY_URL}) · **Code:** [{REPO_URL.removeprefix("https://")}]({REPO_URL})

A community-built benchmark of original cultural riddles from **{len(communities)} communities across {len(languages)} languages and {countries} countries**, written by {writers} native speakers. Each riddle describes something without naming it, so answering it takes cultural knowledge, not translation or web search. A native speaker recognises the answer at once; a model succeeds only if it carries that culture's knowledge.

The release has four tables:

| Table | Rows | What it holds |
|---|---:|---|
| `riddles` (default) | {len(riddles):,} | The riddles and their reference answers. |
| `model_answers` | {len(answers):,} | Every answer from {models["model"].nunique()} model configurations on the {answered:,} benchmarked riddles, with its final score and who scored it. |
| `human_annotations` | {len(annotations):,} | Every native-speaker label on a unique answer, including independent second labels. |
| `models` | {len(models)} | The model configurations, their sizes, and their recorded reasoning settings. |

## Why riddles

A riddle names nothing. "This red-crossed sphere's juice is not sweet for your stomach" means nothing to a search engine, but every Hungarian recognises Unicum, a famous bitters. A model that only knows the formal name misses it. Because each riddle is original, unpublished, and culturally specific, it resists memorisation and lookup.

## Languages and communities

The unit is the present-day national-linguistic community, for example Spanish in Argentina or Bengali in India. Languages and communities differ because several communities share a language: Argentine, Cuban, and Peruvian riddles are all in Spanish but draw on three cultures. {wildcard:,} riddles come from topics each community chose itself. See [the list of communities]({REPO_URL}/blob/main/docs/communities.md).

## Loading the data

```python
from datasets import load_dataset

riddles = load_dataset("{REPO_ID}", split="test")
answers = load_dataset("{REPO_ID}", "model_answers", split="test")
labels = load_dataset("{REPO_ID}", "human_annotations", split="test")
models = load_dataset("{REPO_ID}", "models", split="test")

tamil = riddles.filter(lambda r: r["community"] == "tamil_india")
```

## Tables

### `riddles`

{_table(RIDDLE_FIELDS)}

### `model_answers`

Each configuration answered each benchmarked riddle once ({answered:,} riddles from {answered_communities} communities). Scores come from rules for clear cases ({int(by_scorer.get("rule", 0)):,} answers), from native speakers ({int(by_scorer.get("human", 0)):,}), and from two LLM judges that agree ({int(by_scorer.get("llm_judges", 0)):,}); {int(by_scorer.get("none", 0)):,} answers have no score. Accuracy is the share of scored answers with score 3.

{_table(ANSWER_FIELDS)}

### `human_annotations`

Native speakers labeled each unique (riddle, answer) pair once, so one label can cover several models. {len(first):,} rows are first labels and {len(second):,} are independent second labels used to measure agreement. {annotators} annotators contributed; their IDs are anonymous.

{_table(ANNOTATION_FIELDS)}

### `models`

{_table(MODEL_FIELDS)}

## Curation

Every riddle is original human writing by native speakers writing for their own community. AI tools may be used for brainstorming or to test difficulty, but the riddles are written by people. English translations are for human reference only, never shown to a model. The data contains no names or contact details of writers or annotators.

## Intended use and terms

The data is released under the terms in [LICENSE](https://huggingface.co/datasets/{REPO_ID}/blob/main/LICENSE): Creative Commons Attribution 4.0 attribution terms combined with an evaluation-only restriction.

**This data must not be used to train, fine-tune, or distill models, nor as seeds for synthetic data generation.** It is an evaluation benchmark; training on it contaminates future measurement. This applies to the model answers and labels as much as to the riddles.

## Content note

Although we have filtered the dataset for harmful content, some riddles may still contain sensitive themes or potentially problematic ideas. These are presented solely as cultural and linguistic puzzles, not as endorsements of the views expressed or guidance for real-world behavior.

Riddles flagged in a content review carry a short content note in their `riddle_note` field. Model answers are published as generated and were not filtered: many are hallucinated, and some may be offensive.

## Contributors

This benchmark was proposed by **Károly Boczka** and **Julia Kreutzer**, and built by native-speaker contributors across the Cohere Labs Community.

### Riddle writers by community

Writers who agreed to be credited, by the name they chose. Some writers asked not to be named or have not confirmed yet; they are part of the Cohere Labs Community credit.

{credits}

## Updates

This card is generated from the [source repository]({REPO_URL}) on each release.

- This release adds the `model_answers`, `human_annotations`, and `models` tables. The riddles moved to `data/riddles.parquet` and no longer carry the `release_version` column.
- After a content review, some riddles were removed or corrected. Every change is listed in `data/corrections.csv` in the source repository.

## Citation

```bibtex
@misc{{cultural_riddles_2026,
  title        = {{Cultural Riddles}},
  author       = {{Boczka, K{{\\'a}}roly and Kreutzer, Julia and the Cohere Labs Community}},
  year         = {{2026}},
  publisher    = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{REPO_ID}}}}}
}}
```
"""
