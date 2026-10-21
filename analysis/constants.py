"""Shared paths, taxonomies, and stable labels for analysis scripts."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
CANONICAL_DIR = DATA_DIR / "canonical"
DERIVED_DIR = DATA_DIR / "derived"
RUNS_DIR = DATA_DIR / "runs"
CORPUS_METADATA_PATH = CANONICAL_DIR / "corpus_metadata.csv"
MODEL_NAME_MAPPING_PATH = CANONICAL_DIR / "model_mapping.csv"
COMMONCRAWL_STATS_CSV = DATA_DIR / "reference" / "language_commoncrawl_stats.csv"
CONTRIBUTOR_COUNTS_CSV = CANONICAL_DIR / "contributor_counts.csv"
# The riddle corpus; ``is_benchmarked`` marks the riddles the models answered.
RIDDLES_PATH = CANONICAL_DIR / "riddles.jsonl"
# One row per (model configuration, riddle): raw answer, parse status, tokens.
MODEL_RESPONSES_PATH = CANONICAL_DIR / "model_responses.parquet"
# Regenerated scores; the frozen recorded snapshot stays under canonical/.
SCORED_RESPONSES_PATH = DERIVED_DIR / "scoring" / "responses_scored_consolidated.parquet"
BENCHMARK_RESULTS_PATH = CANONICAL_DIR / "benchmark_results.parquet"
# Every human label on a unique (riddle, answer) pair, both annotation passes.
HUMAN_ANNOTATIONS_PATH = CANONICAL_DIR / "human_eval" / "annotations.parquet"
# Pairs and recorded verdicts of the three judge-validation settings.
JUDGE_DATA_DIR = CANONICAL_DIR / "judge"
ENGLISH_CONDITION_DIR = CANONICAL_DIR / "english_condition"
WRITING_SURVEY_AGGREGATES_DIR = CANONICAL_DIR / "riddle_writing_analysis"
FIGURES_DIR = REPO_ROOT / "figures"
PAPER_FIGURES_DIR = FIGURES_DIR / "paper"
PREPRINT_FIGURES_DIR = FIGURES_DIR / "preprint"
MODEL_RESULTS_OUTPUT_DIR = DERIVED_DIR / "model_results"
RIDDLE_ANALYSIS_OUTPUT_DIR = DERIVED_DIR / "riddle_analysis"
RIDDLE_WRITING_OUTPUT_DIR = DERIVED_DIR / "riddle_writing_analysis"
EVAL_ANALYSIS_OUTPUT_DIR = DERIVED_DIR / "eval_analysis"
DIFFICULTY_OUTPUT_DIR = DERIVED_DIR / "difficulty_decomposition"
TABLES_DIR = DERIVED_DIR / "tables"
CLAIMS_DIR = DERIVED_DIR / "claims"
# Regenerated row-level model matrices remain under the shared data tree.
BUILD_DIR = DERIVED_DIR / "build"

FIXED_TOPICS: tuple[str, ...] = (
    "Politics",
    "History",
    "Geography",
    "Arts",
    "Sports",
    "National Heritage",
    "Kid's World",
    "Everyday Life",
)
TOPICS: tuple[str, ...] = (*FIXED_TOPICS, "Wildcard")
SURVEY_TOPICS: tuple[str, ...] = (*FIXED_TOPICS, "Wild Card")
