"""Load auditable corpus metadata and attach it to analysis frames."""

from __future__ import annotations

import pandas as pd

from analysis.constants import CORPUS_METADATA_PATH

METADATA_COLUMNS: tuple[str, ...] = (
    "community",
    "language_code",
    "language",
    "country",
    "country_code",
    "language_group",
    "un_m49_subregion",
    "language_resourcedness",
    "language_family",
    "script_family",
    "is_human_evaluated",
    "riddle_writers",
)
REQUIRED_METADATA_VALUES: tuple[str, ...] = (
    "language_code",
    "language",
    "country",
    "country_code",
    "un_m49_subregion",
    "language_resourcedness",
    "language_family",
    "script_family",
)


def load_corpus_metadata() -> pd.DataFrame:
    """Read and validate the reviewable per-community metadata table."""
    metadata = pd.read_csv(CORPUS_METADATA_PATH)
    if tuple(metadata.columns) != METADATA_COLUMNS:
        raise ValueError(f"Invalid corpus metadata file: {CORPUS_METADATA_PATH}")
    if metadata["community"].duplicated().any() or metadata["community"].isna().any():
        raise ValueError(
            f"Corpus metadata has duplicate or empty communities: {CORPUS_METADATA_PATH}"
        )
    if metadata[list(REQUIRED_METADATA_VALUES)].isna().any().any():
        raise ValueError(f"Corpus metadata has empty required values: {CORPUS_METADATA_PATH}")
    if not pd.api.types.is_bool_dtype(metadata["is_human_evaluated"]):
        raise ValueError(
            f"Corpus metadata has a non-boolean is_human_evaluated field: {CORPUS_METADATA_PATH}"
        )
    invalid_language_codes = ~metadata["language_code"].str.fullmatch(r"[a-z]{3}")
    invalid_country_codes = ~metadata["country_code"].str.fullmatch(r"[A-Z]{3}")
    if invalid_language_codes.any() or invalid_country_codes.any():
        raise ValueError(
            "Corpus metadata contains invalid ISO 639-3 or ISO alpha-3 codes: "
            f"{CORPUS_METADATA_PATH}"
        )
    country_code_counts = metadata.groupby("country")["country_code"].nunique()
    if country_code_counts.gt(1).any():
        countries = sorted(country_code_counts[country_code_counts.gt(1)].index)
        raise ValueError(f"Countries have conflicting ISO alpha-3 codes: {countries}")
    # An ISO language code can cover benchmark-specific varieties or scripts.
    # Keep the display label at community granularity rather than requiring one
    # label per code (for example, ``zho`` has Traditional and Simplified
    # Chinese communities in this corpus).
    return metadata


def add_corpus_groups(frame: pd.DataFrame, *, community_column: str = "community") -> pd.DataFrame:
    """Attach canonical corpus metadata to an analysis frame."""
    metadata = load_corpus_metadata()
    metadata_columns = [column for column in METADATA_COLUMNS if column != "community"]
    grouped = frame.drop(columns=metadata_columns, errors="ignore").merge(
        metadata,
        left_on=community_column,
        right_on="community",
        how="left",
        validate="many_to_one",
    )
    if community_column != "community":
        grouped = grouped.drop(columns="community")
    if grouped[list(REQUIRED_METADATA_VALUES)].isna().any().any():
        missing = sorted(
            grouped.loc[
                grouped[list(REQUIRED_METADATA_VALUES)].isna().any(axis=1),
                community_column,
            ].unique()
        )
        raise ValueError(f"Missing corpus metadata assignments for: {missing}")
    return grouped
