"""Load the auditable model-name vocabulary used by analysis outputs."""

from __future__ import annotations

import pandas as pd

from analysis.constants import MODEL_NAME_MAPPING_PATH

MODEL_NAME_COLUMNS: tuple[str, ...] = (
    "source_name",
    "display_name",
    "default_reasoning",
    "actual_reasoning_effort",
    "total_parameter_b",
    "active_parameter_count_b",
    "access",
    "info_url",
    "cutoff_date",
)

ACCESS_CATEGORIES = frozenset({"proprietary", "open-weight", "open-source"})
REASONING_CATEGORIES = frozenset(
    {"N/A", "none", "enabled", "adaptive", "medium", "high", "unspecified"}
)


def load_model_name_mapping() -> pd.DataFrame:
    """Read and validate the model-name and parameter metadata catalog."""
    mapping = pd.read_csv(MODEL_NAME_MAPPING_PATH, dtype=str, keep_default_na=False).fillna("")
    if tuple(mapping.columns) != MODEL_NAME_COLUMNS:
        raise ValueError(f"Invalid model metadata file: {MODEL_NAME_MAPPING_PATH}")
    if mapping["source_name"].duplicated().any() or mapping["source_name"].eq("").any():
        raise ValueError(
            f"Model-name mapping has duplicate or empty source names: {MODEL_NAME_MAPPING_PATH}"
        )
    required_text = [
        "display_name",
        "default_reasoning",
        "actual_reasoning_effort",
        "info_url",
        "cutoff_date",
    ]
    if mapping[required_text].eq("").any().any():
        raise ValueError(f"Model mapping has empty required fields: {MODEL_NAME_MAPPING_PATH}")
    invalid_urls = ~mapping["info_url"].str.startswith("https://")
    if invalid_urls.any():
        raise ValueError(
            f"Model mapping has invalid info URLs for: "
            f"{sorted(mapping.loc[invalid_urls, 'source_name'])}"
        )
    invalid_cutoffs = ~mapping["cutoff_date"].str.fullmatch(
        r"(?:\d{4}-\d{2}(?:-\d{2})?|not publicly disclosed)"
    )
    if invalid_cutoffs.any():
        raise ValueError(
            f"Model mapping has invalid cutoff dates for: "
            f"{sorted(mapping.loc[invalid_cutoffs, 'source_name'])}"
        )
    invalid_access = ~mapping["access"].isin(ACCESS_CATEGORIES)
    if invalid_access.any():
        raise ValueError(
            f"Model mapping has invalid access categories for: "
            f"{sorted(mapping.loc[invalid_access, 'source_name'])}"
        )
    for column in ("default_reasoning", "actual_reasoning_effort"):
        invalid_reasoning = ~mapping[column].isin(REASONING_CATEGORIES)
        if invalid_reasoning.any():
            raise ValueError(
                f"Model mapping has invalid {column} categories for: "
                f"{sorted(mapping.loc[invalid_reasoning, 'source_name'])}"
            )
    for column in ("total_parameter_b", "active_parameter_count_b"):
        values = pd.to_numeric(mapping[column].replace("", pd.NA), errors="coerce")
        invalid = mapping[column].ne("") & (values.isna() | values.le(0))
        if invalid.any():
            raise ValueError(
                f"Model metadata has invalid {column} values for: "
                f"{sorted(mapping.loc[invalid, 'source_name'])}"
            )
    return mapping


def model_display_map() -> dict[str, str]:
    """Return source-name to canonical display-name mappings."""
    mapping = load_model_name_mapping()
    return dict(zip(mapping["source_name"], mapping["display_name"], strict=True))


def reasoning_category(default_reasoning: str, actual_reasoning_effort: str) -> str:
    """Return the reasoning setting to show readers.

    A recorded effort is shown as recorded. ``unspecified`` means the historical request did not
    set an effort, so the provider default applied: ``default`` for models that have a reasoning
    mode, and ``N/A`` for models that have none. It never means that reasoning was disabled.
    """
    if actual_reasoning_effort != "unspecified":
        return actual_reasoning_effort
    return "N/A" if default_reasoning == "N/A" else "default"


def model_parameter_metadata() -> pd.DataFrame:
    """Return one row per canonical model display with size metadata.

    The size is the published total parameter count in billions.  Sparse
    mixture-of-experts models also retain their published active count.
    """
    mapping = load_model_name_mapping()
    columns = [
        "display_name",
        "default_reasoning",
        "actual_reasoning_effort",
        "total_parameter_b",
        "active_parameter_count_b",
        "access",
        "info_url",
        "cutoff_date",
    ]
    metadata = mapping[columns].drop_duplicates("display_name")
    metadata = metadata.rename(columns={"display_name": "model_display"})
    metadata = metadata.rename(columns={"total_parameter_b": "model_size_b"})
    return metadata.assign(
        model_size_b=lambda data: pd.to_numeric(
            data["model_size_b"].replace("", pd.NA), errors="coerce"
        ),
        active_parameter_count_b=lambda data: pd.to_numeric(
            data["active_parameter_count_b"].replace("", pd.NA), errors="coerce"
        ),
    )
