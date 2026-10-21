"""Country geometry and coverage counts for the world maps."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from analysis.constants import REPO_ROOT

COUNTRY_GEOMETRY_PATH = REPO_ROOT / "data/reference/ne_110m_admin_0_countries.geojson"

# Natural Earth uses formal map names for a few countries that are shorter in
# corpus_metadata.csv. These aliases are deliberately kept local to the map
# renderer rather than changing the canonical metadata labels.
GEOMETRY_COUNTRY_ALIASES: dict[str, str] = {
    "United States of America": "United States",
    "Republic of Serbia": "Serbia",
}

# Singapore is too small to appear in the 1:110m Natural Earth polygons. Draw
# it as a point so every country represented in the metadata remains visible.
SMALL_COUNTRY_POINTS: dict[str, tuple[float, float]] = {
    "Singapore": (103.82, 1.35),
}


def country_language_coverage(metadata: pd.DataFrame) -> pd.DataFrame:
    """Aggregate distinct languages and communities represented per country.

    Args:
        metadata: Validated rows from ``corpus_metadata.csv``.

    Returns:
        A sorted table with one row per country. ``languages`` is the map
        value; ``communities`` provides a separate audit count.
    """
    coverage = (
        metadata.drop_duplicates(["country", "country_code", "language_code"])
        .groupby(["country", "country_code"], as_index=False)
        .agg(languages=("language_code", "nunique"))
    )
    communities = metadata.groupby(["country", "country_code"], as_index=False).agg(
        communities=("community", "nunique")
    )
    return coverage.merge(
        communities,
        on=["country", "country_code"],
        validate="one_to_one",
    ).sort_values(
        "country",
    )


def geometry_rings(geometry: dict[str, Any]) -> list[list[list[float]]]:
    """Return exterior rings for Polygon and MultiPolygon GeoJSON geometry."""
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "Polygon":
        return [coordinates[0]] if coordinates else []
    if geometry_type == "MultiPolygon":
        return [polygon[0] for polygon in coordinates if polygon]
    return []


def load_country_features() -> list[dict[str, Any]]:
    """Load the vendored Natural Earth country geometries."""
    if not COUNTRY_GEOMETRY_PATH.exists():
        raise FileNotFoundError(f"Country geometry asset is missing: {COUNTRY_GEOMETRY_PATH}")
    payload = json.loads(COUNTRY_GEOMETRY_PATH.read_text(encoding="utf-8"))
    features = payload.get("features")
    if not isinstance(features, list):
        raise TypeError(f"Invalid country geometry asset: {COUNTRY_GEOMETRY_PATH}")
    return features
