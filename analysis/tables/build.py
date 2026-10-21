"""Build all tables, number macros, and machine-readable claims.

Tables go to ``data/derived/tables/``. Macros and claims go to
``data/derived/claims/``. Run from the repository root::

    uv run python -m analysis.tables.build
"""

from __future__ import annotations

import argparse

from analysis.tables import (
    coverage,
    english,
    human_error_types,
    judge_agreement,
    judge_by_community,
    judge_by_model,
    judge_prompt,
    macros,
    model_performance_summary,
    riddle_features,
)

TABLE_SCRIPTS = (
    human_error_types,
    model_performance_summary,
    coverage,
    judge_agreement,
    judge_prompt,
    judge_by_community,
    judge_by_model,
    riddle_features,
    english,
)


def main() -> None:
    """Write every generated table and the claims from the analysis outputs."""
    macros.main()
    for script in TABLE_SCRIPTS:
        script.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
