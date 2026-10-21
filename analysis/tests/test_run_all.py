"""Stage selection keeps one explicit dependency-ordered publication workflow."""

from __future__ import annotations

import pytest

from analysis import run_all


def test_unknown_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown analysis stage"):
        run_all.main(stage="unknown")
