"""Generated figures and tables have standalone, registered output locations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from analysis.figures import _common
from analysis.figures import main as figures
from analysis.style import TARGET_ENV, figure_target
from analysis.tables import macros


@pytest.fixture
def figure_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    paper = tmp_path / "figures" / "paper"
    preprint = tmp_path / "figures" / "preprint"
    paper.mkdir(parents=True)
    preprint.mkdir(parents=True)
    monkeypatch.setattr(figures, "PAPER_FIGURES_DIR", paper)
    monkeypatch.setattr(figures, "PREPRINT_FIGURES_DIR", preprint)
    monkeypatch.setattr(figures, "FIGURES", {"fig_current": "module"})
    monkeypatch.setattr(figures, "PREPRINT_FIGURES", {"fig_extra": "module"})
    return paper, preprint


def _write_render(folder: Path, stem: str) -> None:
    for suffix in (".pdf", ".png"):
        (folder / f"{stem}{suffix}").write_bytes(b"rendered figure")


def test_complete_render_prunes_only_retired_generated_outputs(figure_tree) -> None:
    paper, preprint = figure_tree
    _write_render(paper, "fig_current")
    _write_render(preprint, "fig_current")
    _write_render(preprint, "fig_extra")
    for folder in figure_tree:
        _write_render(folder, "fig_retired")
        (folder / "reference.pdf").write_bytes(b"reference asset")
        (folder / "fig_notes.txt").write_text("notes", encoding="utf-8")

    figures._prune_outputs()

    for folder in figure_tree:
        assert not (folder / "fig_retired.pdf").exists()
        assert not (folder / "fig_retired.png").exists()
        assert (folder / "reference.pdf").read_bytes() == b"reference asset"
        assert (folder / "fig_notes.txt").read_text(encoding="utf-8") == "notes"


def test_incomplete_render_does_not_prune_outputs(figure_tree) -> None:
    paper, preprint = figure_tree
    _write_render(paper, "fig_current")
    _write_render(preprint, "fig_retired")

    with pytest.raises(FileNotFoundError, match="Missing generated figures"):
        figures._prune_outputs()

    assert (preprint / "fig_retired.pdf").exists()
    assert (preprint / "fig_retired.png").exists()


@pytest.mark.parametrize("target", ["paper", "preprint"])
def test_selected_target_does_not_touch_other_outputs(figure_tree, monkeypatch, target) -> None:
    paper, preprint = figure_tree
    selected, untouched = (paper, preprint) if target == "paper" else (preprint, paper)
    _write_render(selected, "fig_current")
    if target == "preprint":
        _write_render(selected, "fig_extra")
    _write_render(selected, "fig_retired")
    _write_render(untouched, "fig_retired")

    figures._prune_outputs(target)

    assert not (selected / "fig_retired.pdf").exists()
    assert not (selected / "fig_retired.png").exists()
    assert (untouched / "fig_retired.pdf").read_bytes() == b"rendered figure"
    assert (untouched / "fig_retired.png").read_bytes() == b"rendered figure"


@pytest.mark.parametrize("target", ["paper", "preprint"])
def test_save_writes_only_selected_target(tmp_path, monkeypatch, target) -> None:
    monkeypatch.setattr(_common, "PAPER_FIGURES_DIR", tmp_path / "paper")
    monkeypatch.setattr(_common, "PREPRINT_FIGURES_DIR", tmp_path / "preprint")
    monkeypatch.setattr(_common, "TARGET", target)
    fig = plt.figure(figsize=(1, 1))

    _common.save(fig, "fig_map")

    assert sorted(path.name for path in (tmp_path / target).iterdir()) == [
        "fig_map.pdf",
        "fig_map.png",
    ]
    other = "preprint" if target == "paper" else "paper"
    assert not (tmp_path / other).exists()
    assert not plt.fignum_exists(fig.number)


def test_paper_skips_preprint_only_stem_and_closes_figure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_common, "PAPER_FIGURES_DIR", tmp_path / "paper")
    monkeypatch.setattr(_common, "PREPRINT_FIGURES_DIR", tmp_path / "preprint")
    monkeypatch.setattr(_common, "TARGET", "paper")
    fig = plt.figure()

    _common.save(fig, "fig_map_wide")

    assert not list(tmp_path.iterdir())
    assert not plt.fignum_exists(fig.number)


def test_unknown_target_is_rejected_before_rendering(monkeypatch) -> None:
    monkeypatch.setenv(TARGET_ENV, "both")
    with pytest.raises(ValueError, match=TARGET_ENV):
        figure_target()


def test_unregistered_stem_cannot_write_a_render(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_common, "PAPER_FIGURES_DIR", tmp_path / "paper")
    monkeypatch.setattr(_common, "PREPRINT_FIGURES_DIR", tmp_path / "preprint")
    fig = plt.figure()
    try:
        with pytest.raises(ValueError, match="Unregistered figure"):
            _common.save(fig, "fig_unregistered")
        assert not (tmp_path / "preprint").exists()
    finally:
        plt.close(fig)


def test_macros_and_machine_readable_claims_share_values(tmp_path, monkeypatch) -> None:
    import json

    claims_dir = tmp_path / "data" / "derived" / "claims"
    commands = {"numriddles": "1,234", "bestmodelaccuracy": r"64.5\%", "language": "Thai"}
    monkeypatch.setattr(macros, "CLAIMS_DIR", claims_dir)
    monkeypatch.setattr(macros, "build_human_error_types", lambda: None)
    monkeypatch.setattr(macros, "build_commands", lambda _: commands)

    macros.main()

    claims = json.loads((claims_dir / "claims.json").read_text(encoding="utf-8"))
    counts = json.loads((claims_dir / "counts.json").read_text(encoding="utf-8"))
    assert counts == {"numriddles": 1234}
    assert claims["bestmodelaccuracy"] == {"value": 64.5, "unit": "percent", "latex": r"64.5\%"}
    assert claims["language"]["value"] == "Thai"
