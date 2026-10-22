import csv
import json
from argparse import Namespace

import pandas as pd
import pytest

from submissions import __main__ as app
from submissions.validation import FIXED_TOPICS, validate


def _correction_file(path, *, action, field, old_value, new_value):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=app.LEDGER_FIELDS)
        writer.writeheader()
        writer.writerow(
            dict(
                riddle_id="x",
                action=action,
                field=field,
                old_value=old_value,
                new_value=new_value,
                reason="reviewed",
            )
        )


def test_stale_correction_fails_without_mutating_canonical(tmp_path, monkeypatch):
    canonical = tmp_path / "riddles.jsonl"
    row = {"riddle_id": "x", "community": "a", "riddle_original": "current", "is_benchmarked": True}
    canonical.write_text(json.dumps(row) + "\n", encoding="utf-8")
    ledger = tmp_path / "corrections.csv"
    _correction_file(
        ledger,
        action="replace",
        field="riddle_original",
        old_value="stale",
        new_value="replacement",
    )
    monkeypatch.setattr(app, "CANONICAL", canonical)
    before = canonical.read_bytes()
    with pytest.raises(ValueError, match="does not equal old_value"):
        app.corrections(Namespace(ledger=ledger, write=True, approved=True))
    assert canonical.read_bytes() == before


def test_ambiguous_append_fails_without_mutating_canonical(tmp_path, monkeypatch):
    canonical = tmp_path / "riddles.jsonl"
    row = {
        "riddle_id": "x",
        "community": "a",
        "riddle_note": "existing note",
        "is_benchmarked": True,
    }
    canonical.write_text(json.dumps(row) + "\n", encoding="utf-8")
    ledger = tmp_path / "corrections.csv"
    _correction_file(
        ledger, action="append", field="riddle_note", old_value="", new_value="new note"
    )
    monkeypatch.setattr(app, "CANONICAL", canonical)
    before = canonical.read_bytes()
    with pytest.raises(ValueError, match="append is ambiguous"):
        app.corrections(Namespace(ledger=ledger, write=True, approved=True))
    assert canonical.read_bytes() == before


def test_export_default_is_read_only_and_output_is_confined(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    monkeypatch.setattr(app, "ROOT", root)
    target = root / "data/derived/submissions/test-dry-run"
    assert app.export(Namespace(output=target, write=False)) == 0
    assert not target.exists()
    with pytest.raises(ValueError):
        app.export(Namespace(output=tmp_path / "outside", write=False))
    assert not root.exists()


def test_publish_dry_run_does_not_require_bundle_or_contact_service(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    monkeypatch.setattr(app, "ROOT", root)
    args = Namespace(bundle=root / "data/export", repo_id="owner/dataset", publish=False)
    assert app.publish(args) == 0
    assert not root.exists()


@pytest.mark.parametrize("column,sheet", [("Author", "Riddles"), ("Email", "Community")])
def test_submission_rejects_identity_columns_in_an_otherwise_valid_workbook(
    tmp_path, column, sheet
):
    path = tmp_path / "submission.xlsx"
    topics = [*sorted(FIXED_TOPICS), "Food", "Traditions"]
    rows = pd.DataFrame(
        [
            {
                "Number": number,
                "Topic": topics[(number - 1) // 10],
                "Riddle (Original Language)": f"Riddle {number}",
                "Reference Answer (Original)": f"Answer {number}",
            }
            for number in range(1, 101)
        ]
    )
    metadata = pd.DataFrame([{"community": "sample", "language_code": "eng"}])

    def save():
        with pd.ExcelWriter(path) as writer:
            metadata.to_excel(writer, sheet_name="Community", index=False)
            rows.to_excel(writer, sheet_name="Riddles", index=False)

    save()
    assert validate(path).errors == []
    (rows if sheet == "Riddles" else metadata)[column] = "private identity"
    save()
    assert validate(path).errors


def test_topic_rename_preserves_id_and_context_edits_invalidate_benchmark():
    prior = {
        "riddle_id": "example_food_081",
        "community": "example",
        "language": "eng",
        "topic": "Food",
        "is_wildcard": True,
        "riddle_original": "clue",
        "riddle_english": None,
        "riddle_hint": None,
        "riddle_note": None,
        "reference_answer_original": "answer",
        "reference_answer_english": None,
        "is_benchmarked": True,
    }
    table = pd.DataFrame(
        [
            {
                "Number": 81,
                "Topic": "Traditions",
                "Riddle (Original Language)": "clue",
                "Reference Answer (Original)": "answer",
                "Hints (Optional)": "new hint",
            }
        ]
    )
    row = app._new_rows({"community": "example", "language_code": "eng"}, table, [prior])[0]
    assert row["riddle_id"] == "example_food_081"
    assert row["topic"] == "Traditions"
    assert row["riddle_hint"] == "new hint"
    assert row["is_benchmarked"] is False


@pytest.mark.parametrize("action", ["replace", "append"])
def test_reviewed_context_correction_invalidates_scores_and_records_history_once(
    tmp_path, monkeypatch, action
):
    canonical = tmp_path / "riddles.jsonl"
    canonical.write_text(
        json.dumps({"riddle_id": "x", "riddle_note": "old note", "is_benchmarked": True}) + "\n"
    )
    pending = tmp_path / "pending.csv"
    history = tmp_path / "applied.csv"
    _correction_file(
        pending, action=action, field="riddle_note", old_value="old note", new_value="new note"
    )
    monkeypatch.setattr(app, "CANONICAL", canonical)
    monkeypatch.setattr(app, "LEDGER", history)
    args = Namespace(ledger=pending, write=True, approved=True)
    app.corrections(args)
    corrected = json.loads(canonical.read_text())
    assert corrected["riddle_note"] == ("new note" if action == "replace" else "old note\nnew note")
    assert corrected["is_benchmarked"] is False
    with history.open() as stream:
        first_history = list(csv.DictReader(stream))
    assert first_history[0]["reason"] == "reviewed"
    app.corrections(args)
    with history.open() as stream:
        assert list(csv.DictReader(stream)) == first_history
    before = canonical.read_bytes()
    with pytest.raises(ValueError, match="applied history"):
        app.corrections(Namespace(ledger=history, write=True, approved=True))
    assert canonical.read_bytes() == before


def test_export_writes_every_public_table_and_card(tmp_path):
    import shutil

    from submissions import dataset_card

    target = app.ROOT / "data/derived/submissions" / f"test-export-{tmp_path.name}"
    try:
        args = Namespace(
            output=target,
            write=True,
            snapshot=app.ROOT / "data/canonical/benchmark_results.parquet",
            baseline=app.ROOT / "data/canonical/evaluation_corpus.jsonl",
        )
        assert app.export(args) == 0
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        for name, path in dataset_card.TABLES.items():
            frame = pd.read_parquet(target / path)
            assert len(frame) == manifest["tables"][name] > 0
        answers = pd.read_parquet(target / dataset_card.TABLES["model_answers"])
        riddles = pd.read_parquet(target / dataset_card.TABLES["riddles"])
        assert set(answers["riddle_id"]) <= set(riddles["riddle_id"])
        card = (target / "README.md").read_text(encoding="utf-8")
        for name, path in dataset_card.TABLES.items():
            assert f"config_name: {name}" in card and path in card
        assert "must not be used to train" in card
        credits = pd.read_csv(app.ROOT / "docs/contributors.csv", dtype=str)
        assert "Riddle writers by community" in card
        assert all(name.strip() in card for name in credits["name"])
    finally:
        shutil.rmtree(target, ignore_errors=True)
