"""Score precedence, missing-evidence behavior, and join-key invariants."""

from __future__ import annotations

import polars as pl
import pytest

from analysis.scoring import assemble_scores, rule_decisions


def evidence():
    identifiers = ["rule", "human", "invalid", "agree", "disagree", "missing", "second"]
    responses = pl.DataFrame(
        {
            "riddle_id": identifiers,
            "model": ["test-model"] * len(identifiers),
            "answer_text": ["reference", *["candidate"] * (len(identifiers) - 1)],
            "community": ["test"] * len(identifiers),
            "topic": ["topic"] * len(identifiers),
        }
    )
    riddles = pl.DataFrame(
        {"riddle_id": identifiers, "reference_answer_original": ["reference"] * len(identifiers)}
    )
    annotations = pl.DataFrame(
        [
            {
                "riddle_id": rid,
                "answer_norm": answer,
                "annotation": annotation,
                "benchmark_status": status,
                "score": score,
                "error_type": None,
                "error_type_label": None,
                "annotator_id": "annotator_test",
                "note": None,
            }
            for rid, answer, annotation, status, score in (
                ("rule", "reference", "first", "valid", 0),
                ("human", "candidate", "first", "valid", 2),
                ("invalid", "candidate", "first", "invalid", 0),
                ("second", "candidate", "second", None, 3),
            )
        ],
        schema_overrides={
            "error_type": pl.String,
            "error_type_label": pl.String,
            "note": pl.String,
        },
    )
    judgments = {
        "rule": (0, 0),
        "human": (3, 3),
        "invalid": (2, 2),
        "agree": (3, 3),
        "disagree": (2, 3),
        "missing": (0, None),
    }
    verdicts = pl.DataFrame(
        [
            {
                "riddle_id": rid,
                "answer_norm": "reference" if rid == "rule" else "candidate",
                "judge": judge,
                "shots": "zero",
                "score": score,
            }
            for rid, scores in judgments.items()
            for judge, score in zip(("DeepSeek", "Gemini"), scores, strict=True)
            if score is not None
        ]
    )
    mapping = pl.DataFrame({"source_name": ["test-model"], "display_name": ["Test model"]})
    return responses, annotations, verdicts, rule_decisions(responses, riddles), mapping


def test_rules_then_valid_first_humans_then_agreeing_jury() -> None:
    scored = assemble_scores(*evidence())
    actual = {
        row["riddle_id"]: (row["score"], row["scorer"]) for row in scored.iter_rows(named=True)
    }
    assert actual == {
        "rule": (3, "rule"),
        "human": (2, "human"),
        "invalid": (2, "llm_judges"),
        "agree": (3, "llm_judges"),
        "disagree": (None, "none"),
        "missing": (None, "none"),
        "second": (None, "none"),
    }
    rows = {row["riddle_id"]: row for row in scored.iter_rows(named=True)}
    assert rows["human"]["annotator_id"] == "annotator_test"
    assert rows["invalid"]["annotator_id"] is None
    assert rows["rule"]["score_DeepSeek"] is None
    assert rows["human"]["score_Gemini"] is None


def test_invalidated_label_does_not_survive_historical_human_routing():
    inputs = list(evidence())
    inputs[1] = inputs[1].with_columns(
        benchmark_status=pl.when(pl.col("riddle_id") == "human")
        .then(pl.lit("invalid"))
        .otherwise(pl.col("benchmark_status")),
    )
    inputs[3] = inputs[3].with_columns(
        routing=pl.when(pl.col("riddle_id") == "human")
        .then(pl.lit("human_eval"))
        .otherwise(pl.col("routing")),
    )
    row = assemble_scores(*inputs).filter(pl.col("riddle_id") == "human").row(0, named=True)
    assert (row["score"], row["scorer"], row["annotator_id"]) == (3, "llm_judges", None)


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_duplicate_evidence_cannot_multiply_scores(index: int) -> None:
    inputs = list(evidence())
    inputs[index] = pl.concat([inputs[index], inputs[index].slice(0, 1)])
    with pytest.raises(ValueError):
        assemble_scores(*inputs)


def test_missing_routing_is_not_silently_dropped() -> None:
    inputs = list(evidence())
    inputs[3] = inputs[3].filter(pl.col("riddle_id") != "human")
    with pytest.raises(ValueError):
        assemble_scores(*inputs)


def test_answer_rules_do_not_turn_unresolved_answers_into_failures() -> None:
    answers = ["  REFERENCE  ", "", "<answer>", "I DON'T KNOW", "different answer"]
    responses = pl.DataFrame(
        {
            "riddle_id": [str(i) for i in range(len(answers))],
            "model": ["model"] * len(answers),
            "answer_text": answers,
        }
    )
    riddles = pl.DataFrame(
        {
            "riddle_id": responses["riddle_id"],
            "reference_answer_original": ["reference"] * len(answers),
        }
    )
    decisions = rule_decisions(responses, riddles)
    assert decisions["rule_score"].to_list() == [3, 0, 0, 1, None]
    assert decisions["routing"].to_list()[-1] == "pending_human_eval"


def test_changed_context_cannot_inherit_historical_labels():
    from analysis.scoring import historical_riddle_ids
    from submissions import evaluation_fingerprint

    row = {"riddle_id": "r1", "riddle_original": "clue", "is_benchmarked": True}
    lineage = pl.DataFrame({"riddle_id": ["r1"], "text_sha256": [evaluation_fingerprint(row)]})
    assert historical_riddle_ids(pl.DataFrame([row]), lineage)["riddle_id"].to_list() == ["r1"]
    row["riddle_hint"] = "a new clue changes what the model saw"
    with pytest.raises(ValueError):
        historical_riddle_ids(pl.DataFrame([row]), lineage)
    row["is_benchmarked"] = False
    assert historical_riddle_ids(pl.DataFrame([row]), lineage).is_empty()


@pytest.mark.parametrize("arguments", [["--model", "new"], ["--verdicts", "unrecorded.jsonl"]])
def test_new_jury_or_model_cannot_modify_implicit_historical_rebuild(arguments):
    from analysis.scoring import main

    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize("destination", ["data/canonical/benchmark_results.parquet", "historical"])
def test_new_run_cannot_overwrite_inputs_or_historical_scores(destination):
    from analysis import scoring

    output = (
        scoring.SCORED_RESPONSES_PATH
        if destination == "historical"
        else scoring.REPO_ROOT / destination
    )
    with pytest.raises(SystemExit) as error:
        scoring.main(["--responses", "new-results.jsonl", "--model", "new", "--out", str(output)])
    assert error.value.code == 2
