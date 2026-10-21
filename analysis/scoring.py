"""Rebuild released scores or score a new run using public evidence.

Historical replay preserves recorded routing decisions. New runs apply rules,
then valid first human labels, then matching recorded two-judge verdicts. Answers
without two agreeing verdicts remain unscored and are exported for review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from analysis.constants import (
    HUMAN_ANNOTATIONS_PATH,
    JUDGE_DATA_DIR,
    MODEL_NAME_MAPPING_PATH,
    MODEL_RESPONSES_PATH,
    REPO_ROOT,
    RIDDLES_PATH,
    SCORED_RESPONSES_PATH,
)
from submissions import evaluation_fingerprint

KEY = ["riddle_id", "answer_norm"]
RESPONSE_KEY = ["riddle_id", "model"]
OUTPUT_COLUMNS = [
    "riddle_id",
    "community",
    "topic",
    "model",
    "model_display",
    "model_answer",
    "answer_norm",
    "score",
    "scorer",
    "routing",
    "error_type",
    "error_type_label",
    "annotator_id",
    "validator_note",
    "score_DeepSeek",
    "score_Gemini",
]


def _unique(frame: pl.DataFrame, keys: list[str], name: str) -> None:
    if frame.select(keys).is_duplicated().any():
        raise ValueError(f"Duplicate {name} keys: {keys}")


def historical_riddle_ids(riddles: pl.DataFrame, lineage: pl.DataFrame) -> pl.DataFrame:
    """Reuse observations only when the evaluated content and status still match."""
    _unique(lineage, ["riddle_id"], "evaluation lineage")
    recorded = dict(lineage.select("riddle_id", "text_sha256").iter_rows())
    selected, stale = [], []
    for row in riddles.iter_rows(named=True):
        if not row["is_benchmarked"]:
            continue
        if recorded.get(row["riddle_id"]) != evaluation_fingerprint(row):
            stale.append(row["riddle_id"])
        else:
            selected.append(row["riddle_id"])
    if stale:
        raise ValueError(
            f"Benchmarked content differs from recorded evaluation inputs; mark changed rows unbenchmarked: {stale}"
        )
    return pl.DataFrame({"riddle_id": selected}, schema={"riddle_id": pl.String})


def rule_decisions(responses: pl.DataFrame, riddles: pl.DataFrame) -> pl.DataFrame:
    """Apply explicit answer rules without consulting any human or jury scores."""
    rows = responses.join(
        riddles.select("riddle_id", "reference_answer_original"),
        on="riddle_id",
        how="left",
        validate="m:1",
    )
    answer = pl.col("answer_text").fill_null("").str.strip_chars().str.to_lowercase()
    reference = (
        pl.col("reference_answer_original").fill_null("").str.strip_chars().str.to_lowercase()
    )
    routing = (
        pl.when(answer == "")
        .then(pl.lit("auto_empty"))
        .when(answer.str.contains(r"^<[^>]+>$"))
        .then(pl.lit("auto_placeholder"))
        .when(answer.is_in(["i don't know", "i don’t know", "i do not know", "idk"]))
        .then(pl.lit("auto_idk"))
        .when((reference != "") & (answer == reference))
        .then(pl.lit("auto_exact_match"))
        .otherwise(pl.lit("pending_human_eval"))
    )
    return rows.select(*RESPONSE_KEY, routing=routing).with_columns(
        rule_score=pl.when(pl.col("routing") == "auto_exact_match")
        .then(3)
        .when(pl.col("routing") == "auto_idk")
        .then(1)
        .when(pl.col("routing").str.starts_with("auto_"))
        .then(0)
        .otherwise(None)
        .cast(pl.Int8),
        rule_error_type=pl.when(pl.col("routing") == "auto_empty")
        .then(pl.lit("0.6"))
        .when(pl.col("routing") == "auto_placeholder")
        .then(pl.lit("0.5"))
        .otherwise(None),
        rule_error_type_label=pl.when(pl.col("routing") == "auto_empty")
        .then(pl.lit("0.6 - Empty response (auto)"))
        .when(pl.col("routing") == "auto_placeholder")
        .then(pl.lit("0.5 - Placeholder answer (auto)"))
        .otherwise(None),
    )


def assemble_scores(
    responses: pl.DataFrame,
    annotations: pl.DataFrame,
    verdicts: pl.DataFrame,
    decisions: pl.DataFrame,
    mapping: pl.DataFrame,
) -> pl.DataFrame:
    """Assemble scores; rules outrank humans, and humans outrank the jury.

    Missing and disagreeing judge verdicts never imply an incorrect answer.
    Historical invalid-human routing remains visible after jury completion.
    """
    _unique(responses, RESPONSE_KEY, "response")
    _unique(decisions, RESPONSE_KEY, "routing decision")
    first = annotations.filter(pl.col("annotation") == "first")
    _unique(first, KEY, "first annotation")
    judged = verdicts.filter(pl.col("shots") == "zero")
    _unique(judged, [*KEY, "judge"], "judge verdict")
    if responses.join(decisions, on=RESPONSE_KEY, how="anti").height:
        raise ValueError("A response has no routing decision")
    first = first.select(
        *KEY,
        "benchmark_status",
        h_score=pl.col("score"),
        h_type=pl.col("error_type"),
        h_label=pl.col("error_type_label"),
        h_annotator=pl.col("annotator_id"),
        h_note=pl.col("note"),
    )
    rows = responses.with_columns(
        answer_norm=pl.col("answer_text").fill_null("").str.strip_chars().str.to_lowercase()
    ).join(decisions, on=RESPONSE_KEY, how="left", validate="1:1")
    rows = rows.join(first, on=KEY, how="left", validate="m:1")
    for judge in ("DeepSeek", "Gemini"):
        rows = rows.join(
            judged.filter(pl.col("judge") == judge).select(
                *KEY, **{f"score_{judge}": pl.col("score")}
            ),
            on=KEY,
            how="left",
            validate="m:1",
        )
    rule = pl.col("routing").str.starts_with("auto_")
    human = ~rule & (pl.col("benchmark_status") == "valid").fill_null(False)
    agree = (pl.col("score_DeepSeek") == pl.col("score_Gemini")).fill_null(False)
    rows = rows.with_columns(
        scorer=pl.when(rule)
        .then(pl.lit("rule"))
        .when(human)
        .then(pl.lit("human"))
        .when(agree)
        .then(pl.lit("llm_judges"))
        .otherwise(pl.lit("none")),
        routing=pl.when(human).then(pl.lit("human_eval")).otherwise(pl.col("routing")),
    )
    rows = rows.with_columns(
        score=pl.when(pl.col("scorer") == "rule")
        .then(pl.col("rule_score"))
        .when(pl.col("scorer") == "human")
        .then(pl.col("h_score"))
        .when(pl.col("scorer") == "llm_judges")
        .then(pl.col("score_Gemini"))
        .otherwise(None)
        .cast(pl.Int8),
        error_type=pl.when(pl.col("scorer") == "human")
        .then(pl.col("h_type"))
        .when(pl.col("scorer") == "rule")
        .then(pl.col("rule_error_type"))
        .otherwise(None),
        error_type_label=pl.when(pl.col("scorer") == "human")
        .then(pl.col("h_label"))
        .when(pl.col("scorer") == "rule")
        .then(pl.col("rule_error_type_label"))
        .otherwise(None),
        annotator_id=pl.when(pl.col("scorer") == "human")
        .then(pl.col("h_annotator"))
        .otherwise(None),
        validator_note=pl.when(pl.col("scorer") == "human").then(pl.col("h_note")).otherwise(None),
        score_DeepSeek=pl.when(~rule & ~human)
        .then(pl.col("score_DeepSeek"))
        .otherwise(None)
        .cast(pl.Int8),
        score_Gemini=pl.when(~rule & ~human)
        .then(pl.col("score_Gemini"))
        .otherwise(None)
        .cast(pl.Int8),
    )
    names = mapping.select(model=pl.col("source_name"), canonical_display=pl.col("display_name"))
    rows = rows.drop("model_display", strict=False).join(
        names, on="model", how="left", validate="m:1"
    )
    return (
        rows.with_columns(
            model_display=pl.col("canonical_display").fill_null(pl.col("model")),
            model_answer=pl.col("answer_text"),
        )
        .select(OUTPUT_COLUMNS)
        .sort("model", "riddle_id")
    )


def load_run(path: Path, model: str | None = None) -> pl.DataFrame:
    """Read canonical response Parquet or a framework results.jsonl export."""
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    if not model:
        raise ValueError("--model is required for a framework results.jsonl export")
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        parsed = row.get("parsed_json") or {}
        answer = parsed.get("riddle_answer")
        if answer is None:
            answer = row.get("raw_response") or ""
        records.append({"riddle_id": row["riddle_id"], "model": model, "answer_text": answer})
    return pl.DataFrame(
        records, schema={"riddle_id": pl.String, "model": pl.String, "answer_text": pl.String}
    )


def load_verdicts(paths: list[Path]) -> pl.DataFrame:
    """Read public verdict Parquet or validated finals from actual judge runs."""
    frames = []
    columns = [*KEY, "judge", "shots", "score"]
    for path in paths:
        if path.suffix == ".parquet":
            frames.append(pl.read_parquet(path).select(columns))
            continue
        from riddles_framework.judging.llm_judge_run import finals, load_manifest, sidecar

        manifest = load_manifest(sidecar(path, "manifest.json"))
        config = manifest["config"]
        judge = {"gemini": "Gemini", "deepseek": "DeepSeek"}[config["provider"]["provider"]]
        identities = {tuple(value.split("\u241f")) for value in config["expected_identities"]}
        records = finals(path, manifest["manifest_id"], identities).values()
        rows = [
            {
                "riddle_id": row["riddle_id"],
                "answer_norm": row["candidate"].strip().lower(),
                "judge": judge,
                "shots": "zero" if row["few_shot_count"] == 0 else "few10",
                "score": row["verdict"]["score"],
            }
            for row in records
        ]
        frames.append(
            pl.DataFrame(
                rows,
                schema={
                    "riddle_id": pl.String,
                    "answer_norm": pl.String,
                    "judge": pl.String,
                    "shots": pl.String,
                    "score": pl.Int8,
                },
            ).select(columns)
        )
    return pl.concat(frames, how="vertical_relaxed")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses", type=Path, help="New response Parquet or framework results.jsonl"
    )
    parser.add_argument("--model", help="Configuration key for a new framework run")
    parser.add_argument(
        "--out", type=Path, help="Scored Parquet; new runs require an explicit path under data/"
    )
    parser.add_argument(
        "--verdicts",
        type=Path,
        nargs="+",
        help="Verdict Parquet or the two judge-run JSONL finals; defaults to recorded setting C",
    )
    args = parser.parse_args(argv)
    if not args.responses and (args.verdicts or args.model):
        parser.error(
            "--verdicts and --model require --responses for an explicitly selected new run"
        )
    if args.responses and args.out is None:
        parser.error("--responses requires --out to avoid overwriting the historical rebuild")
    output = (args.out or SCORED_RESPONSES_PATH).resolve()
    if not any(
        output.is_relative_to((REPO_ROOT / "data" / directory).resolve())
        for directory in ("derived", "runs")
    ):
        parser.error("--out must be under data/derived/ or data/runs/, never canonical inputs")
    if args.responses and output == SCORED_RESPONSES_PATH.resolve():
        parser.error("New runs cannot overwrite the historical scored-response rebuild")
    riddles = pl.read_ndjson(RIDDLES_PATH, infer_schema_length=None)
    benchmarked = historical_riddle_ids(
        riddles,
        pl.read_ndjson(REPO_ROOT / "data/canonical/evaluation_corpus.jsonl"),
    )
    responses = (
        load_run(args.responses, args.model)
        if args.responses
        else pl.read_parquet(MODEL_RESPONSES_PATH)
    )
    if args.responses:
        responses = responses.drop("community", "topic", strict=False).join(
            riddles.select("riddle_id", "community", "topic"),
            on="riddle_id",
            how="left",
            validate="m:1",
        )
        if responses["community"].null_count():
            raise ValueError("New run contains riddle IDs outside the canonical corpus")
        decisions = rule_decisions(responses, riddles)
    else:
        decisions = pl.read_parquet(REPO_ROOT / "data/canonical/scoring_decisions.parquet")
        # The generation export and the actual adjudicated candidate differ on
        # historical parse/curation cases. Preserve both observations explicitly.
        responses = responses.join(benchmarked, on="riddle_id", how="semi").with_columns(
            answer_text=pl.col("evaluation_answer_text"),
        )
        decisions = decisions.join(benchmarked, on="riddle_id", how="semi")
    annotations = pl.read_parquet(HUMAN_ANNOTATIONS_PATH).join(
        benchmarked,
        on="riddle_id",
        how="semi",
    )
    verdicts = load_verdicts([JUDGE_DATA_DIR / "setting_c/verdicts.parquet"]).join(
        benchmarked,
        on="riddle_id",
        how="semi",
    )
    if args.verdicts:
        verdict_keys = [*KEY, "judge", "shots"]
        supplied = load_verdicts(args.verdicts)
        _unique(verdicts, verdict_keys, "recorded judge verdict")
        _unique(supplied, verdict_keys, "supplied judge verdict")
        verdicts = pl.concat([verdicts, supplied], how="vertical_relaxed").unique(
            verdict_keys,
            keep="last",
            maintain_order=True,
        )
    scored = assemble_scores(
        responses,
        annotations,
        verdicts,
        decisions,
        pl.read_csv(MODEL_NAME_MAPPING_PATH, infer_schema_length=0),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    scored.write_parquet(output, compression="zstd")
    if args.responses:
        unresolved = scored.filter(pl.col("scorer") == "none").unique(KEY)
        unresolved.write_parquet(
            output.with_name(output.stem + "_unresolved.parquet"), compression="zstd"
        )
    print(
        f"Scored {scored.height:,} responses: {dict(scored['scorer'].value_counts().iter_rows())}"
    )


if __name__ == "__main__":
    main()
