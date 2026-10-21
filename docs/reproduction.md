# Reproduction and evaluation

Run every command from the repository root. No component has a separate Python environment or lockfile.

## Fresh checkout: reproduce recorded evidence

```bash
uv sync --frozen
uv run --frozen pre-commit install --install-hooks
uv run --frozen python -m scripts.release_support
uv run --frozen python -m analysis.run_all
uv run --frozen python -m analysis.export_dashboard
uv run --frozen python -m dashboard.compute
uv run --frozen python -m dashboard.build_static
```

Open `dashboard/index.html`. Its runtime consumes the exported JSON embedded by the builder, not analysis code or raw Parquet files. Plotly loads from its documented CDN; an offline browser needs that dependency available.

`analysis.run_all` performs these dependency-ordered steps:

1. Assemble scores from recorded responses, routing decisions, anonymous human annotations, and recorded judge verdicts.
2. Rebuild judge-validation, human-evaluation, model-result, corpus, translation, and writing-survey aggregate summaries.
3. Rebuild mixed-model matrices under `data/derived/build/`, and replay the released R fit-summary CSVs only after verifying input-matrix, R-script, and output hashes against their provenance records.
4. Write the tables under `data/derived/tables/`, plus `macros.tex`, `claims.json`, and `counts.json` under `data/derived/claims/`.
5. Render the registered paper and preprint figures with the shared green, blue, and coral theme.

All intermediate data remains under `data/derived/`. This workflow doesn't call an API to regenerate recorded observations. The only survey inputs are the approved aggregate tables.

## Refit all statistical models

Install R with the `lme4` and `data.table` packages, then run:

```bash
uv run --frozen python -m analysis.run_all   # first run builds the inputs the fits read
uv run --frozen python -m analysis.difficulty_decomposition.prepare --fit          # variance and tier models
uv run --frozen python -m analysis.difficulty_decomposition.riddle_features --fit  # riddle-feature models
uv run --frozen python -m analysis.run_all   # replay the fresh fits into tables and figures
```

`analysis.run_all` never refits, so a normal reproduction stays fast. The fit commands are independent and slow (about 30 minutes together on a recent laptop), so run only the one you need. Together they fit the difficulty-decomposition and riddle-feature models. Input matrices, fit summaries, provenance records, and serialized R model objects remain under `data/derived/`. Each successful fit records the actual matrix, R script, and output hashes; default replay rejects stale or missing records instead of trusting old estimates or refreshing them silently. R/package versions and numerical solvers can affect fitted estimates; preserve environment details with a reviewed snapshot.

## Evaluate a new model

Model adapters and configurations live under `benchmark/`; the installed command remains `riddles-framework`. Start with its paid-API-free path:

```bash
uv run --frozen riddles-framework --config benchmark/configs/dummy.yaml --limit 5
```

The command prints a run ID; its files live at `data/runs/<run_id>/`. To make a real run, use or adapt an existing configuration, set only the required provider credentials, and record explicit provider-supported reasoning/search parameters. Do not label a null/unspecified request as reasoning disabled. Local inference dependencies are optional: `uv sync --frozen --extra local`.

Set `RUN_DIR=data/runs/<run_id>` using the printed ID, and set `MODEL_KEY` to the configuration key you are evaluating. Score its actual output:

```bash
uv run --frozen python -m analysis.scoring \
  --responses "$RUN_DIR/results.jsonl" --model "$MODEL_KEY" \
  --out data/runs/new-model/scored.parquet
```

Rules are applied to new answers, followed by reusable valid human labels and the recorded jury evidence for matching `(riddle_id, answer_norm)` pairs. Unmatched or disagreeing jury answers remain null and are exported to `data/runs/new-model/scored_unresolved.parquet`. Historical labels are not reused for changed/unbenchmarked riddles.

Released human labels are in `data/canonical/human_eval/annotations.parquet`; `human_eval/` contains the validator for additional workbooks, not the released results. See the [benchmark guide](../benchmark/README.md) for configuration and output details and the [human-evaluation guide](../human_eval/README.md) for annotation checks. Keep all local scoring output under `data/runs/`, which is excluded from release archives.

### Complete unresolved answers with the jury

The following commands call paid APIs. Do not run them merely to reproduce the released results. Current provider aliases may route to different versions than the historical observations.

```bash
uv run --frozen riddles-judge \
  data/runs/new-model/scored_unresolved.parquet \
  --model gemini-3.5-flash --out data/runs/new-model/gemini.jsonl
uv run --frozen riddles-judge \
  data/runs/new-model/scored_unresolved.parquet \
  --model deepseek-v4.1-flash --out data/runs/new-model/deepseek.jsonl
uv run --frozen python -m analysis.scoring \
  --responses "$RUN_DIR/results.jsonl" --model "$MODEL_KEY" \
  --verdicts data/runs/new-model/gemini.jsonl data/runs/new-model/deepseek.jsonl \
  --out data/runs/new-model/scored.parquet
```

The scorer reads validated final verdicts and their immutable manifest sidecars directly. Explicit new verdicts supplement or override matching historical jury keys; they do not discard unrelated existing evidence. Both zero-shot judges must agree. The runner freezes input context, records request/prompt/rubric hashes, and rejects incompatible resumes. See the [benchmark guide](../benchmark/README.md) for fresh judge runs and the [recorded judge analysis guide](../analysis/llm_judge/README.md) for offline studies.

New runs are not silently promoted into the canonical benchmark. Review their configuration provenance, responses, labels/verdicts, and corpus lineage before creating an approved new snapshot and refitting/rebuilding analyses. Keep historical configurations distinct when provider aliases resolve differently.

## Trace a reported result

- Counts and number macros: `data/derived/claims/{counts.json,claims.json,macros.tex}`.
- Model rankings and subgroup aggregates: `data/derived/model_results/`.
- Judge validation: `data/derived/llm_judge/`.
- Mixed-model inputs/fits: `data/derived/{build,difficulty_decomposition}/`.
- Corpus features: `data/derived/riddle_analysis/`.
- Recomputed survey joins: `data/derived/riddle_writing_analysis/`.
- Figure/table generator inventory: `analysis/figure_registry.py`.
- Requested model IDs and actual supported reasoning/search evidence: `data/canonical/model_runs.csv`.
- Approved input schemas, keys, counts, and hashes: `data/manifests/canonical.json`.

Claims are generated from the same summaries as tables and charts. A custom dataset license with use restrictions must not be presented as unrestricted CC-BY data.

## Verify a self-contained release

After regeneration and dashboard export:

```bash
uv run --frozen python -m scripts.build_release
uv run --frozen python -m scripts.verify_release dist/culturalriddles-repro.zip
# With R dependencies installed, also refit inside the isolated extraction:
uv run --frozen python -m scripts.verify_release dist/culturalriddles-repro.zip --fit
```

The archive contains the root project and configuration files, code, approved data, derived artifacts, and registered figures. Shipped data is limited to `data/{canonical,reference,derived,manifests}/`, `data/{corrections.csv,README.md,LICENSE}`, and the anonymous blank `data/submissions/submission_template.xlsx`. Extraction independently enforces this data allowlist, even for a self-consistent file inventory. The archive also rejects missing, extra, changed, traversal, and symlink members.

The archive leaves out files that the analysis rebuilds and nothing else reads: the model input matrices under `data/derived/build/` and the serialized R model objects (`*.rds`). Default replay still verifies the regenerated matrices against the hashes in the fit provenance records, and `--fit` writes fresh `.rds` objects.

The verifier extracts into a new temporary directory, validates approved inputs, snapshots expected analytical outputs outside the extracted checkout, deletes generated intermediates and figures, installs with `uv sync --frozen`, reruns the analysis/export/standalone dashboard, and compares output inventories. Default replay preserves only the explicit R fit-summary CSVs and their provenance records; every input matrix must be regenerated and match those records. `--fit` preserves none.

CSV/Parquet values and JSON numbers use absolute tolerance `1e-8` for cached replay and `1e-6` for full fits, with no relative tolerance. Full-fit provenance must retain identical input/script signatures and bind the newly fitted CSV bytes; the CSV values are compared numerically rather than requiring solver-dependent output hashes to match the older fit. Generated tables/macros must match exactly. Figures must be regenerated, valid, and nonblank; byte identity is not required because PDF metadata and rendering backends vary. Visual review remains necessary for layout and semantics. Reproduction clears inherited Python/project environment overrides and uses the extracted checkout and its new `.venv`; this is checkout isolation, not an operating-system sandbox.

## Maintainer checks

```bash
uv run --frozen pytest
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pre-commit run --all-files
```

The canonical manifest is an approval boundary, not an auto-updating lock on incoming data. Review anonymity and changed evaluation inputs before explicitly freezing it. The structural checker can't verify free-text notes, so review them for personal information.
