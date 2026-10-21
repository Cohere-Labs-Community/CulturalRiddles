# Working in Cultural Riddles

## Start with the user's task

Run all commands from the repository root. Use `uv sync --frozen` and the shared Python 3.12 environment; do not create component environments or lockfiles.

| Task | Read first | Offline first check |
| --- | --- | --- |
| Submit or review riddles | `CONTRIBUTING.md`, `submissions/README.md` | `uv run --frozen python -m submissions check PATH_TO_WORKBOOK.xlsx` |
| Reproduce paper outputs | `analysis/README.md`, `docs/reproduction.md` | `uv run --frozen python -m scripts.release_support`, then `uv run --frozen python -m analysis.run_all` |
| Benchmark a model | `benchmark/README.md`, `human_eval/README.md` | `uv run --frozen riddles-framework --config benchmark/configs/dummy.yaml --limit 5` |

Replace `PATH_TO_WORKBOOK.xlsx` with an actual completed anonymous workbook. The blank template is not a valid completed submission.

## Data and approval boundaries

- Keep code in the root packages, approved inputs in `data/canonical/` and `data/reference/`, generated analytical artifacts in `data/derived/`, and figures in `figures/`.
- Treat `data/canonical/benchmark_results.parquet` as frozen comparison evidence, not a generated output. Downstream analyses read rebuilt scores, not this reference.
- Do not edit canonical inputs, re-freeze `data/manifests/canonical.json`, or apply corrections merely to make a check pass. Canonical changes require content and anonymity review.
- Release only files that `scripts/build_release.py` allowlists. Intake workbooks, pending correction ledgers, `.env` files, and `data/runs/` stay local: `data/submissions/` intake and `data/runs/` are ignored. Never stage them broadly; a contributor explicitly stages only the reviewed anonymous workbook described in `CONTRIBUTING.md`.
- `submissions check` and import previews must not change canonical files. Import and correction writes require both `--approved` and `--write`. Publishing is a distinct explicitly approved network operation.
- New or changed riddle context cannot inherit historical benchmark labels solely because its ID matches. Preserve evaluated-content fingerprints and stable workbook numbers.
- Preserve the separate dataset terms in `data/LICENSE`; the code's Apache-2.0 license does not replace them.

## Benchmarking and human evaluation

- Physical paths are under `benchmark/`, not `framework/`. The Python package is `riddles_framework` and the installed command is `riddles-framework`.
- `.env.template` contains blank credential values and is tracked and included in releases. Actual `.env` files remain ignored and excluded. Judges use direct Gemini/DeepSeek APIs through LiteLLM and load root `.env` without overriding the shell; solvers use their YAML-selected provider and `uv run --env-file .env` when credentials are not exported.
- Write model responses, configuration/provenance, new scores, and judge traces beneath `data/runs/`. Do not promote local runs into the released benchmark automatically.
- Released human labels live in `data/canonical/human_eval/annotations.parquet`. The root `human_eval/` package validates additional annotation workbooks; it does not contain the released results.
- Answer normalization is `answer.strip().lower()`. Reuse labels only for matching `(riddle_id, answer_norm)` and unchanged evaluated context.
- Fresh judge execution, prompt construction, and study sampling live in `benchmark/src/riddles_framework/judging/`. Use the public `riddles-judge` command for inference rather than internal module paths; `benchmark/README.md` links the implementation. The `analysis/llm_judge/` modules summarize recorded verdicts; keep paid execution out of the offline pipeline.
- Scoring precedence is rules, valid first human label, then agreement between two zero-shot judges. Missing or disagreeing verdicts remain null, not incorrect. Accuracy excludes null scores from its denominator.
- Historical routing and model configuration identity are recorded evidence. Preserve quantization and supported reasoning/search distinctions. Unspecified provider parameters do not mean reasoning was disabled.
- Do not make paid provider/judge calls for a smoke check or historical reproduction. Dummy adapters and recorded evidence provide the offline path. Document the explicit paid boundary.

## Analysis responsibilities

- `analysis.run_all` is the dependency-ordered offline regeneration entrypoint. Keep analytical calculations separate from figure and table formatting; reuse existing loaders and aggregate definitions.
- `analysis/figure_registry.py` is the output inventory. Update it when adding or removing registered figures, tables, or claims, together with release verification and docs.
- Use `analysis/style.py` for both paper and preprint colors and typography. Keep the shared green, blue, and coral condition mapping; do not restore a second target-specific palette.
- Generate number macros and machine-readable claims from the same aggregate evidence as charts and tables.
- Don't write counts in code or documentation: no figure, table, model, file, community, or riddle counts, as digits or as words, in filenames, function or variable names, docstrings, comments, or prose. Counts go stale and don't get updated. Derive them from the data or from the registry, or describe the thing without a number.
- Default R replay validates matrix, R-script, and summary hashes. Do not refresh provenance to bypass changed inputs. `analysis.run_all` never refits. An intentional refit is a separate, slow step (R, `lme4`, `data.table`): run `analysis.difficulty_decomposition.prepare --fit` and `analysis.difficulty_decomposition.riddle_features --fit`, then `analysis.run_all`.
- Inspect generated PNGs for labels, clipping, and encoding; passing Python tests is not visual proof.
- The dashboard consumes exported aggregate JSON. Don't make its renderer import raw analytical pipelines.

## Verify changes

Run the affected real workflow, not only unit tests:

- Intake changes: generate a temporary completed anonymous workbook, check and preview import, and reject malformed inputs. Confirm no canonical write. Keep regression tests for meaningful validation boundaries.
- Analysis changes: run `uv run --frozen python -m analysis.run_all`; inspect claims/tables and representative paper and preprint PNGs. Refit only when intentionally changing model inputs or fitting code.
- Benchmark changes: execute the documented dummy run, score its actual `results.jsonl`, and inspect scored and unresolved outputs. Exercise matching frozen human evidence if scoring reuse changes.
- Export/dashboard changes: run `analysis.export_dashboard`, `dashboard.compute`, and `dashboard.build_static` as modules; inspect the generated surface when changing rendering.

Then run the root checks:

```bash
uv run --frozen pytest
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

For release or dependency/layout changes, also run:

```bash
uv run --frozen python -m scripts.build_release
uv run --frozen python -m scripts.verify_release
```

The archive verifier installs into a fresh extraction and regenerates analytical outputs. Keep `benchmark/`, root instructions, schemas, required package data, and recorded evidence in its allowlist; keep intake workbooks and local run artifacts out.

Update the relevant workflow guide when behavior or commands change. Remove obsolete callers and documentation rather than leaving aliases for old physical paths. Report which commands actually ran, and distinguish offline retained-fit replay from fresh R fits and paid inference.
