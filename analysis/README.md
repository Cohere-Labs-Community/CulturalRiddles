# Reproduce the publication analysis

Run commands from the repository root with its `pyproject.toml` and `uv.lock`.
The analysis rebuilds scores, summaries, LaTeX tables, aggregate number
macros, machine-readable claims and counts, and the paper and preprint figures.
Both figure sets use the green, blue, and coral theme in
[`style.py`](style.py). The analysis retains established canvas sizes and layouts.

## Rebuild from canonical evidence

```bash
uv sync --frozen
uv run --frozen python -m analysis.run_all
uv run --frozen python -m analysis.export_dashboard
uv run --frozen python -m dashboard.compute
uv run --frozen python -m dashboard.build_static
```

The default analysis reads recorded verdicts and retained R fit summaries.
It makes no provider calls and doesn't require API credentials. It reads only
the approved files under `data/`. Missing inputs are errors, not skipped steps.
For more information about the figures, see the [figure guide](figures/README.md).
Statistical replay doesn't require R.
The dashboard is a separate aggregate-only export, then a standard-library
consumer and HTML builder. None of these commands edits canonical evidence.

The `analysis.difficulty_decomposition` fits require R. `run_all` doesn't:
it replays the R models from retained summaries and never refits them,
because the fits are slow (about 30 minutes together) and optional. To refit,
run `run_all` once so the fits have their inputs, install R and the `lme4` and
`data.table` packages, and then run the independent fit commands. Finish
with `run_all` to replay the fresh fits into tables and figures:

```bash
uv run --frozen python -m analysis.difficulty_decomposition.prepare --fit          # variance and tier
uv run --frozen python -m analysis.difficulty_decomposition.riddle_features --fit  # riddle features
uv run --frozen python -m analysis.run_all
```

The released fits used R 4.3.1, `lme4` 1.1-34, and `data.table` 1.14.8. Fits
write `variance.rds`, `tier.rds`, `accuracy_tier.rds`, and
`accuracy_features.rds`, their summary CSVs, and provenance JSON files to
`data/derived/difficulty_decomposition/`. Only a successful fit records
provenance. Replay rebuilds the actual input matrices and rejects absent or
stale hashes of the matrices, R scripts, or fitted summaries. Never regenerate
provenance only to accept an old cache.

## Inputs and outputs

[`constants.py`](constants.py) owns repository paths and topic vocabulary. The
following table lists the inputs and outputs:

| Location | Role |
| --- | --- |
| `data/canonical/riddles.jsonl`, `corpus_metadata.csv` | Corpus text and community metadata |
| `data/canonical/evaluation_corpus.jsonl` | Frozen evaluation snapshot identifying historical benchmark riddles |
| `data/canonical/model_responses.parquet`, `model_mapping.csv` | Recorded answers and model vocabulary |
| `data/canonical/scoring_decisions.parquet`, `human_eval/` | Recorded routing, anonymous annotations, and public rubric |
| `data/canonical/judge/`, `english_condition/` | Frozen study pairs, answers, and verdicts |
| `data/canonical/riddle_writing_analysis/` | Approved aggregate survey tables |
| `data/canonical/benchmark_results.parquet` | Frozen scored snapshot; never overwritten by analysis |
| `data/corrections.csv`, `data/reference/` | Correction ledger and reference metadata/map geometry |
| `data/derived/scoring/responses_scored_consolidated.parquet` | Recomputed historical scores |
| `data/derived/<topic>/` | Regenerated topic summaries and retained R summaries |
| `data/derived/build/difficulty/` | Regenerated row-level R input matrices |
| `data/derived/tables/` | Generated `.tex` tables |
| `data/derived/claims/` | `macros.tex`, `claims.json`, and integer `counts.json` |
| `data/derived/dashboard/dashboard_data.json` | Aggregate-only dashboard contract |
| `figures/paper/`, `figures/preprint/` | Vector PDFs and review PNGs |
| `dashboard/index.html` | Standalone dashboard generated from aggregate JSON |

## Dependency order and module responsibilities

`run_all.py` is the only cross-topic orchestrator. It calls producers directly,
in the order of the following table. Topic-level `main.py` forwarding wrappers no longer exist.
For a focused rebuild, use `--stage STAGE` after its upstream inputs exist.
Replace `STAGE` with a stage name from the table. The stage does not silently
run dependencies.

| Stage | Producers | Main consumers |
| --- | --- | --- |
| `scoring` | `scoring.py` applies rules, anonymous annotations, and recorded judge consensus | All accuracy analyses and benchmark rescoring |
| `judges` | `llm_judge/`: pending-label audit, rater agreement, judge agreement/failures/sample audit, English condition, native counterfactual | Judge/English figures and tables, macros, dashboard |
| `corpus` | `riddle_analysis/`: surface forms and wildcard themes | Surface-form figure/macros; topic prose and audit |
| `evaluation` | `eval_analysis/`: group and country accuracy | Resource figure/macros, country/subregion dashboard views |
| `models` | `model_results/`: human summary, judge-fill, coverage, regional specialization, search flips | Performance figure/table, model-comparison prose, dashboard |
| `survey` | `riddle_writing_analysis/survey_summaries.py` | Survey macros and dashboard topic comparison |
| `difficulty` | `difficulty_decomposition/prepare.py`, `riddle_features.py`, and the R scripts | Variance/feature figures and table; natural-experiment and fitted-effect dashboard views |
| `tables` | `tables/build.py`, registered table producers, `tables/macros.py` | Publication tables, aggregate claims/counts/macros |
| `figures` | `figures/main.py`, registered figure producers | Both publication figure sets |

The following commands are examples:

```bash
uv run --frozen python -m analysis.run_all --stage models
uv run --frozen python -m analysis.run_all --stage difficulty
uv run --frozen python -m analysis.tables.build
uv run --frozen python -m analysis.figures.main --target paper
```

Shared responsibilities are deliberately narrow:

- `corpus.py`: validated community metadata and grouping joins.
- `model_names.py`: canonical configuration/display/parameter vocabulary.
- `model_results/evaluation_data.py`: scored-response loading, score flags,
  community eligibility, and model-by-community accuracy. Figures, tables,
  difficulty models, and the dashboard reuse these definitions.
- `riddle_analysis/corpus_data.py` and `surface_forms.py`: corpus/theme rules
  and deterministic English-text form classification; feature models reuse
  the classifier rather than defining a second taxonomy.
- `style.py`: all figure colors, ramps, markers, hatches, and typography.
  `figures/_common.py` applies them and owns fixed-size export.
- `tables/common.py`: LaTeX escaping, numeric formatting, and table output.
- `utils.py`: CSV writing only.
- `figure_registry.py`: output stem-to-producer inventory for figures, tables,
  and claims; release packaging consumes the same registry.
- `export_dashboard.py`: aggregate serialization, reusing analytical loaders;
  `dashboard/` consumes the JSON without importing analysis dependencies.

## Published artifact dependency map

The [figure guide](figures/README.md) records chart messages, dimensions,
encodings, and individual commands. This map identifies the analytical inputs;
it is also the reason these producers remain in the release.

| Artifacts | Data and shared producers |
| --- | --- |
| `fig_map`, `fig_map_wide`, coverage table | Corpus metadata, canonical riddle counts, local reference map geometry, anonymous annotation coverage |
| `fig_pipeline`, `fig_pipeline_wide` | `evaluation_data.load_flat_frame`, automatic/human/judge routing |
| `fig_model_summary`, model-performance table | `human_eval_summary.build_summary`, model metadata, model-by-community score cells |
| `fig_cultures`, `fig_resourcedness` | Shared community cells and `accuracy_by_group` estimands |
| `fig_errors`, human-error table | `human_error_types` and shared outcome definitions; jury-only rows excluded from error-type shares |
| `fig_variance` | Retained `variance_components.csv` from `prepare`'s provenance-bound input frame |
| `fig_riddle_features`, feature table | Retained feature effects/variance summaries and regenerated feature-frame counts |
| `fig_surface_forms` | Deterministic surface-form classification and per-community shares |
| `fig_rater_agreement`, judge-agreement table | Recorded settings A–C, shared kappa/bootstrap functions and rater summaries |
| `fig_english`, English table | Recorded English answers/verdicts, paired original-language responses, leakage exclusion |
| `fig_judge_by_model`, `fig_judge_by_community`, corresponding tables | Setting-B agreement summaries and shared model/community labels |
| `fig_native_counterfactual` | Setting-B design-weighted human-replacement estimates and bootstrap intervals |
| Judge-prompt table | Shared benchmark judge prompt in `riddles_framework.judging.llm_as_a_judge`; no inference |
| `macros.tex`, `claims.json`, `counts.json` | One `build_commands` result: corpus/contributor counts, score summaries, survey audit, surface forms, English/resource results, judge setting counts and effective sample size |
| Dashboard | Shared model/community cells, country/group estimates, routing, human errors, survey/topic results, paired model settings, English study, provenance-checked R effects/natural experiments |

Regional-specialization, search-flip, wildcard-theme, and judge-failure programs
are not obsolete plot wrappers: they reproduce published prose comparisons or
provide the row/aggregate audits behind them. Their CSVs remain available.
Fresh provider execution and sample preparation are not publication-analysis
stages: their runner, prompt/schema, and sampler moved to
`benchmark/src/riddles_framework/judging/`, with no old-path aliases.

The unused `sync_human_evaluation_metadata.py` maintenance entrypoint is also
removed. It derived a second community-coverage definition and wrote it back
into canonical metadata. Analyses instead obtain coverage from the shared
scored-response loader; updating frozen release metadata is a reviewed data
release operation, not an analysis side effect.

## Accuracy and sample definitions

Accuracy is the share of scored answers with score 3. Scored answers include
automatically resolved cases, valid human labels, and answers on which both
recorded judges agree. Disagreements remain unscored, not incorrect.
`model_results.evaluation_data.load_flat_frame` implements these definitions.

Descriptive model summaries use every community with completed human labels.
Comparisons using `eligible_communities` require at least 600 human-scored
responses per community. Pending answers and coverage metadata remain separate.
Difficulty models mark complete community-by-model blocks and apply the grader
screen; descriptive natural experiments instead average over shared models in
unflagged human-evaluated communities. The figures retain these estimands.

To audit the model summary for all communities outside the registered
directories, run:

```bash
uv run --frozen python -m analysis.figures.fig_model_summary \
  --all-communities --out data/derived/figure_audits
```

Fresh benchmark runs and paid judge outputs belong under `data/runs/` and do
not replace the frozen release. For more information, see the
[benchmark workflow](../benchmark/README.md) and the
[recorded judge-study guide](llm_judge/README.md).
