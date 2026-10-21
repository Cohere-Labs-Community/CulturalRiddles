# Public data contract

All repository data lives here. Code lives in the root Python packages; rendered figures live in `../figures/paper/` and `../figures/preprint/`.

The following table lists each directory or file and its contract:

| Directory or file | Contract |
| --- | --- |
| `canonical/` | Approved recorded observations, anonymous labels, verdicts, configuration metadata, and writing-survey aggregates |
| `reference/` | External reference inputs used by the analysis |
| `derived/` | Rebuilt scores, summaries, statistical-model inputs/fits, tables, macros, machine-readable claims, and dashboard JSON |
| `manifests/canonical.json` | Exact approved input inventory, SHA-256 content/schema hashes, row counts, and primary keys |
| `corrections.csv` | Reviewed remove/replace/append ledger; applying an already-applied or stale entry fails rather than guessing |
| `submissions/` | Anonymous contribution template and intake location; see `../CONTRIBUTING.md` |
| `runs/` | Local model/judge run artifacts; ignored and excluded from release archives |

## Corpus and configuration metadata

The initial snapshot contains 6,191 riddles from 62 communities. `canonical/riddles.jsonl` has one row per stable `riddle_id`, with community, language, topic, wildcard status, original/English clue and answer, optional hint/note, and `is_benchmarked`. It contains no riddle-author credit or identity. Of these riddles, 6,090 have recorded benchmark responses.

The following files hold the corpus and configuration metadata:

- `canonical/corpus_metadata.csv`: community language/country/resource-level metadata and aggregate contributor counts.
- `canonical/contributor_counts.csv`: aggregate contribution counts, never a person-to-riddle mapping.
- `canonical/model_mapping.csv`: canonical configuration keys (`source_name`), display labels, verified reasoning categories, and parameter counts. Recorded quantization is retained in configuration keys and labels (for example, AWQ); the CSV has no separate quantization column, and unrecorded deployment precision is not inferred.
- `canonical/model_runs.csv`: requested/provider IDs, the adapter/configuration evidence supporting a reasoning or search claim, and official-source references. `unspecified` means the recorded request did not explicitly disable or set reasoning; it does not mean reasoning was off. A provider default is not evidence of a historical request's resolved setting.
- `reference/language_commoncrawl_stats.csv`: recorded language-resource reference statistics.

Model keys identify run configurations, not marketing aliases. Quantization is part of model identity. Historical provider aliases might route to a different model today; recorded responses remain the source for the released results.

`canonical/evaluation_corpus.jsonl` records the original 6,090 evaluated riddle IDs and SHA-256 fingerprints of their public clue/answer/context fields. Score reuse checks those fingerprints as well as `is_benchmarked`, so manually changing a clue without clearing benchmark status fails instead of silently inheriting an old score.

## Responses, routing, and rebuilt scores

`canonical/model_responses.parquet` contains 170,520 recorded answers: 28 configurations × 6,090 riddles. `model` joins to the canonical `source_name`. Parse status, token counts, and observed exact-match metadata remain recorded evidence.

`answer_text` preserves the generation export; `evaluation_answer_text` preserves the candidate actually adjudicated. These differ on 1,750 historical rows, including whitespace, escaped Unicode, and missing/changed extraction results. Historical replay uses the adjudicated candidate so that labels attach to the text their evaluators actually saw. Neither field is a generated score. The rebuild matches every column of the frozen historical result, including all 170,520 scores and scorer assignments.

`canonical/scoring_decisions.parquet` preserves the historical dispatch/curation decisions and rule metadata associated with each response. It is not a substitute for human or judge labels. The released routing is retained rather than claiming to reconstruct historical manual decisions from scratch.

`canonical/benchmark_results.parquet` is a frozen historical reference, not the analysis input. `uv run python -m analysis.scoring` rebuilds `derived/scoring/responses_scored_consolidated.parquet` from responses, routing, first valid human annotations, and two-judge verdicts. All downstream analysis reads the rebuilt file.

Scores are 0 incorrect, 1 admits not knowing, 2 partially correct, 3 correct, and null unscored. Precedence is rules, then a valid first human label, then agreement between the two zero-shot judges. Missing/disagreeing judges never become an incorrect score. `scorer` is `rule`, `human`, `llm_judges`, or `none`. Accuracy is the fraction of scored answers with score 3; null scores are not failures.

`routing` preserves exact-match/empty/placeholder/unknown-answer rules and human-evaluation dispatch, including invalid or pending human labels. The derived rows include normalized answer, error category, anonymous annotator ID/note where relevant, and separate judge scores.

Approved corpus imports retain stable IDs but mark changed or new content unbenchmarked. Historical rebuilds exclude those IDs, and scoring a new run does not reuse historical labels or verdicts for an unbenchmarked riddle. Export audits distinguish the frozen evaluation snapshot from the current corpus. Do not attach old scores to changed clue or reference text merely because an ID still exists.

## Human annotations

`canonical/human_eval/annotations.parquet` has first and independent second labels on unique `(riddle_id, answer_norm)` pairs. Answer normalization is `answer.strip().lower()`. First labels have `benchmark_status` valid or invalid. `iaa_row` and `self_review` identify the agreement subset and repeated self-labeling.

`annotator_id` is a stable opaque identifier such as `annotator_k2p9`, consistent across files. Multiple credited annotators are separated by `;`. The public rubric is `canonical/human_eval/human_review_guidelines.txt`; workbook validation is in `../human_eval/`.

## Judge observations and translation condition

`canonical/judge/setting_{a,b,c}/` contains recorded pair tables and verdicts:

- A: doubly annotated pairs; `tuning_exposed` identifies prompt-development overlap.
- B: stratified human-labeled test sample; `sampling_weight`, population, and community minimums retain the sampling design.
- C: answers without a valid human label; two agreeing zero-shot verdicts can complete scores.

Verdicts are keyed by riddle, normalized answer, judge, and example condition (`shots`: `zero` or `few10`). They retain verdict reasoning and available usage/cost evidence. Prompt-development and few-shot examples live in `canonical/judge/prompt_tuning/`; `canonical/judge/runs.json` records run settings and hashes. The released pair tables have their approved duplicate removals applied.

`canonical/english_condition/{responses,verdicts}.parquet` records the English-translation experiment. These are historical API observations, not fresh requests during reproduction. Paid reruns use explicit commands in `../analysis/llm_judge/README.md` and write under `runs/`.

## Survey aggregates and statistical-model caches

`canonical/riddle_writing_analysis/` contains approved aggregate tables: topic difficulty, resources used, AI-use functions, specificity strategy, and the aggregate survey audit. `derived/riddle_writing_analysis/` is regenerated from those aggregates; the topic-difficulty/human-accuracy join is recomputed from current public scores, not copied from a historical join.

The released R fit-result CSVs live in `derived/difficulty_decomposition/`. Default reproduction rebuilds their public input matrices in `derived/build/`, then checks input-matrix, R-script, and summary hashes against the `*_fit_provenance.json` records. Missing or stale provenance stops replay. `analysis.run_all` never refits; the separate `prepare --fit` and `riddle_features --fit` commands under `analysis.difficulty_decomposition` rerun the mixed-model fits and write fresh summaries, provenance records, and RDS objects under `derived/difficulty_decomposition/`.

## Review and freeze a snapshot

To review and freeze a snapshot, run the following commands from the repository root:

```bash
uv run --frozen python -m scripts.release_support
# Only after reviewing approved changes and anonymity:
uv run --frozen python -m scripts.release_support --freeze --snapshot SNAPSHOT_ID
uv run --frozen python -m analysis.difficulty_decomposition.prepare --fit
uv run --frozen python -m analysis.difficulty_decomposition.riddle_features --fit
uv run --frozen python -m analysis.run_all
```

Replace `SNAPSHOT_ID` with the identifier of the snapshot that you are freezing.

The validator rejects input inventory/content/schema drift, duplicate primary keys, identity-bearing structured fields, and malformed anonymous annotator IDs. It cannot establish that free-text notes contain no personal information; maintainers must review text and exports before freezing or publishing. Do not automatically re-pin a manifest to bypass an unexpected change.

Dataset terms are in [`LICENSE`](LICENSE), separate from the repository's code license. Preserve the attribution and evaluation-only restrictions in exports.
