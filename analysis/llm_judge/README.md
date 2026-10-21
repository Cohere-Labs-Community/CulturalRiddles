# Recorded judge studies

This package reads canonical recorded verdicts and produces offline summaries.
It does not run providers or redraw the released study samples. Fresh paid
requests, the shared prompt and verdict schema, and sample preparation live in
[`riddles_framework.judging`](../../benchmark/src/riddles_framework/judging/).
Run fresh inference with `uv run --frozen riddles-judge --help`. For the unpaid
preview and explicit paid commands, see the
[benchmark workflow](../../benchmark/README.md). You don't need to navigate `src/`.

The recorded judges are Gemini 3.5 Flash (`gemini-3.5-flash`) and DeepSeek
V4.1 Flash (`deepseek-v4.1-flash`). Their verdicts live under
`data/canonical/judge/` and `data/canonical/english_condition/`. Offline
reproduction needs no API keys.

## Rebuild the offline summaries

Run these commands from the repository root after `uv sync --frozen`:

```bash
uv run --frozen python -m analysis.run_all --stage scoring
uv run --frozen python -m analysis.run_all --stage judges
```

The complete `uv run --frozen python -m analysis.run_all` command includes both
stages. To regenerate one summary, run its module, for example:

```bash
uv run --frozen python -m analysis.llm_judge.english_condition summarize
uv run --frozen python -m analysis.llm_judge.native_counterfactual
```

The following table lists each module with its inputs, outputs, and consumers.
Paths in the output column are relative to `data/derived/llm_judge/`.

| Module | Canonical inputs | Derived outputs | Consumers |
| --- | --- | --- | --- |
| `pending_labels` | Settings A–C, anonymous human annotations | `rater_agreement/judge_reliability.csv` | Agreement tables, setting counts, label-coverage audit |
| `rater_agreement` | Setting A, anonymous human annotations | `rater_agreement/` | Rater and few-shot agreement figure and table |
| `judge_agreement` | Setting B, corpus metadata | `judge_study/` | Community/model judge figures, tables, and macros |
| `judge_failures` | Setting B, corpus metadata | `judge_failures/` | Shared-error and answer-feature audit |
| `judge_sample_audit` | Setting B | `judge_sample_audit/` | Effective-size macros and sample-composition audit |
| `english_condition` | English responses and recorded verdicts, riddle corpus | `english_condition/` | English figure, table, macros, and dashboard |
| `native_counterfactual` | Setting B, corpus metadata | `native_counterfactual/` | Design-weighted human-replacement figure and intervals |

Accuracy comparisons also read
`data/derived/scoring/responses_scored_consolidated.parquet`. Model joins use
source keys from `data/canonical/model_mapping.csv`, not display-name guesses.
`judge_data.py` owns recorded pair/verdict readers and judge labels. Statistical
agreement functions live in `rater_agreement.py`; other studies reuse them.

## Preserve the recorded study design

The following table lists the recorded settings:

| Setting | Pairs | Conditions |
| --- | --- | --- |
| A | Answers labeled by two annotators | `zero`; `few10` on pairs whose riddles weren't used for prompt tuning |
| B | Human-labeled test pairs, stratified by community and score | `zero` |
| C | Answers without a valid human label | `zero` |

The released samples are frozen records drawn before nine riddles were removed.
Redrawing a sample from the released corpus can produce a different draw; that
is a new study, not paper reproduction. To create a separate sample, run:

```bash
uv run --frozen python -m riddles_framework.judging.judge_study_sample
uv run --frozen python -m riddles_framework.judging.judge_study_sample \
  --design 20000 \
  --community-minimums data/canonical/judge/setting_b/community_minimums.csv
```

These commands write `data/derived/llm_judge/sampling/`, not canonical evidence.
If you pass `--out-dir`, use the same directory in both commands so that the
second command can read `pairs_all.csv`.

## Prepare a separate English-condition run

To write English judge input without calling a provider:

```bash
uv run --frozen python -m analysis.llm_judge.english_condition prepare
```

This writes `judge_input.csv` and `validation.json` under
`data/derived/llm_judge/english_condition/`. Pass that CSV to
`uv run --frozen riddles-judge` with a fresh output under
`data/runs/llm_judge/` for each judge. `summarize` continues to read canonical
recorded verdicts; preparation and paid runs do not replace released evidence.
