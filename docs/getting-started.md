# Get started

This guide gets you from a fresh clone to a working result in each way people use this repository. None of them needs an API key.

| Your goal | Section | What you get |
| --- | --- | --- |
| Contribute riddles from your community | [Submit riddles](#submit-riddles) | An anonymous workbook that passes the intake check |
| Evaluate a model on the benchmark | [Benchmark a model](#benchmark-a-model) | Raw answers, scores, and a comparison with the released configurations |
| Check the paper's figures, tables, and numbers | [Reproduce the paper](#reproduce-the-paper) | Regenerated figures, LaTeX tables, number macros, and JSON claims |

If you only want to read the riddles or recorded results, skip ahead to [Use the data](#use-the-data).

## Before you begin

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). It installs the pinned Python 3.12 environment for you. Then clone the repository and install its environment:

```bash
git clone https://github.com/Cohere-Labs-Community/CulturalRiddles.git
cd CulturalRiddles
uv sync --frozen
```

That's the whole setup. Run every command in this repository from this folder. All components share one environment and one `uv.lock`, so don't create separate environments.

Each goal has its own extra requirements:

| Requirement | Needed for |
| --- | --- |
| R with the `lme4` and `data.table` packages | Refitting the statistical models, which is optional and separate from `run_all` |
| Provider API keys | Paid model or judge calls only |

## Submit riddles

You contribute one community's set of 100 riddles in an anonymous Excel workbook. A maintainer reviews the content before it enters the corpus.

1. Copy the blank template to a new file. Replace `YOUR_WORKBOOK` with an anonymous description of your community, such as `example_country`:

   ```bash
   cp data/submissions/submission_template.xlsx data/submissions/YOUR_WORKBOOK.xlsx
   ```

2. Open the workbook. On the `Community` sheet, fill in the single row: `community` (a lowercase key such as `example_country`), `language_code`, `language`, `country`, and `country_code`.
3. On the `Riddles` sheet, enter 100 riddles and their reference answers in the original language. Keep each `Number` from 1 through 100 exactly once, and keep 10 riddles per topic.
4. Rename the placeholders `Wildcard 1 [rename]` and `Wildcard 2 [rename]` to two different topics that are specific to your community. You can add English translations, hints, and comments, but they are optional.
5. Don't add names, email addresses, or any per-riddle attribution to cells, file names, comments, or the workbook's author properties.
6. Check the workbook:

   ```bash
   uv run --frozen python -m submissions check data/submissions/YOUR_WORKBOOK.xlsx
   ```

A passing workbook prints `PASS` and exits with code 0:

```text
PASS: data/submissions/YOUR_WORKBOOK.xlsx (0 issue(s), 0 warning(s))
```

If the check prints `FAIL`, it exits with code 1 and lists each blocking problem with worksheet row numbers. The unmodified template always fails, because it has no community row and no riddles. Review any `WARNING` lines too: duplicate answers are allowed but can point to a copy-and-paste mistake.

The check never changes the canonical corpus. A passing check also isn't content approval. To submit the workbook, follow [Submit through GitHub](../CONTRIBUTING.md#submit-through-github), which explains how to add your ignored workbook to a pull request after you review it for personal information.

Maintainers preview the import with `uv run --frozen python -m submissions import PATH_TO_WORKBOOK`. The preview ends with `Dry run; canonical data was not changed.` Writing requires both `--approved` and `--write`. For correction ledgers and the full validation rules, see the [submission tools guide](../submissions/README.md).

## Benchmark a model

You run a model on the riddles, score its answers with the released human labels and judge verdicts, and compare the result with the paper's models.

### Run the offline smoke test

The dummy adapter makes no network requests and downloads no model:

```bash
uv run --frozen riddles-framework --config benchmark/configs/dummy.yaml --limit 5
```

The command prints `Run complete: RUN_ID` and writes `data/runs/RUN_ID/`, which contains `results.jsonl`, `run_config.json`, `run_metadata.json`, and `manifest.json`. Replace `RUN_ID` with that printed value in the rest of this section. Every run directory stays under `data/runs/`, which Git ignores and releases exclude.

### Score the answers

Replace `MODEL_KEY` with a stable label for your model, such as `my-model`:

```bash
uv run --frozen python -m analysis.scoring \
  --responses data/runs/RUN_ID/results.jsonl \
  --model MODEL_KEY \
  --out data/runs/RUN_ID/scored.parquet
```

The scorer applies exact-match and placeholder rules first. Next it reuses a valid human label for a matching `(riddle_id, normalized answer)` pair. Finally it accepts a score when two recorded zero-shot judges agree. Answers that no rule, label, or verdict resolves get a null score. They aren't counted as wrong, and they're listed in `scored_unresolved.parquet`. The dummy model answers every riddle with `dummy_answer`, so all five of its answers stay unresolved.

The `score` column uses 0 for incorrect, 1 for admits not knowing, 2 for partially correct, and 3 for correct. Accuracy is the share of scored answers with a score of 3.

### Run a real model

1. Copy a configuration from `benchmark/configs/benchmark_v0/`, such as `openrouter_deepseek_v4_pro.yaml`.
2. Set `model_spec.provider_model_name` to a model ID that your provider account can call.
3. Set `api_key_env_var` to the environment variable that holds your key. To create a credentials file without overwriting an existing one, run `cp -n .env.template .env`, and then fill in only the key you need. Never commit `.env`.
4. Run the configuration. Replace `CONFIG.yaml` with your copied file name. Provider calls can incur charges, so start with `--limit 20`:

   ```bash
   uv run --frozen --env-file .env riddles-framework \
     --config benchmark/configs/benchmark_v0/CONFIG.yaml --limit 20
   ```

   If you've already exported the key in your shell, omit `--env-file .env`.

To score the results that remain unresolved with fresh judge calls, see [Complete a separate paid judge evaluation](../benchmark/README.md#complete-a-separate-paid-judge-evaluation). You can preview that work without any paid calls by adding `--dry-run` to `riddles-judge`.

### Compare with the released models

The following script prints the accuracy of your run next to the released configurations. Replace `RUN_ID` with your run directory name:

```python
import pandas as pd

released = pd.read_parquet("data/derived/scoring/responses_scored_consolidated.parquet")
new = pd.read_parquet("data/runs/RUN_ID/scored.parquet")
scored = pd.concat([released, new]).dropna(subset=["score"])
accuracy = scored["score"].eq(3).groupby(scored["model"]).mean().sort_values(ascending=False)
print((accuracy * 100).round(1))
```

The released scores exist after you run [Reproduce the paper](#reproduce-the-paper), or after you run only the scoring stage with `uv run --frozen python -m analysis.run_all --stage scoring`. The released models answered all 6,090 benchmarked riddles. A `--limit` run covers only the first riddles, so compare it against the released models on the same riddle IDs rather than on overall accuracy.

A new run never changes the paper's results. The frozen scores stay in `data/canonical/benchmark_results.parquet`. For the full benchmark workflow, including provider routing, see the [benchmark guide](../benchmark/README.md).

## Reproduce the paper

The default workflow rebuilds every figure, table, and number from the recorded evidence. It needs no API keys, no R installation, and no model calls, and it never runs the slow R fits. On a recent laptop it takes about two minutes. To rebuild everything, run:

```bash
uv run --frozen python -m analysis.run_all
```

Then inspect the outputs:

| Output | Location |
| --- | --- |
| Paper and preprint figures, as PDF and PNG | `figures/paper/`, `figures/preprint/` |
| LaTeX tables | `data/derived/tables/` |
| Number macros | `data/derived/claims/macros.tex` |
| Machine-readable claims and corpus counts | `data/derived/claims/claims.json`, `data/derived/claims/counts.json` |
| Rebuilt scores and analysis summaries | `data/derived/` |

Open a few PNG files to check labels and layout. A passing run doesn't prove that a figure looks right.

Optional: to confirm that your copy of the data is the approved snapshot (for example after a download or an unzip), run `uv run --frozen python -m scripts.release_support`. It prints a `Validated` line and fails if any file is missing, changed, or unexpected. No workflow in this guide requires it.

To rebuild one part, run `uv run --frozen python -m analysis.run_all --stage STAGE`, where `STAGE` is one of `scoring`, `judges`, `corpus`, `evaluation`, `models`, `survey`, `difficulty`, `tables`, or `figures`. A stage needs the outputs of the stages before it.

The statistical models are replayed from released fit summaries, and the replay rejects any summary whose inputs have changed. `run_all` never refits them, because the R fits are slow (about 30 minutes) and optional. To refit, install R with `lme4` and `data.table`, and then follow [Refit all statistical models](reproduction.md#refit-all-statistical-models).

To trace a figure or table to its code, see `analysis/figure_registry.py`. For the verification procedure, numerical tolerances, and the self-contained release archive, see the [reproduction guide](reproduction.md).

### Explore the results dashboard

The dashboard is optional and separate from paper reproduction:

```bash
uv run --frozen python -m analysis.export_dashboard
uv run --frozen python -m dashboard.compute
uv run --frozen python -m dashboard.build_static
```

Open `dashboard/index.html` in a browser. It loads Plotly from a CDN, so it needs an internet connection.

## Use the data

The corpus is one row per riddle in `data/canonical/riddles.jsonl`:

```python
import pandas as pd

riddles = pd.read_json("data/canonical/riddles.jsonl", lines=True)
print(
    riddles[
        ["riddle_id", "community", "topic", "riddle_original", "reference_answer_english"]
    ].head()
)
```

The same data is on [Hugging Face](https://huggingface.co/datasets/CohereLabsCommunity/CulturalRiddles). For schemas, score meanings, and the snapshot boundaries, see the [data contract](../data/README.md).

The data terms differ from the code license. You can use the benchmark for evaluation with attribution, but you must not train, fine-tune, or distill models on it, or use it to seed synthetic data. See [`data/LICENSE`](../data/LICENSE).

## Troubleshoot

| Symptom | Fix |
| --- | --- |
| `uv: command not found` | Install uv, and then open a new terminal. |
| `analysis.run_all` rejects a stale or missing fit record | Don't regenerate the provenance file. Restore the released files in `data/derived/difficulty_decomposition/`, or refit as described in [Refit all statistical models](reproduction.md#refit-all-statistical-models). |
| `Output artifacts must remain in a dedicated path beneath repository data/` | Write the template under `data/`, for example `data/submissions/YOUR_WORKBOOK.xlsx`. |
| `Required environment variable not set: NAME` | Export the key named by `api_key_env_var` in your configuration, or run with `--env-file .env`. |
| `release_support` reports drift in a data file | Don't edit or re-freeze the manifest to pass the check. Restore the file. A maintainer must review any intended change to canonical data. |
