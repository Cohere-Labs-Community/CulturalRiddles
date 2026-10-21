# Cultural Riddles

[![Dataset on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-CulturalRiddles-yellow)](https://huggingface.co/datasets/CohereLabsCommunity/CulturalRiddles)
[![Code: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue)](LICENSE)
[![Data terms](https://img.shields.io/badge/data-evaluation--only-lightgrey)](data/LICENSE)

Cultural Riddles is a multilingual benchmark of 6,191 original riddles written by native speakers from 62 communities. It tests whether language models can solve riddles that depend on lived experience, local associations, traditions, and wordplay. This repository holds the data, the benchmark runner, and the code that reproduces every figure, table, and number in the paper *CulturalRiddles: A Multicultural Benchmark of Riddles*.

The released snapshot records 170,520 answers from 28 model configurations on 6,090 benchmarked riddles. Each answer is scored by rules, anonymous native-speaker labels, or two agreeing LLM judges.

![Overview of the benchmark. Panel A shows example riddles from Canada and Thailand and the topics they cover. Panel B shows how answers are scored: rules first, then native speakers, then an LLM jury. Panel C summarizes the results: accuracy falls with fewer language resources, and models more often hallucinate than admit that they don't know.](docs/assets/overview.png)

## Links

The dataset, its schemas, and its terms are available at these locations:

- **Dataset:** [CohereLabsCommunity/CulturalRiddles on Hugging Face](https://huggingface.co/datasets/CohereLabsCommunity/CulturalRiddles)
- **Results dashboard:** [Cultural Riddles on Hugging Face Spaces](https://huggingface.co/spaces/CohereLabsCommunity/CulturalRiddles)
- **Data contract:** [schemas, score meanings, and snapshot boundaries](data/README.md)
- **Licenses:** code under [Apache-2.0](LICENSE); data under the separate [data terms](data/LICENSE)

## Get started

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and nothing else. It installs the pinned Python 3.12 environment for you.

```bash
git clone https://github.com/Cohere-Labs-Community/CulturalRiddles.git
cd CulturalRiddles
uv sync --frozen
```

That's the whole setup. Run every command from this folder. Everything below works offline and needs no API key.

Choose what you want to do. The following table links each goal to a step-by-step walkthrough in the [getting started guide](docs/getting-started.md).

| Your goal | Start with | You get |
| --- | --- | --- |
| [Submit riddles](docs/getting-started.md#submit-riddles) | `uv run --frozen python -m submissions check data/submissions/YOUR_WORKBOOK.xlsx` | A pass or fail report for your anonymous workbook |
| [Benchmark a model](docs/getting-started.md#benchmark-a-model) | `uv run --frozen riddles-framework --config benchmark/configs/dummy.yaml --limit 5` | Raw answers under `data/runs/`, ready to score |
| [Reproduce the paper](docs/getting-started.md#reproduce-the-paper) | `uv run --frozen python -m analysis.run_all` | Figures, LaTeX tables, number macros, and JSON claims |
| [Use the data](docs/getting-started.md#use-the-data) | `pd.read_json("data/canonical/riddles.jsonl", lines=True)` | One row per riddle |

Replace `YOUR_WORKBOOK` with the name of your completed workbook.

Reproducing the paper doesn't need R, because `analysis.run_all` replays the released summaries of the statistical models. Refitting those models with `analysis.difficulty_decomposition` is optional and slow, and it requires R with the `lme4` and `data.table` packages. For more information, see [Refit all statistical models](docs/reproduction.md#refit-all-statistical-models).

### Credentials for paid calls

Only paid model or judge calls need credentials. Create a local `.env` without overwriting an existing one, fill in only the keys you need, and never commit it:

```bash
cp -n .env.template .env
```

Judges call Gemini and DeepSeek directly, so they need `GEMINI_API_KEY` and `DEEPSEEK_API_KEY`. Solver providers come from the model configuration. See [credentials and provider routing](benchmark/README.md#credentials-and-provider-routing).

## Guides

The following guides cover each area in depth:

| Guide | Read it to |
| --- | --- |
| [Contributing riddles](CONTRIBUTING.md) | Prepare a workbook, open a pull request, and propose corrections |
| [Submission tools](submissions/README.md) | Learn the validation rules and the maintainer import workflow |
| [Benchmark guide](benchmark/README.md) | Run and score models, configure providers, and run judges |
| [Reproduction guide](docs/reproduction.md) | Verify results, refit the statistical models, and build a release archive |
| [Analysis guide](analysis/README.md) | Trace each figure, table, and claim to its code |
| [Human evaluation guide](human_eval/README.md) | Validate additional annotation workbooks |
| [Dashboard guide](dashboard/README.md) | Build the optional results dashboard |
| [AGENTS.md](AGENTS.md) | Follow the maintenance boundaries and verification checks |

## Repository map

The repository has this layout:

```text
benchmark/      Model adapters, runner, judge runner, configurations, and guides
submissions/    Anonymous intake checks, approved imports, corrections, and publishing
human_eval/     Validation code for anonymous annotation workbooks
analysis/       Scoring, aggregate analysis, figures, tables, and claims
scripts/        Data validation and self-contained release verification
dashboard/      Standalone renderer for exported aggregate JSON
docs/           Getting started and reproduction guides
data/
  canonical/    Approved riddles, responses, human labels, judge evidence, and aggregates
  reference/    External reference inputs
  derived/      Rebuilt analysis outputs and retained R model summaries
  manifests/    Approved input schemas, keys, counts, and hashes
  submissions/  Blank intake template; your local workbooks stay ignored by Git
  runs/         Your local model and judge runs (ignored by Git)
figures/        Rendered paper and preprint figures
```

## Develop

Install the commit hooks, and then run the checks from the repository root:

```bash
uv run --frozen pre-commit install --install-hooks
uv run --frozen pytest
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

To build and verify a self-contained archive after you regenerate the outputs, see [Verify a self-contained release](docs/reproduction.md#verify-a-self-contained-release).

## License and use

Code uses [Apache-2.0](LICENSE). The dataset has separate terms in [`data/LICENSE`](data/LICENSE): attribution is required, and you must not use the data to train, fine-tune, or distill models, or to seed synthetic data. The data terms aren't unmodified CC-BY-4.0.
