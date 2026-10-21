# Model-results analysis

For setup and score regeneration, follow the [analysis guide](../README.md).
This package reads computed consolidated scores from
`data/derived/scoring/responses_scored_consolidated.parquet` and writes
aggregate results under `data/derived/model_results/`.

From the repository root, run:

```bash
uv run --frozen python -m analysis.run_all --stage models
```

The following table lists the modules and the CSV files they generate:

| Module | Generated CSVs |
| --- | --- |
| `human_eval_summary` | `human_evaluated_model_summary.csv`: accuracy, exact match, and per-community spread |
| `leaderboard` | `judge_fill_effect.csv`: accuracy with and without recorded judge-agreed scores |
| `new_model_coverage` | `new_model_coverage_by_model.csv`: coverage supplied by existing labels |
| `regional_specialization` | `regional_specialization*.csv`: regional comparisons and matched-model controls |
| `search_flips` | `search_flips.csv`: human-labeled error types when web search changes a correct answer to an incorrect answer |

Canonical recorded answers are in `data/canonical/model_responses.parquet`.
Model configuration keys, display names, reasoning settings, and parameter
metadata come from `data/canonical/model_mapping.csv`. Fresh framework runs
live under `data/runs/` and do not automatically replace release inputs.
