# Evaluation analysis

This package writes accuracy summaries by UN M49 subregion, language resource
level, and country. Response loading and model summaries belong to
`analysis/model_results/`.

After you rebuild scores, run the following command from the repository root:

```bash
uv run --frozen python -m analysis.run_all --stage evaluation
```

This package writes `accuracy_by_subregion.csv`, `accuracy_by_resourcedness.csv`,
and `accuracy_by_country.csv` under `data/derived/eval_analysis/`. The resource
figure and numeric claims consume the subregion and resource-level files;
dashboard exports consume all of them. Inputs use the shared canonical metadata and computed score paths in
`analysis.constants`.
