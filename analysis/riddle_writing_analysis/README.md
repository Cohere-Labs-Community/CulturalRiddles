# Riddle-writing reflection analysis

For root environment setup and the complete reproducibility command, follow the
[analysis guide](../README.md).

After you rebuild scores, regenerate the survey summaries from the repository
root:

```bash
uv run --frozen python -m analysis.run_all --stage survey
```

The only survey inputs are the approved aggregate tables under
`data/canonical/riddle_writing_analysis/`: `survey_audit.csv`,
`topic_difficulty.csv`, `specificity_strategy.csv`, `resources_used.csv`, and
`ai_use_functions.csv`. The producer checks their schemas and writes them to
`data/derived/riddle_writing_analysis/`. A missing source table fails the run;
nothing is skipped silently.

The producer also rebuilds `topic_difficulty_human_accuracy.csv` by joining
aggregate difficulty means to public scored results through
`analysis.model_results.evaluation_data.load_human_evaluation`. The comparison
uses the eight fixed topics and communities with at least 600 human-scored
responses. It retains response counts, eligible-community counts, and the
Spearman correlation alongside the joined means and accuracies.

These tables support number macros and dashboard exports. This package does
not render figures, reconstruct individual survey responses, or code free-text
answers. The canonical aggregate files are the reproducibility boundary for
survey results.
