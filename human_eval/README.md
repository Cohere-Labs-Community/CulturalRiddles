# Human evaluation

Human evaluation uses the four-point rubric and error-type consistency rules in the [human review guidelines](../data/canonical/human_eval/human_review_guidelines.txt). Keep the main and Extras sheet conventions, and complete Extras only when requested. Use a stable anonymous `Annotator ID`, not a name.

## Validate submitted workbooks

The validator checks submission structure and data quality. It is not an evaluator: it does not assign scores, import workbooks into canonical evidence, or update benchmark results. It reads workbook files without modifying them or writing logs beside them.

Run validation from the repository root:

```bash
uv run python -m human_eval.validate_humaneval_submission PATH_TO_WORKBOOKS
```

Replace `PATH_TO_WORKBOOKS` with the directory containing `Human Eval <language>_<country>.xlsx` workbooks. The validator checks both sheets, required values, anonymous IDs, and notes for `Other` error categories; it rejects identity/name columns.

## Use released annotations when scoring a model run

The released annotation evidence is `data/canonical/human_eval/annotations.parquet`. To score a new benchmark run against released human labels and recorded judge verdicts, use the command in the [benchmark scoring workflow](../benchmark/README.md#score-a-new-run-with-released-evidence). The scoring tool matches labels to the same riddle and normalized answer, applies valid first human labels after explicit answer rules, and writes derived scores outside canonical data.

Workbook validation does not add annotations to this Parquet file, and the repository has no workbook-to-annotations import command. Annotations that you create must go through a reviewed evidence release before scoring can consume them.
