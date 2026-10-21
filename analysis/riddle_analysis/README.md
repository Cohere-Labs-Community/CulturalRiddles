# Corpus analysis

For setup and the reproducibility command, follow the shared
[analysis conventions](../README.md).

The corpus suite reads `data/canonical/riddles.jsonl` and the community metadata
in `data/canonical/corpus_metadata.csv` (through `analysis/corpus.py`). It writes
CSV summaries under `data/derived/riddle_analysis/`.

To run the whole stage, run `uv run --frozen python -m analysis.run_all --stage corpus`.
To run one producer directly, run a module such as
`python -m analysis.riddle_analysis.surface_forms`. The producers are:

- `surface_forms` defines the riddle surface forms, read from the English
  text of the benchmarked riddles. The difficulty model groups them into
  fewer classes. Regular expressions on the closing sentence of the English text
  (no embeddings or LLM) assign the forms, so a form changes only when the
  text or the rules change.
  `data/derived/riddle_analysis/surface_forms_by_riddle.csv` holds every
  riddle's form and closing question, so a rule change shows up as a row-level
  diff, and `test_surface_forms.py` pins hand-labeled examples.
- `wildcard_themes` groups the wildcard topics into themes.
