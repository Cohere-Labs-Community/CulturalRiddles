# Riddle submission and release tools

This root module validates incoming workbooks, imports maintainer-approved community replacements, applies reviewed corrections, and exports the canonical corpus. The canonical riddles live in `data/canonical/riddles.jsonl`; metadata lives in `data/canonical/corpus_metadata.csv`; the correction ledger is `data/corrections.csv`. `data/submissions/` holds the anonymous template and reviewed intake workbooks. This package contains code only.

Only the anonymous blank template is versioned from `data/submissions/`. Intake workbooks, pending correction files, and `data/derived/submissions/` export scratch bundles are ignored by Git and excluded from public reproduction archives.

Run commands from the repository root with the root `uv` environment:

```bash
uv run python -m submissions template
uv run python -m submissions check data/submissions/YOUR_WORKBOOK.xlsx
uv run python -m submissions clean data/submissions/YOUR_WORKBOOK.xlsx
uv run python -m submissions communities
uv run python -m submissions communities --write
uv run python -m submissions import data/submissions/YOUR_WORKBOOK.xlsx
uv run python -m submissions import data/submissions/YOUR_WORKBOOK.xlsx --approved --write
uv run python -m submissions corrections --ledger data/submissions/reviewed_corrections.csv
uv run python -m submissions corrections --ledger data/submissions/reviewed_corrections.csv --approved --write
uv run python -m submissions export
uv run python -m submissions export --write
uv run python -m submissions publish --repo-id OWNER/DATASET
uv run python -m submissions publish --repo-id OWNER/DATASET --publish --native-review-approved
```

`clean` rewrites the named workbook in place without its document author properties and cell comments. `communities` prints the list of communities in the canonical corpus; `--write` updates `docs/communities.md`, and `--check` fails when that file is out of date. `check`, `import` without `--write`, `corrections` without `--write`, and `export` without `--write` are read-only. Import and correction writes require both `--approved` and `--write`. Export writes only when you pass `--write`, and its output must remain beneath `data/`. The manifest compares selected benchmark IDs and their content fingerprints with the immutable `data/canonical/evaluation_corpus.jsonl` baseline; it distinguishes changed or removed evaluated riddles from never-benchmarked and added rows. Publishing is separate from export, read-only by default, and contacts Hugging Face only when you pass `--publish` and `--native-review-approved`. It uploads an existing export bundle, and all bundle paths must remain beneath `data/`.

Replace `YOUR_WORKBOOK` with the name of your workbook, and replace `OWNER/DATASET` with the Hugging Face dataset repository ID.

## Validation contract

`check` and `import` use one validator in `submissions/validation.py`. Both require a readable `.xlsx` workbook with `Community` and `Riddles` sheets, exactly one nonempty community metadata row, a stable lowercase `community` key, and a nonempty `language_code`. Descriptive `language`, `country`, and `country_code` fields are recommended for review; missing values produce warnings. CSV is rejected by both commands because it can't carry the community metadata. Neither command needs a contributor registry, network access, or an API key.

The riddle table requires `Number`, `Topic`, `Riddle (Original Language)`, and `Reference Answer (Original)`, with no empty or whitespace-only required values. It must contain 100 rows, each integer from 1 through 100 exactly once, all eight standard topics, and two distinct renamed wildcard topics, with 10 rows per topic. The standard topics are Politics, History, Geography, Arts, Sports, National Heritage, Kid's World, and Everyday Life. Outer topic whitespace is ignored. IDs normalize equivalent number spellings such as `001` and `1.0`; these spellings cannot bypass duplicate-number checks.

Optional columns are `Riddle (English Translation)`, `Hints (Optional)`, `Comments (Optional)`, and `Reference Answer (English)`. Missing optional columns and extra columns produce warnings; extra columns are not imported. Duplicate column headers are blocking because they make ingestion ambiguous. Identity and attribution column headers are rejected on every worksheet, including extra sheets. This is a structural check, not an identity detector for cell text, workbook properties, comments, or filenames. Review those manually before publishing a workbook.

### Check outcomes

The following table lists what `check` and `import` report:

| Check | Outcome |
| --- | --- |
| Required riddle columns and nonempty original clue and answer | Blocking; errors name missing columns or worksheet rows |
| Identity and attribution columns | Blocking; they're prohibited on every sheet |
| Author names in document properties (author, last modified by, company, manager) or cell comments | Blocking; `clean` removes them |
| Missing, repeated, or invalid `Number` values | Blocking; normalized integers must cover 1 through 100 exactly once |
| Row count and missing numbers | Blocking, because an import replaces a complete 100-riddle community |
| Ten rows per topic and 10 distinct topics | Blocking; eight fixed topics plus two distinct replacements |
| Unrenamed wildcard placeholders or missing standard topics | Blocking; inferred from the complete topic set, not row position |
| Required-cell missing values | Blocking; messages include worksheet row locations, and whitespace-only values count as missing |
| Empty rows below the data | Ignored, including rows that only carry formatting |
| Invisible characters, extra spaces, or different capitalization in topic names | Normalized; recapitalized standard topics produce a warning |
| Duplicate original-language answers, ignoring case and whitespace | Warning with worksheet row groups; repeated answers can be legitimate |
| Duplicate English answers | Warning when the optional column is present |
| Missing optional columns or extra columns | Warning; unrecognized columns aren't imported |
| Reference answer inside its own riddle, or repeated riddle text | Warning with worksheet rows |
| `language_code` not a lowercase ISO 639-3 code, or `country_code` not an uppercase ISO 3166-1 alpha-3 code | Warning |
| Community key already in the corpus (`check` only, after the workbook passes) | Warning; an import would replace that community's set |
| Number ordering and non-wildcard topic changes relative to the template | Warning for review; alignment uses stable `Number` values, not row position |

Validation errors are identical between checking and importing. `check` exits 0 for a pass and 1 for a failed contract. `import` exits 2 for rejected intake or an unapproved write. Warnings are printed by both commands and don't authorize a write. A dry-run import reads the canonical corpus to describe the planned replacement but doesn't mutate it. For template creation, repair, and pull request instructions, see [Contributing riddles](../CONTRIBUTING.md).

Import replaces the full community set. IDs for added riddles derive from community, topic, and `Number`; a wildcard topic whose name has no ASCII spelling uses its position instead (`wildcard_1` or `wildcard_2`, by lowest `Number`); existing IDs are retained by stable `Number`, including topic renames. Changes to any evaluated clue/answer/context field mark the row unbenchmarked. Root scoring checks the frozen evaluation-content fingerprint before attaching historical labels. The required `--ledger` names pending corrections; approved writes append their audit entries to the already-applied `data/corrections.csv`, which cannot itself be replayed as pending. Review content and native-review requirements before approval.

## Data use and release

Exports include the dataset terms from `data/LICENSE`. Preserve the attribution and evaluation-only restrictions. Before you publish a reviewed bundle, see [Dataset export and release review](RELEASE.md).
