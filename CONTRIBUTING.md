# Contributing riddles

Submit an anonymous workbook through a pull request from your fork. Automated checks validate the intake contract; a maintainer reviews the content before importing it. Passing validation is not content approval and never automatically adds riddles to the canonical corpus.

## Before you begin

Check [the list of communities](docs/communities.md) already in the benchmark. If yours is not there, open an [Add my community](https://github.com/Cohere-Labs-Community/CulturalRiddles/issues/new?template=add-community.yml) issue before you start writing, so that others can see it is in progress and maintainers can confirm the community key. If your community is already listed, open an issue to discuss improvements instead of submitting a replacement: an import replaces the community's whole set.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone your fork, and run `uv sync --frozen` from the repository root. These checks run locally without API keys or network downloads after dependency installation.

Do not include personal names, email addresses, contributor identities, or per-riddle attribution in sheet columns, cell contents, filenames, comments, or workbook author properties. The template has blank author properties, but Excel and other spreadsheet apps add your name to them when you save. The check fails while the workbook stores a name in its document properties (author, last modified by, company, manager) or has cell comments, which record their author. Structural validation rejects identity columns but cannot determine whether free text contains personal information; review the complete workbook before making it public.

## Prepare and check a workbook

1. Copy [`data/submissions/submission_template.xlsx`](data/submissions/submission_template.xlsx) to `data/submissions/YOUR_WORKBOOK.xlsx`. Replace `YOUR_WORKBOOK` with an anonymous description of your community. To generate a fresh copy instead, run:

   ```bash
   uv run python -m submissions template --output data/submissions/YOUR_WORKBOOK.xlsx
   ```

2. Fill in the single row on the `Community` sheet. Set `community` to a stable key such as `example_country` and `language_code` to your community's language code. The community key must contain lowercase ASCII letters, digits, and single underscores. Fill in `language`, `country`, and `country_code` for maintainer review; missing descriptive fields produce warnings.
3. Complete the `Riddles` sheet. Keep the exact column headers, enter 100 original-language riddles and their reference answers, and use each `Number` from 1 through 100 exactly once. Keep each number stable when you correct a submitted set. English translations, hints, and comments are optional.
4. Keep the eight standard topics, with 10 riddles per topic. Rename `Wildcard 1 [rename]` and `Wildcard 2 [rename]` to two distinct community-specific topics, also with 10 riddles each. Surrounding topic whitespace is ignored. Rows can be reordered; IDs use `Number`, not worksheet position.
5. Check the completed workbook:

   ```bash
   uv run python -m submissions check data/submissions/YOUR_WORKBOOK.xlsx
   ```

   If the check reports personal information stored outside the cells, remove it and check again:

   ```bash
   uv run python -m submissions clean data/submissions/YOUR_WORKBOOK.xlsx
   ```

   `clean` rewrites the workbook in place without its author properties and cell comments. It does not change cell text.

   `PASS` exits with code 0. `FAIL` exits with code 1 and lists blocking issues, including worksheet row numbers for missing values. Review `WARNING` messages as well: duplicate answers are allowed, but can indicate copying mistakes. Warnings also flag a riddle whose reference answer appears in its own text, repeated riddle text, language and country codes in an unexpected format, and a community key that already exists. Missing optional columns, unimported extra columns, and changes to the template's topic assignments are also reported for review. A file that cannot be read is rejected with a repair hint. Both `check` and `import` require an `.xlsx` workbook with `Community` and `Riddles` sheets; CSV is not a complete intake format.

For the full list of checks, see [Validation contract](submissions/README.md#validation-contract).

## Submit through GitHub

Workbooks in `data/submissions/` are ignored by default so that unreviewed intake is not accidentally committed. A public pull request makes your workbook public. Review its contents and file properties before explicitly adding it:

```bash
git add -f data/submissions/YOUR_WORKBOOK.xlsx
```

Commit the workbook to your fork and open a pull request. The `Validate riddle submission` check runs without repository secrets. If the check fails, correct the reported issues and update the pull request. Wait for a maintainer to review the content and approve the import. Do not edit the canonical corpus or immutable benchmark evidence to submit a workbook.

## Preview and approve an import

Maintainers can inspect a planned community replacement without writing:

```bash
uv run python -m submissions import data/submissions/YOUR_WORKBOOK.xlsx
```

The command runs the same validation as `check` and prints the community's old and incoming row counts. Successful previews end with `Dry run; canonical data was not changed.` The preview reads canonical data but does not change it or contact a service.

**Caution:** Import replaces the full community set. Review the community key and all 100 incoming rows before approving a write. Both flags are required to write:

```bash
uv run python -m submissions import data/submissions/YOUR_WORKBOOK.xlsx --approved --write
```

IDs derive from the community, topic, and `Number`. Existing rows retain their ID by stable `Number`, including when a wildcard topic is renamed. Changes to an evaluated clue, answer, translation, hint, note, or context mark the row unbenchmarked so that evaluation results aren't silently attached to changed content. Approval is a human review decision; a passing structural check does not establish cultural authenticity, accuracy, or permission to publish.

## Correct canonical riddles

Put reviewed corrections in a separate pending file such as `data/submissions/reviewed_corrections.csv`, with columns `riddle_id,action,field,old_value,new_value,reason`. `data/corrections.csv` is already-applied history and must not be replayed as pending changes. Actions are `remove`, `replace`, and `append`. Replace and append require the exact current old value (empty for an empty append target); already-applied replacements and appends are idempotent. Ambiguous or stale corrections fail without writing.

Preview the pending corrections:

```bash
uv run python -m submissions corrections --ledger data/submissions/reviewed_corrections.csv
```

After review, apply them explicitly:

```bash
uv run python -m submissions corrections --ledger data/submissions/reviewed_corrections.csv --approved --write
```

An approved write updates the corpus and appends entries to `data/corrections.csv`. Edited context is marked unbenchmarked; review the changed corpus, audit ledger, and snapshot manifest before release.

## Check annotation workbooks

Human evaluation uses a four-point rubric and consistency checks. Use a stable anonymous `Annotator ID`; do not include a person's name or identity columns. To validate a folder of `Human Eval <language>_<country>.xlsx` files, run:

```bash
uv run python -m human_eval.validate_humaneval_submission PATH_TO_WORKBOOKS
```

Replace `PATH_TO_WORKBOOKS` with the folder that contains the annotation workbooks. See the [human evaluation rubric](data/canonical/human_eval/human_review_guidelines.txt).
