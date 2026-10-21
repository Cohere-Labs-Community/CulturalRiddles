# Dataset export and release review

Exporting is a local, read-only plan by default. Publishing uploads an exported, reviewed bundle to Hugging Face only when you pass the explicit `--publish` and `--native-review-approved` flags.

From the repository root, inspect the export plan and then write a bundle:

```bash
uv run python -m submissions export
uv run python -m submissions export --write
```

The bundle is written to `data/derived/submissions/export/` and contains a Parquet dataset, a dataset card, the data license, and a manifest. The manifest records the corpus fingerprint, selected benchmark IDs, matching and changed evaluated IDs, removed IDs, and never-benchmarked IDs. A snapshot mismatch after a reviewed corpus update requires native-speaker review before publication.

To preview publication without writing files or contacting Hugging Face, run:

```bash
uv run python -m submissions publish --repo-id OWNER/DATASET
```

After native-speaker review and maintainer approval, publish the already-exported bundle with:

```bash
uv run python -m submissions publish --repo-id OWNER/DATASET --publish --native-review-approved
```

If the destination dataset repository doesn't exist, the publish command creates it with private visibility, then uploads only the bundle files. It uses standard Hugging Face authentication, such as `HF_TOKEN` or a locally stored login token. Never put a token in a workbook or commit it. The command does not change the visibility of an existing repository. To make the repository public, change its visibility setting on Hugging Face as a separate step after you review the uploaded dataset.

The dataset terms in `data/LICENSE` combine CC-BY-4.0 attribution terms with the existing evaluation-only restriction. Preserve the license file and do not label the combined policy as unmodified CC-BY-4.0. Do not add contributor names, per-riddle attribution, or identity fields to an export.
