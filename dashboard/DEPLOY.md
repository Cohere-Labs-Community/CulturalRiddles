# Rebuild and publish the dashboard

Use the repository's root uv environment. The exporter needs the released
canonical inputs and generated scoring and analysis artifacts under `data/`.
Before exporting after a data change, regenerate those artifacts with:

```bash
uv run python -m analysis.run_all
```

The workflow reads the released aggregate writing-survey and fitted-model
summaries and doesn't refit R models.
The exporter reads the English-condition summary under
`data/derived/llm_judge/english_condition/`, the natural-experiment and fitted
summaries under `data/derived/difficulty_decomposition/`, and survey difficulty
aggregates under `data/canonical/riddle_writing_analysis/`.

## Build

To generate the standalone page, follow these steps from the repository root:

1. Export the aggregate JSON:

   ```bash
   uv run python -m analysis.export_dashboard
   ```

   The destination is `data/derived/dashboard/dashboard_data.json`.

2. Build the page:

   ```bash
   uv run python -m dashboard.build_static
   ```

   The builder prints `verify: all N checks passed` and writes
   `dashboard/index.html`. It reads only the JSON export, not canonical tables
   or analysis CSVs. If a consistency check fails, correct the data or the
   shared calculation rather than replacing it with a fixed expected value.

## Browser smoke checks

Open `dashboard/index.html` directly in a browser with internet access. No local
server is required. Before publishing, check these flows:

1. Confirm that every section contains populated charts, the header shows
   corpus counts, and the browser console has no JavaScript errors.
2. In **Every riddle, every model**, pick a community and hover over cells.
   Confirm that the solve counts in the tooltip and the summary text update.
3. In the leaderboard, switch between **With review** and **Exact match only**
   and the access filters. Hover a community dot and confirm that the same
   community highlights in every row; click it to open that community.
4. In **Communities**, reorder the grid, click a square, and confirm that the
   community page opens and its link (`#community=...`) reopens the same view.
5. Switch the topic chart between **By model** and **Writers' expectations**,
   open each comparison tab, and confirm that hover values agree with the text.
6. Toggle the light and dark themes, reload to check theme persistence, and
   resize to a phone-width viewport. Confirm that charts remain readable and
   navigation reaches each section.

## Publish

Upload the generated `dashboard/index.html` and `dashboard/README.md` to the root
of a static Hugging Face Space or another static host. Keep `sdk: static` and
`app_file: index.html` in the Space README metadata. Do not upload the Python
source, canonical data, or aggregate JSON separately: the HTML embeds the data
that it needs. Plotly and its map assets are fetched from the CDN at runtime.

Paper, dataset, and code links are configured in `LINKS` in
`dashboard/build_static.py`. Empty links are hidden. Rebuild after editing them.
