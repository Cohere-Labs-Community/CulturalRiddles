---
title: Cultural Riddles
emoji: 🧩
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
---

# Cultural Riddles: results dashboard

Paper: [CulturalRiddles: A Multicultural Benchmark of Riddles](https://arxiv.org/abs/2610.29968)

The dashboard is one static HTML page with Plotly loaded from a CDN. It shows
which model solved which riddle, the leaderboard with each model's spread across
communities, a model-by-community grid, a page for each community, geography
and language resources, how answers were scored, the controlled comparisons,
and how the benchmark is built. Published pages need an internet connection for Plotly, but no
Python runtime or application server. The front matter at the top of this file
configures a static Hugging Face Space when you upload this README and the built
page together.

## Data and build contract

Run commands from the repository root, using the root uv environment:

```bash
uv run python -m analysis.export_dashboard
uv run python -m dashboard.build_static
```

`analysis.export_dashboard` uses the shared analysis loaders and table functions,
canonical inputs under `data/canonical/`, and topic summaries under
`data/derived/`. It writes aggregates only to
`data/derived/dashboard/dashboard_data.json`: no answers, riddle text, individual
survey responses, or contributor identities.
The export also carries fitted local-term odds ratios and confidence intervals
from the released riddle-feature summary. Translation comparison model labels
come from the shared English-condition configuration map; the high-resource
comparison is recomputed for those same configurations. These headline values
are rendered from JSON, not fixed in the page template.

The riddle matrix records, for riddles that every model has a score for, which
models solved each one, as one bit per riddle and model. It carries each
column's community and solve count, not riddle IDs or text.

`dashboard.compute` loads that JSON and validates relationships between the
leaderboard, score distributions, model-community grid, per-community topics,
riddle matrix, coverage, error types, topics, and paired comparisons. It does not pin historical numerical results.
`dashboard.build_static` uses only that JSON and Python's standard library; it
does not import the analysis pipeline or read its source tables. It embeds the
validated aggregates in `template.html` (layout, styles and chart code) and
writes the ignored build artifact `dashboard/index.html`.
You can rebuild from an existing JSON export without regenerating analysis.

To run the data consistency checks without writing HTML:

```bash
uv run python -m dashboard.compute
```

For prerequisites, deployment, and browser smoke checks, see
[Rebuild and publish the dashboard](https://github.com/Cohere-Labs-Community/CulturalRiddles/blob/main/dashboard/DEPLOY.md). There is no nested Python project
or lockfile.
