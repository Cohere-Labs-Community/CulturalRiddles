# Figures

One script per figure, each drawn at its final size: column width (3.03 in)
or text width (6.30 in or 6.50 in).
Run from the repository root:

```bash
uv run python -m analysis.figures.fig_map                  # preprint map variants
CULTURALRIDDLES_FIGURE_TARGET=paper uv run python -m analysis.figures.fig_map
uv run python -m analysis.figures.main                     # both targets
uv run python -m analysis.figures.main --target preprint   # preprint only
uv run python -m analysis.figures.main --target paper      # paper only
```

Every script writes `<stem>.pdf` (vector, live text) and a review `.png` to
only its selected target. Individual generators default to `preprint`; set
`CULTURALRIDDLES_FIGURE_TARGET=paper` to select the paper output folder.
Other environment values are rejected. The dispatcher defaults to both targets,
running each module in a fresh child process with an explicit target environment:
the shared modules render the paper stems, and all modules render the preprint
stems. The map's wide variant is skipped for paper.

`analysis/style.py` is the single source for the shared green/blue/coral palette,
semantic encodings, sequential ramps, and typography. `_common.py` applies these tokens
before drawing and saves only the selected target; it never recolors completed
artists. Both Matplotlib targets retain bundled DejaVu Sans and the original
final-size typography. `_resource_data.py` holds the resource-level data.
All numbers come from the same loaders as `analysis/tables/`, so the figures,
tables, and macros agree. Canonical inputs are under `data/canonical/`;
regenerated analysis outputs are under `data/derived/<topic>/`.

Run `uv run python -m analysis.run_all` before a figure-only build to
regenerate its inputs. The feature forest plot counts distinct riddles in
`data/derived/build/difficulty/riddle_feature_frame.csv`, rather than using
fixed sample-size labels. After changing the dataset, also run the refit
commands in the [analysis guide](../README.md#rebuild-from-canonical-evidence)
to refresh the R estimates as well as the sample counts.

Inspect the review PNGs in `figures/paper/` and `figures/preprint/`.
A complete render checks that all registered PDF and PNG outputs exist in the
selected target(s), then removes retired `fig_*.pdf` and `fig_*.png` files only
from those folders. A single-target render neither requires nor changes the
other target's outputs. Generation does not require external document directories
and does not copy outputs elsewhere.

To regenerate all tables, number macros, and machine-readable claims,
run from the repository root:

```bash
uv run python -m analysis.tables.build
```

Tables go to `data/derived/tables/`. The `macros.tex`, `claims.json`, and
`counts.json` files go to `data/derived/claims/`. Each claim records its value,
unit (`percent` or `null`), and original LaTeX formatting; counts contain the
integer-valued `num*` claims. Both use the same calculated values as the macros.

## Shared encodings

Every condition retains its secondary marker, hatch, or line-style encoding.
Both output sets use the actual overview artwork palette, not a generic
blue/green replacement. Light fills use dark text or outlines. Green and coral
also have shapes, direct labels, or fixed row positions, rather than relying
on hue alone. The target controls output membership and location, not style.

The following table lists the encodings:

| Encoding | Both paper and preprint |
| --- | --- |
| Outcome (`OUTCOME`) | Correct green; partial peach/hatched; hallucinated coral; other incorrect blue tint/hatched; declined blue/crosshatched; placeholder gray/dotted |
| Resource (`RESOURCE`) | High green circle; medium pale-green square; low peach triangle; extremely low coral diamond |
| Access (`ACCESS`) | Proprietary coral square; open-weight green triangle; open-source blue-tint circle |
| Judges / forms / variance | Shared `JUDGE`, `FORMS`, `VARIANCE` tokens; markers/hatches retained |
| Sequential values | Pale-blue → blue-tint → blue ramps; area and printed counts retain the numerical encoding |

The following table lists the colors that the paper target used before and the preprint tokens that replace them:

| Token / use | Before | After |
| --- | --- | --- |
| Blue | `#0072B2` | `#2D4CB9` |
| Green | `#009E73` | `#5BBF8A` |
| Vermillion | `#D55E00` | `#E0533A` |
| Orange | `#E69F00` | `#F0A08C` |
| Sky / purple | `#56B4E9` / `#CC79A7` | `#8DA2F2` |
| Yellow | `#F0E442` | `#CDE8B4` |
| Black | `#000000` | overview ink `#1A1A1A` |
| Context fill | `#C9C9C9` | overview context `#C4C7CC` |
| High / medium / low / extremely-low resource | `#253494` / `#2C9FD8` / `#2FBFA4` / `#C7E9A0` | `#5BBF8A` / `#CDE8B4` / `#F0A08C` / `#E0533A` |
| Mean bar | `#F0E442` | `#2D4CB9` |
| Rules / human / jury routes | `#F0E442` / `#56B4E9` / `#8A8A8A` | overview rules `#ECECEC` / `#2D4CB9` / `#E6EBFC` |
| Language-count ramp | `#FBD9A6` → `#E69F00` → `#D55E00` → `#7A2E00` | `#E6EBFC` → `#8DA2F2` → `#2D4CB9` |
| Accuracy bins | reversed viridis | white → `#E6EBFC` → `#8DA2F2` → `#2D4CB9` |
| Variance topic | `#CC79A7` | `#5BBF8A` |
| Variance community hatch / resource outline | `#6B4A00` / `#08306B` | ink `#1A1A1A` |
| Variance connector fill / edge | `#E3EDF5` / `#7FA6C4` | `#E6EBFC` / `#8DA2F2` |
| Light-marker outlines / jury hatch | white or no outline | ink `#1A1A1A` |

`OVERVIEW` holds the base palette tokens. Its blue, green, red, pale green,
peach, blue tint, and pale blue are shared with Matplotlib; there is no
post-render hex substitution. Canvas sizes, data, estimands, model mapping,
marker shapes, and hatch patterns are unchanged.

Open-weight means the model releases weights, code, and a technical report;
open-source additionally releases the training data and full training recipes
(see `data/canonical/model_mapping.csv`).

## Shared figures

The following table lists the figures that both targets render:

| Stem | Replaces | Message and design |
| --- | --- | --- |
| `fig_map` | coverage choropleth | 45 countries, India hosts 10 languages. Proportional circles on borderless land, area and color both give the number of languages, inline size key, direct labels. |
| `fig_pipeline` | None | Responses go to rules, native speakers, or the LLM jury; native speakers score more than half of all responses. Single-column Sankey with proportional bands. |
| `fig_model_summary` | model results table | Stacked accuracy bars separate exact match from additional answers accepted by human/jury review; a column reports the percentage of correct answers exact match misses. Half-violins give equally weighted community accuracies without individual black ticks; peach diamonds identify Lumasaba and Mongondow, and a column counts communities below 50%. |
| `fig_cultures` | top and bottom communities raincloud | Accuracy differs sharply between the easiest and the hardest communities. Cloud color is resource level, dots are model configurations (shape: human-review share), outlined bar the mean. |
| `fig_resourcedness` | accuracy by resource level | Accuracy falls with resources, but communities spread widely within each level. Pooled diamonds with 95% intervals joined by a line, dots are communities. Pools all human-evaluated communities. |
| `fig_variance` | variance components | Most benchmark-side difficulty sits in the individual riddle. Top bar: share of total variance; bottom bar: the benchmark variance rescaled to 100%. |
| `fig_errors` | outcomes by model | Hallucination dominates and falls with accuracy; abstention is rare. Correct, hallucinated, and other incorrect on one axis; the remaining outcomes on a zoomed 0-30% axis. |
| `fig_riddle_features` | riddle features table | Local-term answers reduce the odds; longer clues increase them. The refit also finds a modest first-person-form association. Forest plot on a log axis, filled markers: the CI excludes 1. Counts are distinct riddles in the feature frame. |
| `fig_surface_forms` | riddle forms bars | Riddle form varies strongly by community. One dot per community (all 61), overall share as a diamond. |

## Additional figures

The following table lists the additional figures:

| Stem | Message and design |
| --- | --- |
| `fig_map_wide` | `fig_map` at 6.50 in text width, from the same script. |
| `fig_pipeline_wide` | `fig_pipeline` at full width, with every route and score count labeled. |
| `fig_rater_agreement` | Judge agreement with humans depends on setting and granularity. One dumbbell per rater pair, grouped by setting A--C (names, annotator glyphs, and pair counts sit in the margin, before the rater pairs). Three markers give Cohen's kappa for accuracy (solid), the weighted 0--3 score (hatched), and the 13 error types (hollow), with 95% intervals; values repeat in a separate column. The human--human bands are a setting-A reference, not an equivalence test in settings B or C. |
| `fig_english` | Translating into English helps, but closes about a fifth of the gap to high-resource communities. One panel: dumbbell from original to English accuracy per community, with the paired gain and its 95% interval printed beside each row. Solid blue connectors mark gains whose interval excludes zero; dashed vermillion connectors mark the rest. The high-resource accuracy is a reference line. |
| `fig_judge_by_model` | Judge leniency by answering model. Every plotted value is also printed in a column (rejected answers, DS, Gem., Both, then the accuracy shift with its interval), in the table's order; bold shifts have an interval that excludes zero. The accuracy shift uses jury consensus with human labels retained on disagreements, unlike the full human-replacement counterfactual. |
| `fig_judge_by_community` | Judge-human kappa per community, as a chart of the table. Each dot's kappa and the human-scored share correct are printed beside it for every row; bold marks the lowest and highest community in each panel. |
| `fig_native_counterfactual` | Estimated change if the jury replaced every human label on test riddles. The first panel shows per-model shifts under each jury rule, with intervals for agreement-only scoring and its shift printed in a column. The second panel shows community shifts against accuracy, colored and shaped by resource level. `run_all` regenerates the design-weighted estimates and bootstrap intervals from setting B; no API calls. |

The overview image in the root README is static artwork at
`docs/assets/overview.png`. No script generates it.

Unique answers are `(riddle_id, stripped lowercase answer)` pairs, using the
annotation pipeline's Unicode normalization. Resource bars pool scored
responses within each level; error bars exclude jury-only and unscored answers.

`fig_model_summary` also takes `--all-communities` (adds the communities
without human labels and a rank-change column), `--panel bins`, `--panel barcode`
(one tick per community), and `--left dumbbell`. These alternative layouts
require `--out DIR`; the registered default is `--panel violin --left stacked`.
In `fig_native_counterfactual`, community labels use ISO language and country
codes, as in the coverage table.

## Tables kept as tables

The examples, annotation example, judge prompt, prompt box, benchmark
comparison, and error type tables are text lookups or categorical matrices. The
coverage table is a reference list; its shading repeats the printed counts.

