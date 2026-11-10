# CulturalCrosswords

53 crosswords containing 5,049 clue entries across 42 study language labels and 53 language-country sets. The crosswords were constructed by Poindexter Labs from the crowd-sourced CulturalRiddles corpus provided by Cohere Labs' collaborators.

## Content-reviewed public edition

This edition incorporates the source team's content-review removals. Eight entries from the previous handover have been removed:

| Crossword | Entries removed | Entries retained | Layout update |
|---|---:|---:|---|
| Telugu, India | 6 | 77 | New connected layout using all remaining answers |
| Belarusian, Belarus | 1 | 96 | Removed entry's exclusive cells blocked; clues renumbered |
| Mongondow, Indonesia | 1 | 98 | Removed entry's exclusive cells blocked; clues renumbered |

The other 50 layouts are unchanged. Every retained answer keeps its original spelling and cell units. The three revised grids have been checked for connectivity, consistent crossings and complete clue coverage, with no unclued runs.

Content notes from the public source dataset are included as `riddle_note`. The source team's wording correction to the US English clue about Mars is also included. Other applicable answer and clue corrections were already reflected in the previous handover. Original experiment prompts are omitted, so they cannot retain the removed clues.

This public edition differs from the puzzles used in the paper's experiments. Published scores describe the study version and have not been recomputed for these revised layouts. `release_version` identifies this edition; `layout_revision` identifies the three altered layouts. Puzzle and entry IDs remain stable, so numbering gaps in entry IDs are intentional.

## Files

- `data/crosswords.parquet`: recommended for the Hugging Face dataset viewer.
- `data/crosswords.jsonl`: identical records as readable JSON, one crossword per line.

Each record contains the blank and solved matrices, clues, answers, entry coordinates and language conventions. These files are alternative representations of the same 53 puzzles, not separate sets.

## Adding to the existing Hugging Face dataset

Copy this `crosswords` folder into the root of `CohereLabsCommunity/CulturalRiddles`, replacing the earlier handover if it has already been copied.

Add this single list item under the existing `configs:` section in the root dataset card. Keep its other entries and default unchanged. It adds one option named `crosswords` to the dataset viewer's subset dropdown.

```yaml
- config_name: crosswords
  data_files:
  - split: test
    path: crosswords/data/crosswords.parquet
```

The parent dataset card is generated from its source repository, so retain this addition in its template or generator too.

Load after upload:

```python
from datasets import load_dataset
puzzles = load_dataset("CohereLabsCommunity/CulturalRiddles", "crosswords", split="test")
```

## Reading the matrices

`grid` and `solution` each contain 50 rows of 50 cell strings. `#` marks a block; `.` marks an empty input cell. A cell can contain multiple Unicode code points. Preserve each cell string rather than splitting it into characters.

Coordinates are zero-based `[row, column]`, physically top-to-bottom and left-to-right for every language. For each entry, `grid_units[k]` occupies `cells[k]`. `answer_units` and `answer` use logical spelling order, which can differ from traversal order in mixed-direction text. Follow the supplied language conventions.

Use `direction`, `clue_number`, `start` and `cells` for the current layout. `entry_id` is a stable identifier within its puzzle. `source_answer_id` retains the historical layout reference and must not be used to infer a revised entry's direction or clue number. `source_enumeration` is the original answer-length notation; `hint25` contains optional positions and units from the study's hint condition. Neither defines the current layout.

For a solving task, expose the blank grid, clues, directions and start positions. Keep the solution and answer fields with the evaluator. The package contains data, not a prescribed evaluation prompt.

## Source links and attribution

`source_riddle_id` links an entry to the source corpus when the join is unambiguous. Twenty entries have candidate IDs instead because the clue matches more than one source record. All supplied source IDs and candidates exist in the public riddle table checked for this release. Notes are identical across candidates for each ambiguous entry.

The content review follows [the source repository's corrections list](https://github.com/Cohere-Labs-Community/CulturalRiddles/blob/main/data/corrections.csv). Public source data were checked on 9 November 2026 at Hugging Face revision `b7a5c62b07b36fcafcbbafa8b73e2c48f9465800`.

Suggested attribution:

> Crosswords derived from CulturalRiddles by Isaac Saxton-Knight, Gabriel Saxton-Knight, Diego Oliveira, Tidor-Vlad Pricope, Aston Garratt and Jocelyn D'Arcy (Poindexter Labs). Original crossword layouts by Diego Oliveira, constructed with Crossword Compiler 11.51. The Telugu, Belarusian and Mongondow layouts were revised by Poindexter Labs for this content-reviewed release.

Please also credit CulturalRiddles and its contributors for the source riddles.

- Source dataset: https://huggingface.co/datasets/CohereLabsCommunity/CulturalRiddles
- Source paper: https://arxiv.org/abs/2610.29968
- Crossword paper: Isaac Saxton-Knight et al. (2026), *CulturalCrosswords: A Multilingual Benchmark of Answer Constraints and Grid Completion*.

Retain the parent repository's custom evaluation-only terms and attribution. This package grants no additional license.
