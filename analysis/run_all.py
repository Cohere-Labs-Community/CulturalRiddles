"""Rebuild publication artifacts from canonical evidence in dependency order.

The run uses recorded verdicts and the retained R fit summaries, with no API
calls and no R. It never refits the mixed-effects models; the slow R fits are
separate commands (see ``analysis.difficulty_decomposition``). ``--stage``
selects one stage whose earlier dependencies have already been regenerated.
"""

from __future__ import annotations

import argparse

from analysis import scoring
from analysis.difficulty_decomposition import prepare, riddle_features
from analysis.eval_analysis import accuracy_by_country, accuracy_by_group
from analysis.figures.main import main as run_figures
from analysis.llm_judge import (
    english_condition,
    judge_agreement,
    judge_failures,
    judge_sample_audit,
    native_counterfactual,
    pending_labels,
    rater_agreement,
)
from analysis.model_results import (
    human_eval_summary,
    leaderboard,
    new_model_coverage,
    regional_specialization,
    search_flips,
)
from analysis.riddle_analysis import surface_forms, wildcard_themes
from analysis.riddle_writing_analysis import survey_summaries
from analysis.tables.build import main as build_tables

STAGES = (
    "scoring",
    "judges",
    "corpus",
    "evaluation",
    "models",
    "survey",
    "difficulty",
    "tables",
    "figures",
)


def run_judge_studies() -> None:
    """Analyze recorded verdicts and the English condition without API calls."""
    pending_labels.main([])
    rater_agreement.main([])
    judge_agreement.main([])
    judge_failures.main([])
    judge_sample_audit.main([])
    english_condition.summarize()
    native_counterfactual.run()


def main(stage: str = "all") -> None:
    """Regenerate every stage, or one stage after its inputs exist."""
    if stage not in {"all", *STAGES}:
        raise ValueError(f"Unknown analysis stage: {stage}")
    stages = STAGES if stage == "all" else (stage,)
    for name in stages:
        print(f"\n=== {name} ===", flush=True)
        if name == "scoring":
            scoring.main([])
        elif name == "judges":
            run_judge_studies()
        elif name == "corpus":
            surface_forms.main()
            wildcard_themes.main()
        elif name == "evaluation":
            accuracy_by_group.main()
            accuracy_by_country.main()
        elif name == "models":
            for producer in (
                leaderboard,
                human_eval_summary,
                new_model_coverage,
                regional_specialization,
                search_flips,
            ):
                producer.main()
        elif name == "survey":
            survey_summaries.main()
        elif name == "difficulty":
            prepare.main()
            riddle_features.main()
        elif name == "tables":
            build_tables()
        elif name == "figures":
            run_figures()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild figures, tables, and claims from canonical evidence."
    )
    parser.add_argument(
        "--stage",
        choices=("all", *STAGES),
        default="all",
        help="Run one stage; upstream artifacts must already exist.",
    )
    main(stage=parser.parse_args().stage)
