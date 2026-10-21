"""Export aggregate dashboard views with the publication's shared score loaders.

Run after ``python -m analysis.run_all``. This exporter reads canonical evidence
and derived summaries, with no live APIs. It writes no answer
or riddle text. The standalone dashboard needs only the resulting JSON.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from analysis.constants import (
    DERIVED_DIR,
    DIFFICULTY_OUTPUT_DIR,
    FIXED_TOPICS,
    RIDDLE_WRITING_OUTPUT_DIR,
)
from analysis.corpus import load_corpus_metadata
from analysis.difficulty_decomposition import prepare, riddle_features
from analysis.difficulty_decomposition.fit_provenance import validate_fit_provenance
from analysis.eval_analysis.accuracy_by_country import country_accuracy
from analysis.eval_analysis.accuracy_by_group import community_accuracy, summarize_grouping
from analysis.llm_judge.english_condition import ENGLISH_MODELS, high_resource_accuracy
from analysis.llm_judge.english_condition import OUTPUT_DIR as ENGLISH_OUTPUT_DIR
from analysis.model_names import model_display_map
from analysis.model_results.evaluation_data import (
    AUTO_METHODS,
    cell_accuracy,
    eligible_communities,
    human_evaluated_communities,
    load_flat_frame,
    load_human_evaluation,
)
from analysis.model_results.human_eval_summary import build_summary
from analysis.model_results.leaderboard import judge_fill_effect
from analysis.riddle_analysis.corpus_data import load_corpus
from analysis.tables.human_error_types import build_human_error_types
from analysis.tables.model_performance_summary import build_model_performance

OUT = DERIVED_DIR / "dashboard" / "dashboard_data.json"
PAIR_MODELS = {
    "claude_reasoning": ("claude-opus-4-7", "claude-opus-4-7-thinking"),
    "deepseek_reasoning": ("deepseek-v4-pro", "deepseek-v4-pro-thinking"),
    "gpt_search": ("GPT5.5-none", "GPT5.5-search"),
}


def clean(value):
    """Convert scalar missing values to JSON null and numpy values to Python."""
    if hasattr(value, "item"):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def records(frame: pd.DataFrame) -> list[dict]:
    """Serialize aggregate rows without leaking pandas or numpy scalar types."""
    return [{key: clean(value) for key, value in row.items()} for row in frame.to_dict("records")]


def export() -> dict:
    """Build the aggregate contract consumed by dashboard.compute/build_static."""
    for group, frame_path, script_path in (
        ("prepare", prepare.FRAME_PATH, prepare.R_SCRIPT),
        ("riddle_features", riddle_features.FEATURE_FRAME_PATH, riddle_features.R_SCRIPT),
    ):
        validate_fit_provenance(
            group, frame_path=frame_path, script_path=script_path, output_dir=DIFFICULTY_OUTPUT_DIR
        )
    meta = load_corpus_metadata().copy()
    meta["label"] = meta["language"] + " (" + meta["country"] + ")"
    labels = meta.set_index("community")["label"]
    frame = load_flat_frame()
    evaluated = human_evaluated_communities(frame)
    results = frame[frame["community"].isin(evaluated)]
    scored_results = results[results["is_scored"]]
    displays = model_display_map()
    corpus = load_corpus()
    out = {
        "n_human_evaluated_communities": len(evaluated),
        "corpus": {
            "riddles": len(corpus),
            "communities": int(corpus["community"].nunique()),
            "languages": int(meta["language_code"].nunique()),
            "countries": int(meta["country"].nunique()),
        },
        "model_displays": displays,
    }
    summary = build_summary()
    leaderboard = build_model_performance().merge(
        summary[
            [
                "model",
                "max_community_accuracy",
                "max_accuracy_community",
                "min_community_accuracy",
                "min_accuracy_community",
                "n",
            ]
        ],
        on="model",
        validate="1:1",
    )
    out["leaderboard"] = records(leaderboard)
    method = frame["scoring_method"]
    routing = frame.assign(
        rules=method.isin(AUTO_METHODS),
        human=frame["is_human"],
        judges=frame["is_judged"],
        unscored=~frame["is_scored"],
    )
    out["routing"] = {
        "responses": len(frame),
        **{name: int(routing[name].sum()) for name in ("rules", "human", "judges", "unscored")},
        "exact_match": int(method.eq("auto_exact_match").sum()),
        "admits_not_knowing": int(frame["score"].eq(1).sum()),
    }
    coverage = routing.groupby("community", as_index=False).agg(
        answers=("rules", "size"),
        rules=("rules", "sum"),
        human=("human", "sum"),
        judges=("judges", "sum"),
        unscored=("unscored", "sum"),
    )
    coverage["label"] = coverage["community"].map(labels)
    coverage["in_results"] = coverage["community"].isin(evaluated)
    out["coverage"] = records(coverage)
    out["judge_fill"] = records(judge_fill_effect())

    cells = cell_accuracy(frame, evaluated)
    grid = cells.pivot(index="community", columns="model_display", values="value")
    communities = cells.groupby("community", as_index=False).agg(mean_accuracy=("value", "mean"))
    for name, models in (("frontier", prepare.FRONTIER_MODELS), ("small", prepare.SMALL_MODELS)):
        model_labels = [displays[model] for model in models]
        out[f"{name}_models"] = model_labels
        communities[f"{name}_accuracy"] = communities["community"].map(
            grid[model_labels].mean(axis=1)
        )
    for name, values in (
        ("n_below_10", (grid < 10).sum(axis=1)),
        ("n_above_20", (grid > 20).sum(axis=1)),
        ("median_model", grid.median(axis=1)),
    ):
        communities[name] = communities["community"].map(values)
    communities = communities.merge(
        meta[["community", "label", "country", "language_resourcedness", "un_m49_subregion"]],
        on="community",
        validate="1:1",
    ).sort_values("mean_accuracy")
    out["communities"] = records(communities)
    out["cells"] = records(cells[["model_display", "community", "value", "total"]])

    grouping = community_accuracy()
    out["eligible_communities"] = len(grouping)
    out["excluded_from_groups"] = sorted(evaluated - set(grouping["community"]))
    resource = summarize_grouping(grouping, "language_resourcedness").set_index("group")
    out["resource"] = records(
        resource.loc[["high", "medium", "low", "extremely-low"]].reset_index()
    )
    out["subregion"] = records(summarize_grouping(grouping, "un_m49_subregion"))
    out["countries"] = records(
        country_accuracy()
        .dropna(subset=["human_correct"])
        .rename(
            columns={"human_correct": "accuracy", "communities_with_human_scores": "communities"}
        )
    )

    parts = pd.DataFrame(
        {
            "exact": scored_results["scoring_method"].eq("auto_exact_match"),
            "review3": scored_results["is_correct"]
            & ~scored_results["scoring_method"].eq("auto_exact_match"),
            "s2": scored_results["score"].eq(2),
            "s1": scored_results["score"].eq(1),
            "s0": scored_results["score"].eq(0),
            "model": scored_results["model_display"],
        }
    )
    out["score_dist"] = records(parts.groupby("model").mean().mul(100).reset_index())
    human = results[results["is_human"]].copy()
    human["hallucinated"] = pd.to_numeric(human["error_type"], errors="coerce").eq(0.3)
    hallucination = human.groupby("model_display", as_index=False).agg(
        human_labels=("is_human", "size"), hallucinated=("hallucinated", "sum")
    )
    hallucination["pct_hallucinated"] = (
        100 * hallucination["hallucinated"] / hallucination["human_labels"]
    )
    out["hallucination"] = records(hallucination)
    out["error_types"] = records(
        build_human_error_types()[["code", "error_type", "responses", "percent", "score_group"]]
    )

    human_frame = load_human_evaluation()
    topics = human_frame[
        human_frame["is_scored"] & human_frame["community"].isin(eligible_communities(human_frame))
    ].copy()
    topics["topic_group"] = topics["topic"].where(topics["topic"].isin(FIXED_TOPICS), "Wildcard")
    pooled = topics.groupby("topic_group", as_index=False).agg(
        accuracy=("is_correct", "mean"), responses=("is_correct", "size")
    )
    pooled["accuracy"] *= 100
    out["topic_pooled"] = records(pooled)
    by_model = topics.groupby(["model_display", "topic_group"], as_index=False).agg(
        accuracy=("is_correct", "mean")
    )
    by_model["accuracy"] *= 100
    out["topic_model"] = records(by_model)
    out["topic_writing"] = records(
        pd.read_csv(RIDDLE_WRITING_OUTPUT_DIR / "topic_difficulty_human_accuracy.csv")[
            ["topic", "mean_difficulty", "human_accuracy"]
        ]
    )
    per_riddle = scored_results.groupby("riddle_id").agg(
        scored=("is_correct", "size"), correct=("is_correct", "sum")
    )
    complete = per_riddle[per_riddle["scored"] == frame["model_display"].nunique()]
    out["riddle_solves"] = {
        "riddles": len(complete),
        "hist": {
            int(k): int(v) for k, v in complete["correct"].value_counts().sort_index().items()
        },
    }
    answers = results.assign(
        answer_norm=results["model_answer"].fillna("").astype(str).str.strip().str.lower()
    )
    counts = (
        answers[answers["answer_norm"].ne("")]
        .groupby(["riddle_id", "answer_norm"], as_index=False)
        .agg(
            n=("model_display", "size"), correct=("is_correct", "max"), scored=("is_scored", "max")
        )
    )
    top = (
        counts.sort_values(["riddle_id", "n"], ascending=[True, False])
        .drop_duplicates("riddle_id")
        .copy()
    )
    top["status"] = "not scored"
    top.loc[top["scored"] & top["correct"], "status"] = "correct"
    top.loc[top["scored"] & ~top["correct"], "status"] = "wrong"
    histogram = top.groupby(["n", "status"]).size().unstack(fill_value=0)
    out["agreement"] = {
        "models": int(frame["model_display"].nunique()),
        "riddles": len(top),
        "by_n": {
            int(k): {key: int(value) for key, value in row.items()}
            for k, row in histogram.iterrows()
        },
    }

    pair_labels = {
        name: [displays[model] for model in models] for name, models in PAIR_MODELS.items()
    }
    accuracy = leaderboard.set_index("model")["human_accuracy"]
    out["pair_models"] = pair_labels
    out["pairs"] = {
        name: [float(accuracy[model]) for model in models] for name, models in pair_labels.items()
    }
    baseline, search = pair_labels["gpt_search"]
    original = results[results["model_display"].eq(baseline)].set_index("riddle_id")
    searched = results[results["model_display"].eq(search)].set_index("riddle_id")
    joined = original[["score"]].join(
        searched[["score", "error_type", "scoring_method"]], rsuffix="_search", how="inner"
    )
    losses = joined[joined["score"].eq(3) & joined["score_search"].lt(3)]
    labeled = losses[losses["scoring_method"].eq("human_eval")]
    shares = (
        labeled["error_type"]
        .map(lambda value: f"{float(value):.1f}")
        .value_counts(normalize=True)
        .mul(100)
    )
    out["search_losses"] = {
        "lost": len(losses),
        "human_labeled": len(labeled),
        "shares": shares.to_dict(),
    }
    english = pd.read_csv(ENGLISH_OUTPUT_DIR / "accuracy_by_community_without_answer_in_riddle.csv")
    english["label"] = english["community"].map(labels).fillna("All")
    out["english"] = records(
        english[
            [
                "community",
                "label",
                "responses",
                "paired_original",
                "paired_english",
                "paired_delta",
                "paired_delta_low",
                "paired_delta_high",
            ]
        ]
    )
    out["english_models"] = [displays[model] for model in ENGLISH_MODELS]
    out["english_high_resource_accuracy"] = high_resource_accuracy()
    natural = pd.read_csv(DIFFICULTY_OUTPUT_DIR / "natural_experiments.csv")
    chinese = natural[natural["comparison"].eq("same_language") & natural["group"].eq("zho")].copy()
    chinese["label"] = chinese["community"].map(labels)
    out["chinese"] = records(chinese[["community", "label", "shared_models", "accuracy"]])
    out["tier_effects"] = records(pd.read_csv(DIFFICULTY_OUTPUT_DIR / "tier_effects.csv"))
    features = pd.read_csv(DIFFICULTY_OUTPUT_DIR / "riddle_feature_effects.csv").set_index("term")
    local_term = features.loc["answer_kindlocal_term"]
    out["local_term_effect"] = {
        name: float(local_term[name]) for name in ("odds_ratio", "ci_low", "ci_high")
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    data = export()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(data, ensure_ascii=False, indent=1, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
