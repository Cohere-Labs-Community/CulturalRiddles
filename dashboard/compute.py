"""Load and validate the aggregate JSON contract, without analysis dependencies.

Run ``uv run python -m dashboard.compute`` from the repository root.
The static builder needs only this JSON file and the Python standard library.
Checks compare the dashboard's related views, not historical benchmark values.
"""

from __future__ import annotations

import base64
import json
import math
import statistics
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data/derived/dashboard/dashboard_data.json"


def load(path: Path = DATA) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(d: dict | None = None) -> None:
    if d is None:
        d = load()
    errors: list[str] = []
    n_checks = 0

    def check(name, got, want, tol=1e-6):
        nonlocal n_checks
        n_checks += 1
        if isinstance(got, (int, float)) and isinstance(want, (int, float)):
            ok = math.isfinite(got) and math.isfinite(want) and abs(got - want) <= tol
        else:
            ok = got == want
        if not ok:
            errors.append(f"{name}: got {got!r}, expected {want!r}")

    def percentage(name, value):
        check(
            name,
            isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 100,
            True,
        )

    def indexed(name, rows, key):
        result = {row[key]: row for row in rows}
        check(f"{name} unique keys", len(result), len(rows))
        return result

    lb = indexed("leaderboard", d["leaderboard"], "model")
    communities = indexed("communities", d["communities"], "community")
    check("nonempty leaderboard", bool(lb), True)
    check("nonempty communities", bool(communities), True)
    check("human-evaluated community count", len(communities), d["n_human_evaluated_communities"])
    check(
        "corpus includes evaluated communities",
        d["corpus"]["communities"] >= len(communities),
        True,
    )
    for key, value in d["corpus"].items():
        check(f"corpus {key} positive", value > 0, True)

    routing = d["routing"]
    check(
        "routing partitions responses",
        sum(routing[k] for k in ("rules", "human", "judges", "unscored")),
        routing["responses"],
    )
    check("exact matches are rule-scored", 0 <= routing["exact_match"] <= routing["rules"], True)
    coverage = indexed("coverage", d["coverage"], "community")
    check(
        "coverage response count",
        sum(x["answers"] for x in coverage.values()),
        routing["responses"],
    )
    for key in ("rules", "human", "judges", "unscored"):
        check(f"coverage {key}", sum(x[key] for x in coverage.values()), routing[key])
    check(
        "coverage results membership",
        {k for k, x in coverage.items() if x["in_results"]},
        set(communities),
    )
    for key, row in coverage.items():
        check(
            f"coverage partition {key}",
            sum(row[k] for k in ("rules", "human", "judges", "unscored")),
            row["answers"],
        )

    cells = {(x["model_display"], x["community"]): x for x in d["cells"]}
    check("grid unique pairs", len(cells), len(d["cells"]))
    check("grid coverage", set(cells), {(m, c) for m in lb for c in communities})
    buckets = (
        "communities_eq_0",
        "communities_0_25",
        "communities_25_50",
        "communities_50_75",
        "communities_75_100",
    )
    distributions = indexed("score distribution", d["score_dist"], "model")
    check("score distribution models", set(distributions), set(lb))
    accuracies = [row["human_accuracy"] for row in d["leaderboard"]]
    check("leaderboard descending", accuracies, sorted(accuracies, reverse=True))
    check(
        "leaderboard ranks", [row["rank"] for row in d["leaderboard"]], list(range(1, len(lb) + 1))
    )
    for model, row in lb.items():
        model_cells = [x for (m, _), x in cells.items() if m == model]
        values = [x["value"] for x in model_cells]
        for value in values:
            percentage(f"{model} cell accuracy", value)
        counts = (
            sum(v == 0 for v in values),
            sum(0 < v < 25 for v in values),
            sum(25 <= v < 50 for v in values),
            sum(50 <= v < 75 for v in values),
            sum(v >= 75 for v in values),
        )
        check(f"{model} grid buckets", counts, tuple(row[k] for k in buckets))
        check(f"{model} evaluated communities", len(values), row["n_evaluated_communities"])
        check(f"{model} max", round(max(values), 1), row["max_community_accuracy"], 0.051)
        check(f"{model} min", round(min(values), 1), row["min_community_accuracy"], 0.051)
        check(
            f"{model} review contribution",
            row["human_accuracy"] - row["exact_match"],
            row["difference_pp"],
            0.11,
        )
        dist = distributions[model]
        check(
            f"{model} distribution total",
            sum(dist[k] for k in ("exact", "review3", "s2", "s1", "s0")),
            100,
        )
        check(f"{model} distribution exact", dist["exact"], row["exact_match"], 0.051)
        check(
            f"{model} distribution correct",
            dist["exact"] + dist["review3"],
            row["human_accuracy"],
            0.051,
        )
        total = sum(x["total"] for x in model_cells)
        check(f"{model} scored responses", total, row["n"])
        pooled = sum(x["value"] * x["total"] for x in model_cells) / total
        check(f"{model} grid pooled accuracy", pooled, row["human_accuracy"], 0.051)

    means = [row["mean_accuracy"] for row in d["communities"]]
    check("communities ascending", means, sorted(means))
    for community, row in communities.items():
        values = [cells[m, community]["value"] for m in lb]
        check(f"{community} mean", statistics.mean(values), row["mean_accuracy"])
        check(f"{community} median", statistics.median(values), row["median_model"])
        check(f"{community} below 10", sum(v < 10 for v in values), row["n_below_10"])
        check(f"{community} above 20", sum(v > 20 for v in values), row["n_above_20"])
        for group, column in (
            ("frontier_models", "frontier_accuracy"),
            ("small_models", "small_accuracy"),
        ):
            check(
                f"{community} {column}",
                statistics.mean(cells[m, community]["value"] for m in d[group]),
                row[column],
            )

    # Resource levels pool every human-evaluated community; subregions keep the coverage rule.
    for group, expected in (
        ("resource", len(communities)),
        ("subregion", d["eligible_communities"]),
    ):
        check(
            f"{group} communities",
            sum(x["communities"] for x in d[group]),
            expected,
        )
        for row in d[group]:
            percentage(f"{group} {row['group']}", row["pooled_accuracy"])
    check(
        "group exclusions",
        d["eligible_communities"] + len(d["excluded_from_groups"]),
        len(communities),
    )

    error_types = indexed("error types", d["error_types"], "code")
    labeled = sum(x["responses"] for x in error_types.values())
    hallucination = indexed("hallucination", d["hallucination"], "model_display")
    check("hallucination models", set(hallucination), set(lb))
    check("error labels total", labeled, routing["human"] + routing["rules"])
    check(
        "human label coverage",
        sum(x["human_labels"] for x in hallucination.values()),
        routing["human"],
    )
    for code, row in error_types.items():
        check(f"error {code} share", 100 * row["responses"] / labeled, row["percent"])
    check(
        "hallucinated labels",
        sum(x["hallucinated"] for x in hallucination.values()),
        error_types["0.3"]["responses"],
    )
    for model, row in hallucination.items():
        check(
            f"{model} hallucination share",
            100 * row["hallucinated"] / row["human_labels"],
            row["pct_hallucinated"],
        )

    topics = indexed("topics", d["topic_pooled"], "topic_group")
    for row in d["topic_writing"]:
        check(
            f"topic {row['topic']} accuracy",
            row["human_accuracy"] * 100,
            topics[row["topic"]]["accuracy"],
        )
    for setting, models in d["pair_models"].items():
        check(f"{setting} pair", d["pairs"][setting], [lb[m]["human_accuracy"] for m in models])
    for row in d["judge_fill"]:
        check(
            f"{row['model_display']} judge gain",
            row["accuracy_after"] - row["accuracy_before"],
            row["change_pp"],
        )
        check(
            f"{row['model_display']} pending decrease",
            row["pending_after"] <= row["pending_before"],
            True,
        )
    check("English models in leaderboard", set(d["english_models"]) <= set(lb), True)
    percentage("English high-resource accuracy", d["english_high_resource_accuracy"])
    effect = d["local_term_effect"]
    check(
        "local-term effect positive finite",
        all(
            math.isfinite(effect[k]) and effect[k] > 0 for k in ("odds_ratio", "ci_low", "ci_high")
        ),
        True,
    )
    check(
        "local-term confidence interval",
        effect["ci_low"] <= effect["odds_ratio"] <= effect["ci_high"],
        True,
    )
    for row in d["english"]:
        # The released English-condition CSV rounds percentages to four decimals.
        check(
            f"{row['community']} translation delta",
            row["paired_english"] - row["paired_original"],
            row["paired_delta"],
            0.00015,
        )
    check(
        "riddle solve histogram",
        sum(d["riddle_solves"]["hist"].values()),
        d["riddle_solves"]["riddles"],
    )
    check("agreement models", d["agreement"]["models"], len(lb))
    check(
        "agreement histogram",
        sum(sum(x.values()) for x in d["agreement"]["by_n"].values()),
        d["agreement"]["riddles"],
    )
    check(
        "score outcomes cover scored answers",
        sum(routing["scores"].values()),
        routing["responses"] - routing["unscored"],
    )
    check("admits-not-knowing score", routing["scores"].get("1", 0), routing["admits_not_knowing"])
    percentage("abstention maximum", d["abstention_max"])
    check("corpus model count", d["corpus"]["models"], len(lb))
    sizes = indexed("model sizes", d["model_sizes"], "model_display")
    check("model sizes cover leaderboard", set(sizes), set(lb))

    for community, row in communities.items():
        percentage(f"{community} pooled accuracy", row["pooled_accuracy"])
        total = sum(cells[m, community]["total"] for m in lb)
        pooled = sum(cells[m, community]["value"] * cells[m, community]["total"] for m in lb)
        check(f"{community} pooled accuracy from grid", pooled / total, row["pooled_accuracy"])
    for group in d["resource"]:
        values = [
            x["pooled_accuracy"]
            for x in communities.values()
            if x["language_resourcedness"] == group["group"]
        ]
        check(f"resource {group['group']} community count", len(values), group["communities"])
        check(
            f"resource {group['group']} community mean",
            statistics.mean(values),
            group["mean_community_accuracy"],
        )

    by_topic = {(x["community"], x["topic_group"]): x for x in d["community_topics"]}
    check("community topics unique", len(by_topic), len(d["community_topics"]))
    check("community topics communities", {c for c, _ in by_topic}, set(communities))
    for community in communities:
        rows = [x for (c, _), x in by_topic.items() if c == community]
        for x in rows:
            percentage(f"{community} {x['topic_group']} accuracy", x["accuracy"])
        check(
            f"{community} topic responses",
            sum(x["responses"] for x in rows),
            sum(cells[m, community]["total"] for m in lb),
        )
        check(
            f"{community} topic pooled accuracy",
            sum(x["accuracy"] * x["responses"] for x in rows) / sum(x["responses"] for x in rows),
            communities[community]["pooled_accuracy"],
        )

    matrix = d["riddle_matrix"]
    solved_by = matrix["solved_by"]
    check("riddle matrix models", matrix["models"], [row["model"] for row in d["leaderboard"]])
    check("riddle matrix riddles", len(solved_by), d["riddle_solves"]["riddles"])
    check("riddle matrix communities", len(matrix["community"]), len(solved_by))
    check("riddle matrix community keys", set(matrix["community"]) <= set(communities), True)
    check(
        "riddle matrix histogram",
        dict(Counter(solved_by)),
        {int(k): v for k, v in d["riddle_solves"]["hist"].items()},
    )
    check("riddle matrix order", solved_by, sorted(solved_by, reverse=True))
    rows = [base64.b64decode(bits) for bits in matrix["bits"]]
    check("riddle matrix rows", len(rows), len(lb))
    width = (len(solved_by) + 7) // 8
    check("riddle matrix row width", {len(row) for row in rows}, {width})
    column_sums = [
        sum((row[i >> 3] >> (7 - (i & 7))) & 1 for row in rows) for i in range(len(solved_by))
    ]
    check("riddle matrix column sums", column_sums, solved_by)

    if errors:
        raise ValueError("Dashboard data failed consistency checks:\n" + "\n".join(errors))
    print(f"verify: all {n_checks} checks passed")


if __name__ == "__main__":
    verify()
