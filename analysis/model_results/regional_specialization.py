"""Test whether region-focused models gain on their target languages.

Three comparisons, all on the paper's score-3 accuracy over human-evaluated
communities:

* Each region-specialized Tiny Aya variant against Tiny Aya Global, which
  shares its base model, on every target region. Off-target regions act as a
  control, so a specialization effect should appear only on the diagonal.
* All four Tiny Aya models against each other within each region (Cochran's Q).
* Qwen-SEA-LION 4.5 27B Instruct in-region versus out-of-region, relative to every other
  model. Its base model, Qwen3.6 27B, is not evaluated, so the comparison asks
  whether SEA-LION over-performs in-region given its out-of-region accuracy,
  and reports its in-minus-out gap against Qwen3.5 27B only as a
  difference-in-differences.

Tiny Aya regions follow Table 1 of the Tiny Aya technical report
(arXiv:2603.11510) and the model cards' stated focus regions. SEA-LION target
languages follow its model card. Languages are matched against ``language``
or ``language_group`` in corpus metadata.

Run from the repository root with::

    uv run python -m analysis.model_results.regional_specialization
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, linregress

from analysis.constants import MODEL_RESULTS_OUTPUT_DIR
from analysis.corpus import load_corpus_metadata
from analysis.model_names import model_display_map
from analysis.model_results.evaluation_data import (
    human_evaluated_communities,
    load_flat_frame,
)
from analysis.utils import write_csv

PAIRWISE_CSV = MODEL_RESULTS_OUTPUT_DIR / "regional_specialization.csv"
VARIANTS_CSV = MODEL_RESULTS_OUTPUT_DIR / "regional_specialization_tiny_aya.csv"
IN_REGION_CSV = MODEL_RESULTS_OUTPUT_DIR / "regional_specialization_in_region.csv"
SEA_LION_DID_CSV = MODEL_RESULTS_OUTPUT_DIR / "regional_specialization_sea_lion.csv"

MODEL_DISPLAYS = model_display_map()
REFERENCE_MODEL = MODEL_DISPLAYS["tiny-aya-global"]
REGIONAL_MODELS: dict[str, str] = {
    MODEL_DISPLAYS["tiny-aya-fire"]: "South Asia",
    MODEL_DISPLAYS["tiny-aya-earth"]: "Africa and West Asia",
    MODEL_DISPLAYS["tiny-aya-water"]: "Asia-Pacific",
}
TINY_AYA_MODELS: tuple[str, ...] = (REFERENCE_MODEL, *REGIONAL_MODELS)
REGION_LANGUAGES: dict[str, frozenset[str]] = {
    "South Asia": frozenset(
        {"Bengali", "Gujarati", "Marathi", "Nepali", "Tamil", "Telugu", "Urdu"}
    ),
    "Africa and West Asia": frozenset(
        {"Arabic varieties", "Persian", "Turkish", "Swahili", "Yoruba"}
    ),
    "Asia-Pacific": frozenset(
        {
            "Indonesian",
            "Malay",
            "Vietnamese",
            "Thai",
            "Japanese",
            "Chinese",
            "Traditional Chinese",
            "Simplified Chinese",
        }
    ),
}

SEA_LION_MODEL = MODEL_DISPLAYS["Qwen-SEA-LION-v4.5-27B-IT"]
SEA_LION_COMPARISON_MODEL = MODEL_DISPLAYS["Qwen3.5-27B"]
SEA_LION_LANGUAGES = frozenset(
    {"Burmese", "Filipino", "Indonesian", "Malay", "Tamil", "Thai", "Vietnamese"}
)
SEA_SUBREGION = "Southeast Asia"
MIN_FIT_ACCURACY = 1.0
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 0


def scored_correctness() -> pd.DataFrame:
    """Return score-3 correctness per riddle and model.

    Returns:
        One row per ``(community, riddle_id)`` in human-evaluated communities
        and one float column per model: 1.0 correct, 0.0 incorrect, NaN when
        the response is not scored.
    """
    frame = load_flat_frame()
    rows = frame[
        frame["community"].isin(human_evaluated_communities(frame)) & frame["is_scored"]
    ].drop_duplicates(["community", "riddle_id", "model_display"])
    return rows.pivot(
        index=["community", "riddle_id"], columns="model_display", values="is_correct"
    ).astype("float64")


def communities_for(
    communities: set[str],
    languages: frozenset[str] | None = None,
    subregion: str | None = None,
) -> list[str]:
    """Select evaluated communities by language or UN M49 subregion.

    Args:
        communities: Evaluated community identifiers to select from.
        languages: Names matched against ``language`` or ``language_group``.
        subregion: Value matched against ``un_m49_subregion``.

    Returns:
        Sorted identifiers matching either criterion.

    Raises:
        ValueError: If no evaluated community matches.
    """
    metadata = load_corpus_metadata()
    metadata = metadata[metadata["community"].isin(communities)]
    match = pd.Series(False, index=metadata.index)
    if languages is not None:
        match |= metadata["language"].isin(languages)
        match |= metadata["language_group"].isin(languages)
    if subregion is not None:
        match |= metadata["un_m49_subregion"].eq(subregion)
    selected = sorted(metadata.loc[match, "community"])
    if not selected:
        raise ValueError(f"No evaluated community for {languages=} {subregion=}")
    return selected


def sign_test(first: pd.Series, second: pd.Series) -> tuple[int, int, float]:
    """Run a two-sided exact sign test on paired binary outcomes.

    Args:
        first: Boolean correctness of the first model.
        second: Boolean correctness of the second model on the same riddles.

    Returns:
        Riddles only ``first`` solved, riddles only ``second`` solved, and the
        p-value (NaN without discordant riddles).
    """
    first_only = int((first & ~second).sum())
    second_only = int((~first & second).sum())
    discordant = first_only + second_only
    p_value = binomtest(first_only, discordant).pvalue if discordant else float("nan")
    return first_only, second_only, p_value


def cochrans_q(outcomes: np.ndarray) -> tuple[float, float]:
    """Test whether k paired binary outcomes share one success rate.

    Args:
        outcomes: Array of shape ``(n_items, k)`` with 0/1 entries.

    Returns:
        Cochran's Q statistic and its chi-squared p-value with k-1 df.
    """
    k = outcomes.shape[1]
    column_totals = outcomes.sum(axis=0)
    row_totals = outcomes.sum(axis=1)
    total = row_totals.sum()
    denominator = k * total - (row_totals**2).sum()
    if denominator == 0:
        return 0.0, 1.0
    statistic = (k - 1) * (k * (column_totals**2).sum() - total**2) / denominator
    return float(statistic), float(chi2.sf(statistic, k - 1))


def tiny_aya_frames(correct: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare Tiny Aya models within each region.

    Args:
        correct: Output from :func:`scored_correctness`.

    Returns:
        A pairwise table of each regional variant against Tiny Aya Global in
        every region, and a per-region table of all four accuracies with
        Cochran's Q across them. Both use riddles scored for all four models.
    """
    aya = correct[list(TINY_AYA_MODELS)].dropna().astype(bool)
    community = aya.index.get_level_values("community")
    regions = {
        region: communities_for(set(community), languages)
        for region, languages in REGION_LANGUAGES.items()
    }
    pairwise, variants = [], []
    for region, members in regions.items():
        subset = aya[community.isin(members)]
        accuracy = subset.mean() * 100
        statistic, p_value = cochrans_q(subset.to_numpy(dtype=int))
        min_pair_p = min(sign_test(subset[a], subset[b])[2] for a, b in combinations(aya, 2))
        variants.append(
            {
                "region": region,
                "n_communities": len(members),
                "n_riddles": len(subset),
                **{f"{model} accuracy": accuracy[model] for model in aya},
                "all_agree_share": subset.nunique(axis=1).eq(1).mean() * 100,
                "cochran_q": statistic,
                "cochran_q_p": p_value,
                "min_pairwise_sign_test_p": min_pair_p,
            }
        )
        for variant, target in REGIONAL_MODELS.items():
            variant_only, global_only, p_value = sign_test(subset[variant], subset[REFERENCE_MODEL])
            pairwise.append(
                {
                    "variant": variant,
                    "region": region,
                    "is_target_region": region == target,
                    "n_communities": len(members),
                    "n_riddles": len(subset),
                    "variant_accuracy": accuracy[variant],
                    "global_accuracy": accuracy[REFERENCE_MODEL],
                    "difference_pp": accuracy[variant] - accuracy[REFERENCE_MODEL],
                    "variant_only": variant_only,
                    "global_only": global_only,
                    "sign_test_p": p_value,
                }
            )
    return pd.DataFrame(pairwise).round(3), pd.DataFrame(variants).round(3)


def in_region_profile(correct: pd.DataFrame, members: list[str]) -> pd.DataFrame:
    """Rank every model's in-region accuracy against its out-of-region accuracy.

    A line fit on the log-odds scale across models predicts in-region accuracy
    from out-of-region accuracy; the residual is a model's in-region
    over-performance beyond what its general strength explains.

    Args:
        correct: Output from :func:`scored_correctness`.
        members: In-region community identifiers.

    Returns:
        One row per model with accuracy above ``MIN_FIT_ACCURACY`` both in and
        out of region, sorted by residual.
    """
    in_region = correct.index.get_level_values("community").isin(members)
    table = pd.DataFrame(
        {
            "in_region_accuracy": correct[in_region].mean() * 100,
            "out_region_accuracy": correct[~in_region].mean() * 100,
        }
    )
    table = table[table.gt(MIN_FIT_ACCURACY).all(axis=1)]
    log_odds = np.log(table / (100 - table))
    table["gap_pp"] = table["in_region_accuracy"] - table["out_region_accuracy"]
    fit = linregress(log_odds["out_region_accuracy"], log_odds["in_region_accuracy"])
    table["residual_log_odds"] = log_odds["in_region_accuracy"] - (
        fit.intercept + fit.slope * log_odds["out_region_accuracy"]
    )
    for column in ("in_region_accuracy", "out_region_accuracy", "residual_log_odds"):
        rank = table[column].rank(ascending=False, method="min").astype(int)
        table[column.replace("accuracy", "rank").replace("_log_odds", "_rank")] = rank
    table.index.name = "model"
    return table.sort_values("residual_log_odds", ascending=False)


def difference_in_differences(correct: pd.DataFrame, members: list[str]) -> dict[str, float]:
    """Bootstrap SEA-LION's in-minus-out gap relative to Qwen3.5 27B.

    Communities are resampled with replacement separately inside and outside
    the region, keeping riddles clustered within their community.

    Args:
        correct: Output from :func:`scored_correctness`.
        members: In-region community identifiers.

    Returns:
        The observed difference-in-differences in points and its 95%
        percentile interval.
    """
    paired = correct[[SEA_LION_MODEL, SEA_LION_COMPARISON_MODEL]].dropna()
    difference = paired[SEA_LION_MODEL] - paired[SEA_LION_COMPARISON_MODEL]
    per_community = difference.groupby(level="community").agg(["sum", "count"])
    in_region = per_community.index.isin(members)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    def mean_difference(cells: pd.DataFrame, draws: np.ndarray) -> np.ndarray:
        sums = cells["sum"].to_numpy()[draws].sum(axis=-1)
        return sums / cells["count"].to_numpy()[draws].sum(axis=-1)

    inside, outside = per_community[in_region], per_community[~in_region]
    observed = (
        inside["sum"].sum() / inside["count"].sum() - outside["sum"].sum() / outside["count"].sum()
    )
    boots = mean_difference(
        inside, rng.integers(len(inside), size=(BOOTSTRAP_SAMPLES, len(inside)))
    ) - mean_difference(outside, rng.integers(len(outside), size=(BOOTSTRAP_SAMPLES, len(outside))))
    low, high = np.percentile(boots, [2.5, 97.5])
    return {
        "did_pp": observed * 100,
        "did_ci_low_pp": low * 100,
        "did_ci_high_pp": high * 100,
    }


def sea_lion_frames(correct: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profile every model in and out of SEA-LION's region under two definitions.

    Args:
        correct: Output from :func:`scored_correctness`.

    Returns:
        The per-model in-region profile for each region definition, and one
        SEA-LION summary row per definition with its ranks and the
        difference-in-differences against Qwen3.5 27B.
    """
    communities = set(correct.index.get_level_values("community"))
    definitions = {
        "SEA-LION languages": communities_for(communities, SEA_LION_LANGUAGES),
        "Southeast Asia": communities_for(communities, subregion=SEA_SUBREGION),
    }
    profiles, summaries = [], []
    for definition, members in definitions.items():
        profile = in_region_profile(correct, members)
        profiles.append(profile.reset_index().assign(region_definition=definition))
        summaries.append(
            {
                "region_definition": definition,
                "communities": ";".join(members),
                "n_models_ranked": len(profile),
                **profile.loc[[SEA_LION_MODEL]].to_dict(orient="records")[0],
                f"{SEA_LION_COMPARISON_MODEL} residual_rank": profile.loc[
                    SEA_LION_COMPARISON_MODEL, "residual_rank"
                ],
                **difference_in_differences(correct, members),
            }
        )
    profile_table = pd.concat(profiles, ignore_index=True)
    columns = ["region_definition", *profile_table.columns[:-1]]
    return profile_table[columns].round(3), pd.DataFrame(summaries).round(3)


def main() -> None:
    """Write the Tiny Aya and SEA-LION regional-specialization CSVs."""
    correct = scored_correctness()
    pairwise, variants = tiny_aya_frames(correct)
    profiles, sea_lion = sea_lion_frames(correct)
    for frame, path in (
        (pairwise, PAIRWISE_CSV),
        (variants, VARIANTS_CSV),
        (profiles, IN_REGION_CSV),
        (sea_lion, SEA_LION_DID_CSV),
    ):
        print(f"Wrote {write_csv(frame, path)}")


if __name__ == "__main__":
    main()
