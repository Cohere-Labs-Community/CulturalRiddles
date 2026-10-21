"""Write number macros and machine-readable claims to ``data/derived/claims``.

Run from the repository root with ``uv run python -m analysis.tables.macros``.
All values use the same populations and calculations as the generated tables.
"""

from __future__ import annotations

import argparse
import json
import re

import pandas as pd
import polars as pl

from analysis.constants import (
    CLAIMS_DIR,
    CONTRIBUTOR_COUNTS_CSV,
    DERIVED_DIR,
    EVAL_ANALYSIS_OUTPUT_DIR,
    HUMAN_ANNOTATIONS_PATH,
    JUDGE_DATA_DIR,
    RIDDLE_WRITING_OUTPUT_DIR,
)
from analysis.corpus import load_corpus_metadata
from analysis.llm_judge.judge_data import run_summaries
from analysis.model_results.evaluation_data import (
    human_evaluated_communities,
    load_flat_frame,
)
from analysis.riddle_analysis.corpus_data import load_corpus
from analysis.riddle_analysis.surface_forms import SHARES_PATH as SURFACE_FORMS_CSV
from analysis.tables.common import (
    JUDGE_STUDY_DIR,
    format_count,
    format_percent,
)
from analysis.tables.coverage import load_iaa_communities
from analysis.tables.english import english_summary
from analysis.tables.human_error_types import (
    build_human_error_types,
    error_type_frame,
    human_evaluation_frame,
)
from analysis.tables.model_performance_summary import build_model_performance

JURY_RULE = "agree-or-human (zero)"


def _jury_community_commands() -> dict[str, str]:
    """Return jury (two-judge rule) kappa macros quoted in the setting-B discussion.

    The rule is the one ``judge_by_community`` sorts by (``JURY_RULE``): the judges'
    score where they agree, the human label otherwise. Values are rounded to two
    decimals like the table cells, and tied extremes are all named.
    """
    meta = pl.from_pandas(load_corpus_metadata()[["community", "language", "country"]])
    by_community = (
        pl.read_csv(JUDGE_STUDY_DIR / "agreement_by_community.csv")
        .filter(pl.col("condition") == JURY_RULE)
        .join(meta, on="community")
        .with_columns(rounded=pl.col("kappa").round(2))
    )
    by_level = (
        pl.read_csv(JUDGE_STUDY_DIR / "agreement_by_resourcedness.csv")
        .filter(pl.col("condition") == JURY_RULE)
        .with_columns(rounded=pl.col("kappa").round(2))
        .select("language_resourcedness", "rounded")
        .iter_rows()
    )
    level = dict(by_level)

    def extreme(value: float) -> str:
        names = sorted(by_community.filter(pl.col("rounded") == value)["language"])
        return " and ".join(names)

    low = float(by_community["rounded"].min())  # ty: ignore[invalid-argument-type]
    high = float(by_community["rounded"].max())  # ty: ignore[invalid-argument-type]
    uncertain = by_community.filter(pl.col("low") < 0.6).sort("language")
    return {
        "jurykappahigh": f"{level['high']:.2f}",
        "jurykappaextremelylow": f"{level['extremely-low']:.2f}",
        "jurykappamin": f"{low:.2f}",
        "juryminlanguage": extreme(low),
        "jurykappamax": f"{high:.2f}",
        "jurymaxlanguage": extreme(high),
        "juryuncertaincommunities": " and ".join(
            f"{row['language']} ({row['country']})" for row in uncertain.iter_rows(named=True)
        ),
    }


def _setting_count_commands() -> dict[str, str]:
    """Return the pair counts of the three judge-validation settings (A, B, C).

    Setting A is the strictest doubly annotated sample: cleaned pairs minus
    placeholder answers and riddles used to tune the judge prompt. Setting B was
    drawn as 20,000 pairs; those with a valid human type and a matched judge
    verdict enter the agreement statistics. All counts come from the committed
    ``llm_judge.rater_agreement`` and ``llm_judge.pending_labels`` aggregates.
    """
    rater_dir = DERIVED_DIR / "llm_judge" / "rater_agreement"
    audit = pl.read_csv(rater_dir / "sample_audit.csv")

    def audit_row(step: str) -> dict[str, int]:
        return audit.filter(pl.col("step") == step).row(0, named=True)

    reliability = pl.read_csv(rater_dir / "judge_reliability.csv")

    def units(setting: str) -> int:
        values = reliability.filter(pl.col("setting") == setting)
        return int(values.select(pl.col("units").cast(pl.Int64).max()).item())

    raw = audit_row("raw IAA pairs")
    cleaned = audit_row("cleaned human pairs")
    primary = audit_row("primary four-rater sample")
    return {
        "numrawiaapairs": format_count(int(raw["pairs"])),
        "numcleanediaapairs": format_count(int(cleaned["pairs"])),
        "numcleanediaariddles": format_count(int(cleaned["riddles"])),
        "numsettingapairs": format_count(int(primary["pairs"])),
        "numsettingariddles": format_count(int(primary["riddles"])),
        "numsettingbpairs": format_count(units("stratified")),
        "numsettingcpairs": format_count(units("pending")),
    }


def _resource_level_commands() -> dict[str, str]:
    """Return the pooled accuracy and community-spread macros of the resource-level figure.

    Values come from ``eval_analysis.accuracy_by_group`` (all human-evaluated
    communities), so the text quotes the numbers the figure draws.
    """
    table = pl.read_csv(EVAL_ANALYSIS_OUTPUT_DIR / "accuracy_by_resourcedness.csv")
    level = {row["group"]: row for row in table.iter_rows(named=True)}
    sd = [row["between_community_sd"] for row in level.values()]
    return {
        "poolaccuracyhigh": f"{level['high']['pooled_accuracy']:.1f}" + r"\%",
        "poolaccuracymedium": f"{level['medium']['pooled_accuracy']:.1f}" + r"\%",
        "poolaccuracylow": f"{level['low']['pooled_accuracy']:.1f}" + r"\%",
        "poolaccuracyextremelylow": f"{level['extremely-low']['pooled_accuracy']:.1f}" + r"\%",
        "resourcesdmin": f"{min(sd):.0f}",
        "resourcesdmax": f"{max(sd):.0f}",
    }


def _surface_form_commands(num_benchmarked: int) -> dict[str, str]:
    """Return the riddle-form macros quoted next to the surface-form figure.

    Counts and shares come from the committed
    ``riddle_analysis.surface_forms`` output, so the text and the figure share
    one population: the benchmarked riddles that have an English text.
    """
    shares = pl.read_csv(SURFACE_FORMS_CSV)

    def percent(community: str, form: str) -> str:
        value = shares.filter(pl.col("community") == community)[form].item()
        return f"{value:.0f}" + r"\%"

    pooled = int(shares.filter(pl.col("community") == "Overall")["n"].item())
    return {
        "numformriddles": format_count(pooled),
        "numformuntranslated": format_count(num_benchmarked - pooled),
        "numformcommunities": format_count(shares.height - 1),
        "formlugandawhoamipercent": percent("luganda_uganda", "who_am_i"),
        "formjapanesewhatisitpercent": percent("japanese_japan", "what_is_it"),
    }


def _english_commands() -> dict[str, str]:
    """Return the English-translation macros quoted in the discussion.

    Accuracy and gain are over the riddles whose translation does not reveal
    the answer, as in the text; the table shows both sets.
    """
    summary = english_summary()
    pooled = summary.without_revealing.filter(pl.col("community") == "all").row(0, named=True)
    return {
        "numenglishriddles": format_count(summary.riddles),
        "numenglishrevealing": format_count(summary.revealing),
        "numenglishrevealinglumasaba": format_count(
            summary.revealing_by_community.get("lumasaba_uganda", 0)
        ),
        "numenglishrevealingsudanese": format_count(
            summary.revealing_by_community.get("arabic_sudan", 0)
        ),
        "numenglishkept": format_count(summary.riddles - summary.revealing),
        "englishoriginalaccuracy": f"{pooled['paired_original']:.1f}" + r"\%",
        "englishaccuracy": f"{pooled['paired_english']:.1f}" + r"\%",
        "englishgain": f"{pooled['paired_delta']:.1f}",
        "englishgainlow": f"{pooled['paired_delta_low']:.1f}",
        "englishgainhigh": f"{pooled['paired_delta_high']:.1f}",
        "englishhighresourceaccuracy": f"{summary.high_resource_accuracy:.0f}" + r"\%",
    }


def _setting_b_design_commands() -> dict[str, str]:
    """Return the setting-B sample size, effective size, and judge token use."""
    sample = pl.read_parquet(JUDGE_DATA_DIR / "setting_b" / "pairs.parquet")
    minimums = pl.read_csv(JUDGE_DATA_DIR / "setting_b" / "community_minimums.csv")
    effective = pl.read_csv(
        DERIVED_DIR / "llm_judge" / "judge_sample_audit" / "effective_size.csv"
    ).filter(pl.col("community") == "Pooled")
    runs = {
        run["judge"]: run
        for run in run_summaries()
        if run["setting"] == "setting_b" and run["shots"] == "zero"
    }

    def tokens(judge: str) -> str:
        prompt = float(str(runs[judge]["prompt_tokens"])) / 1e6
        completion = float(str(runs[judge]["completion_tokens"])) / 1e6
        return f"{prompt:.1f}M/{completion:.1f}M"

    return {
        "numsettingbsampled": format_count(sample.height),
        "settingbeffective": format_count(round(effective["effective_pairs"].item())),
        "numsettingbraisedcommunities": format_count(minimums.height),
        "settingbtokensgemini": tokens("Gemini"),
        "settingbtokensdeepseek": tokens("DeepSeek"),
    }


def build_commands(error_types: pd.DataFrame) -> dict[str, str]:
    """Return the canonical number macros derived from released data."""
    corpus = load_corpus()
    metadata = load_corpus_metadata()
    analysis_frame = load_flat_frame()
    included = human_evaluated_communities(analysis_frame)
    human = human_evaluation_frame()
    iaa_communities = load_iaa_communities(metadata)
    all_runs = load_flat_frame(with_judges=False)
    before_judges = all_runs
    responses = len(before_judges)
    pending = before_judges.loc[before_judges["is_pending"]]
    judged = int(analysis_frame["is_judged"].sum())
    methods = before_judges["scoring_method"].value_counts()
    screened = {
        name: methods.get(method, 0) / len(before_judges) * 100
        for name, method in (
            ("exactmatchpercent", "auto_exact_match"),
            ("emptypercent", "auto_empty"),
            ("idkpercent", "auto_idk"),
            ("placeholderpercent", "auto_placeholder"),
        )
    }
    # Annotators labelled unique (riddle, answer) pairs over all 30 benchmarked
    # configurations. Flatten those labels onto the 28 reported models'
    # responses: a label also covers every response that gave the same answer.
    labels = pd.read_parquet(
        HUMAN_ANNOTATIONS_PATH,
        columns=[
            "riddle_id",
            "community",
            "answer_norm",
            "annotation",
            "benchmark_status",
        ],
    )
    labels = labels.loc[
        labels["annotation"].eq("first")
        & labels["benchmark_status"].eq("valid")
        & labels["community"].isin(included)
    ]

    def answer_keys(frame: pd.DataFrame) -> list[tuple[str, str]]:
        answers = frame["model_answer"].fillna("").astype(str).str.strip().str.lower()
        return list(zip(frame["riddle_id"], answers, strict=True))

    labelled_answers = set(zip(labels["riddle_id"], labels["answer_norm"], strict=True))
    unique_human_answers = len(labelled_answers)
    reported_scope = before_judges.loc[before_judges["community"].isin(included)]
    human_responses = sum(key in labelled_answers for key in answer_keys(reported_scope))
    if human_responses != len(human):
        raise ValueError(
            f"{human_responses} flattened responses but {len(human)} carry a human "
            "score; the error-type and score shares need one denominator"
        )
    reported_human = int(before_judges["is_human"].sum())
    invalid_human_rows = int(before_judges["scoring_method"].eq("invalid_human_eval").sum())
    to_judges = len(pending) + invalid_human_rows
    score_counts = human["score"].value_counts()
    error_percentages = error_types.set_index("code")["percent"]
    # Error-type shares use rule-scored + human-scored responses only (no LLM
    # jury, no unscored); the declined share is checked against the table row.
    error_frame = error_type_frame()
    declined = error_frame["score"].eq(1)
    declined_by_model = declined.groupby(error_frame["model_display"]).mean() * 100
    if abs(declined.mean() * 100 - float(error_percentages["1"])) > 1e-9:
        raise ValueError("declined share disagrees with the error-type table row")
    missing_codes = sorted(
        {"0.1", "0.2", "0.3", "1", "2.1", "2.2", "2.3", "3", "3.1", "3.2"}.difference(
            error_percentages.index
        )
    )
    if missing_codes:
        raise ValueError(f"Human error-type summary lacks expected codes: {missing_codes}")

    contributors = pd.read_csv(CONTRIBUTOR_COUNTS_CSV, index_col="contribution")
    total_writers = int(contributors.at["riddle_writers", "people"])
    total_reviewers = int(contributors.at["annotators", "people"])
    total_second_annotators = int(contributors.at["second_annotators", "people"])
    survey = pd.read_csv(RIDDLE_WRITING_OUTPUT_DIR / "survey_audit.csv").iloc[0]
    survey_respondents = int(survey["responses"])
    survey_communities = int(survey["communities"])
    commands = {
        "numcommunities": format_count(int(corpus["community"].nunique())),
        "numlang": format_count(int(corpus["language_code"].nunique())),
        "numcountries": format_count(int(corpus["country"].nunique())),
        "numriddles": format_count(int(corpus["riddle_id"].nunique())),
        "numbenchmarkedriddles": format_count(int(all_runs["riddle_id"].nunique())),
        "numbenchmarkedcommunities": format_count(int(all_runs["community"].nunique())),
        "nummodels": format_count(int(analysis_frame["model_display"].nunique())),
        "numresponses": format_count(responses),
        "numhumanresponses": format_count(human_responses),
        "numuniquehumanresponses": format_count(unique_human_answers),
        "numinvalidhumanresponses": format_count(invalid_human_rows),
        "invalidhumanpercent": format_percent(
            invalid_human_rows / (reported_human + invalid_human_rows) * 100
        )
        + r"\%",
        "numhumanscoredcommunities": format_count(len(included)),
        "numhumanevaluatedlanguages": format_count(
            int(metadata.loc[metadata["is_human_evaluated"], "language_code"].nunique())
        ),
        "numiaacommunities": format_count(len(iaa_communities)),
        "numhumanevaluatedcommunities": format_count(int(metadata["is_human_evaluated"].sum())),
        "humanevalpercent": format_percent(human_responses / responses * 100) + r"\%",
        "numpendingresponses": format_count(len(pending)),
        "numjudgedresponses": format_count(judged),
        "autoresolvedpercent": format_percent(sum(screened.values())) + r"\%",
        **{name: format_percent(value) + r"\%" for name, value in screened.items()},
        "judgedpercent": format_percent(judged / max(to_judges, 1) * 100) + r"\%",
        "humanroutedpercent": format_percent(reported_human / responses * 100) + r"\%",
        "judgeroutedpercent": format_percent(to_judges / responses * 100) + r"\%",
        "judgeagreedpercent": format_percent(judged / responses * 100) + r"\%",
        "unscoredpercent": format_percent((to_judges - judged) / responses * 100) + r"\%",
        "humanincorrectpercent": format_percent(score_counts.get(0.0, 0) / human_responses * 100)
        + r"\%",
        "humanpartialpercent": format_percent(score_counts.get(2.0, 0) / human_responses * 100)
        + r"\%",
        "humancorrectpercent": format_percent(score_counts.get(3.0, 0) / human_responses * 100)
        + r"\%",
        "humanuncertainpercent": format_percent(score_counts.get(1.0, 0) / human_responses * 100)
        + r"\%",
        "numerrortyperesponses": format_count(len(error_frame)),
        "declinedpercent": format_percent(float(error_percentages["1"])) + r"\%",
        "declinedmaxpercent": format_percent(float(declined_by_model.max())) + r"\%",
        "toobroadpercent": format_percent(float(error_percentages["0.1"])) + r"\%",
        "rightcategorypercent": format_percent(float(error_percentages["0.2"])) + r"\%",
        "hallucinationpercent": format_percent(float(error_percentages["0.3"])) + r"\%",
        "relatedculturepercent": format_percent(float(error_percentages["2.1"])) + r"\%",
        "wrongdetailpercent": format_percent(float(error_percentages["2.2"])) + r"\%",
        "surfacepatternpercent": format_percent(float(error_percentages["2.3"])) + r"\%",
        "correctvariantpercent": format_percent(float(error_percentages["3"])) + r"\%",
        "differentvalidpercent": format_percent(float(error_percentages["3.1"])) + r"\%",
        "correctenglishpercent": format_percent(float(error_percentages["3.2"])) + r"\%",
        **_jury_community_commands(),
        **_setting_count_commands(),
        **_resource_level_commands(),
        **_surface_form_commands(int(all_runs["riddle_id"].nunique())),
        **_english_commands(),
        **_setting_b_design_commands(),
        "bestmodelaccuracy": f"{build_model_performance()['human_accuracy'].max():.1f}" + r"\%",
        "numwriters": format_count(total_writers),
        "numreviewers": format_count(total_reviewers),
        "numsecondannotators": format_count(total_second_annotators),
        "numsurveyrespondents": format_count(survey_respondents),
        "numsurveycommunities": format_count(survey_communities),
        "surveyresponserate": format_percent(survey_respondents / total_writers * 100) + r"\%",
        "surveycommunitycoverage": format_percent(
            survey_communities / int(corpus["community"].nunique()) * 100
        )
        + r"\%",
    }
    return commands


def render_commands_latex(commands: dict[str, str]) -> str:
    """Render the generated commands without changing their formatting."""
    lines = [
        "% Written by analysis.tables.macros from data/canonical and data/derived;",
        "% edits here are overwritten. Change analysis/tables/macros.py instead.",
        *(rf"\newcommand{{\{name}}}{{{value}\xspace}}" for name, value in commands.items()),
    ]
    return "\n".join(lines)


def machine_readable_claims(commands: dict[str, str]) -> dict[str, dict]:
    """Expose each displayed value and its unit without LaTeX formatting."""
    claims = {}
    for name, latex in commands.items():
        percent = latex.endswith(r"\%")
        text = latex.removesuffix(r"\%").replace(",", "")
        value: int | float | str = latex
        if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
            value = float(text) if "." in text else int(text)
        claims[name] = {
            "value": value,
            "unit": "percent" if percent else None,
            "latex": latex,
        }
    return claims


def main() -> None:
    """Write macros, claims, and counts from one set of calculated values."""
    commands = build_commands(build_human_error_types())
    claims = machine_readable_claims(commands)
    counts = {
        name: claim["value"]
        for name, claim in claims.items()
        if name.startswith("num") and isinstance(claim["value"], int)
    }
    CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "macros.tex": render_commands_latex(commands) + "\n",
        "claims.json": json.dumps(claims, ensure_ascii=False, indent=2) + "\n",
        "counts.json": json.dumps(counts, ensure_ascii=False, indent=2) + "\n",
    }
    for filename, contents in outputs.items():
        path = CLAIMS_DIR / filename
        path.write_text(contents, encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
