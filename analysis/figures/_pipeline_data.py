"""Response counts by scoring route and final score, for the pipeline figures."""

from __future__ import annotations

from analysis.model_results.evaluation_data import load_flat_frame

frame = load_flat_frame()
raw = load_flat_frame(with_judges=False)
N = len(frame)
assert N == len(raw)

m = frame["scoring_method"]
RULES = ["auto_exact_match", "auto_empty", "auto_idk", "auto_placeholder"]
n_rules = int(m.isin(RULES).sum())
n_human = int(m.eq("human_eval").sum())
n_jury = int(raw["scoring_method"].isin(["invalid_human_eval", "pending_human_eval"]).sum())
n_agree = int(m.eq("llm_judge_agree").sum())
n_unsc = n_jury - n_agree
assert n_rules + n_human + n_jury == N

route_of = {
    **dict.fromkeys(RULES, "rules"),
    "human_eval": "human",
    "llm_judge_agree": "jury",
}
scored = frame[m.isin(route_of)].assign(route=lambda d: d["scoring_method"].map(route_of))
flow = scored.groupby(["route", "score"]).size().unstack(fill_value=0)
assert {0.0, 1.0, 2.0, 3.0} <= set(flow.columns)
tot = {s: int(flow[s].sum()) for s in (3.0, 2.0, 1.0, 0.0)}
assert sum(tot.values()) == N - n_unsc
n_models = frame["model_name"].nunique()
n_riddles = frame["riddle_id"].nunique()


def pct(v: float) -> str:
    """Share of all responses, as a percentage string."""
    return f"{v / N * 100:.2f}%"
