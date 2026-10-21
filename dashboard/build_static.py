"""Build the static Cultural Riddles dashboard (index.html).

Reads ``data/derived/dashboard/dashboard_data.json``, validates relationships
between the exported views, and bakes everything into one page (Plotly from
a CDN). Building requires only Python's standard library; serving the
result requires no Python process or server.

    uv run python -m dashboard.build_static

Optional links show up only when set below. Leave them empty until the
addresses are final.
"""

from __future__ import annotations

import json
from pathlib import Path

from dashboard import compute as C

OUT = Path(__file__).resolve().parent / "index.html"

LINKS = {
    "demo": "",
    "quiz": "",
    "paper": "",
    "dataset": "",
}


def main() -> None:
    data = C.load()
    C.verify(data)
    data["links"] = {k: v for k, v in LINKS.items() if v}
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    OUT.write_text(TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cultural Riddles</title>
<meta name="description" content="Explore community-written riddles, model results, and native-speaker evaluation in the Cultural Riddles benchmark.">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{
  color-scheme:light;
  --page:#f9f9f7; --surface:#fcfcfb; --card:#ffffff; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10); --wash:#eef4fc;
  --blue:#2a78d6; --blue-dk:#1c5cab; --blue-lt:#86b6ef; --blue-xlt:#cde2fb;
  --orange:#eb6834; --yellow:#eda100; --aqua:#1baf7a; --red:#e34948; --violet:#4a3aa7; --grey:#b4b2a9;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
@media (prefers-color-scheme:dark){
  :root:where(:not([data-theme="light"])){
    color-scheme:dark;
    --page:#0d0d0d; --surface:#1a1a19; --card:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10); --wash:#1d2836;
    --blue:#3987e5; --blue-dk:#5598e7; --blue-lt:#256abf; --blue-xlt:#184f95;
    --orange:#d95926; --yellow:#c98500; --aqua:#199e70; --red:#e66767; --violet:#9085e9; --grey:#5d5c58;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --page:#0d0d0d; --surface:#1a1a19; --card:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10); --wash:#1d2836;
  --blue:#3987e5; --blue-dk:#5598e7; --blue-lt:#256abf; --blue-xlt:#184f95;
  --orange:#d95926; --yellow:#c98500; --aqua:#199e70; --red:#e66767; --violet:#9085e9; --grey:#5d5c58;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:64px}
body{margin:0;background:var(--page);color:var(--ink);font:16px/1.55 var(--sans);-webkit-text-size-adjust:100%}
a{color:var(--blue)}
.top{position:sticky;top:0;z-index:20;background:color-mix(in srgb,var(--page) 92%,transparent);backdrop-filter:blur(8px);border-bottom:1px solid var(--border)}
.top .in{max-width:1080px;margin:0 auto;padding:0 16px;display:flex;align-items:center;gap:16px;height:52px}
.brand{font-weight:700;letter-spacing:-.01em;white-space:nowrap}
.nav{display:flex;gap:2px;overflow-x:auto;scrollbar-width:none;-webkit-overflow-scrolling:touch}
.nav::-webkit-scrollbar{display:none}
.nav a{font-size:14px;color:var(--ink2);text-decoration:none;padding:6px 10px;border-radius:8px;white-space:nowrap}
.nav a:hover{background:var(--wash);color:var(--ink)}
.nav a.on{background:var(--wash);color:var(--blue-dk)}
.wrap{max-width:1080px;margin:0 auto;padding:0 16px 64px}
header.hero{padding:40px 0 8px}
.kicker{font-size:13px;font-weight:600;color:var(--blue-dk);text-transform:uppercase;letter-spacing:.06em;margin:0 0 8px}
h1{font-size:clamp(28px,6vw,44px);line-height:1.1;letter-spacing:-.025em;margin:0 0 14px;max-width:760px}
.lede{font-size:clamp(16px,2.4vw,19px);color:var(--ink2);max-width:720px;margin:0 0 22px}
.links{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 24px}
.links a{display:inline-block;padding:9px 14px;border-radius:999px;border:1px solid var(--border);background:var(--card);color:var(--ink);text-decoration:none;font-size:14px;font-weight:600}
.links a.primary{background:var(--blue-dk);border-color:var(--blue-dk);color:#fff}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:0 0 8px}
.stat{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px 16px}
.stat .n{font-size:26px;font-weight:700;letter-spacing:-.02em;line-height:1.1}
.stat .l{font-size:13px;color:var(--ink2);margin-top:4px}
section{padding-top:40px}
section>h2{font-size:clamp(22px,4vw,28px);letter-spacing:-.02em;margin:0 0 6px}
section>.intro{color:var(--ink2);max-width:760px;margin:0 0 18px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px 18px 12px;margin:0 0 14px;min-width:0}
.card h3{font-size:17px;margin:0 0 4px;letter-spacing:-.01em}
.card .d{font-size:14px;color:var(--ink2);margin:0 0 10px;max-width:760px}
.take{font-size:15px;line-height:1.5;border-left:3px solid var(--blue);padding:4px 0 4px 12px;margin:10px 0 8px}
.src{font-size:12px;color:var(--muted);margin:6px 0 2px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:820px){.grid2{grid-template-columns:1fr}}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:13px;color:var(--ink2);margin:2px 0 6px}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.chart{width:100%}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.big{font-size:clamp(34px,7vw,48px);font-weight:700;letter-spacing:-.03em;line-height:1}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}
.steps{list-style:none;padding:0;margin:0;display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;counter-reset:s}
.steps li{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:16px;counter-increment:s}
.steps li::before{content:counter(s);display:inline-grid;place-items:center;width:26px;height:26px;border-radius:50%;background:var(--wash);color:var(--blue-dk);font-weight:700;font-size:14px;margin-bottom:8px}
.steps b{display:block;margin-bottom:4px}
.steps span{font-size:14px;color:var(--ink2)}
table{border-collapse:collapse;width:100%;font-size:14px}
th{font-size:12px;color:var(--muted);font-weight:600;text-align:right;padding:8px;border-bottom:1px solid var(--grid);white-space:nowrap}
td{padding:7px 8px;border-bottom:1px solid var(--grid);text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
th.l,td.l{text-align:left}
details{margin:6px 0 4px}
summary{cursor:pointer;font-size:14px;color:var(--blue);padding:4px 0}
.toggle{display:inline-flex;border:1px solid var(--border);border-radius:999px;padding:2px;margin:0 0 8px;flex-wrap:wrap}
.toggle button{font:600 13px var(--sans);border:0;background:none;color:var(--ink2);padding:6px 12px;border-radius:999px;cursor:pointer}
.toggle button.on{background:var(--wash);color:var(--blue-dk)}
footer{color:var(--muted);font-size:13px;line-height:1.7;margin-top:48px;border-top:1px solid var(--border);padding-top:18px}
.theme{margin-left:auto;font:600 13px var(--sans);border:1px solid var(--border);background:var(--card);color:var(--ink2);border-radius:999px;padding:5px 10px;cursor:pointer;white-space:nowrap}
@media(max-width:560px){.top .brand{display:none}.nav a{padding:6px 7px;font-size:13px}.top .in{gap:8px}.card{padding:14px 12px 8px}.stat .n{font-size:22px}}
</style></head><body>

<div class="top"><div class="in">
  <span class="brand">Cultural Riddles</span>
  <nav class="nav" id="nav">
    <a href="#results">Results</a><a href="#cultures">Cultures</a><a href="#answers">Answers</a>
    <a href="#helps">What helps</a><a href="#built">How it's built</a>
  </nav>
  <button class="theme" id="theme" aria-label="Switch colour theme">Dark</button>
</div></div>

<div class="wrap">
<header class="hero">
  <p class="kicker">Cohere Labs Community</p>
  <h1>Can AI solve riddles from your culture?</h1>
  <p class="lede" id="lede"></p>
  <div class="links" id="links"></div>
  <div class="stats" id="stats"></div>
</header>

<section id="results"></section>
<section id="cultures"></section>
<section id="answers"></section>
<section id="helps"></section>
<section id="built"></section>
<footer id="foot"></footer>
</div>

<script>
const D = __DATA__;
// ---------- helpers ----------
const F = n => Number(n).toLocaleString("en-US");
const p1 = n => Number(n).toFixed(1) + "%";
const p0 = n => Math.round(Number(n)) + "%";
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const $ = id => document.getElementById(id);
const narrow = () => window.innerWidth < 560;
const LB = D.leaderboard;
const LBM = Object.fromEntries(LB.map(r => [r.model, r]));
const LABEL = {}; D.communities.forEach(r => LABEL[r.community] = r.label); D.coverage.forEach(r => LABEL[r.community] = r.label);
const RES_ORDER = ["high", "medium", "low", "extremely-low"];
const RES_LABEL = {"high": "High", "medium": "Medium", "low": "Low", "extremely-low": "Extremely low"};
const resColor = k => ({"high": css("--blue-dk"), "medium": css("--blue-lt"), "low": css("--orange"), "extremely-low": css("--red")})[k] || css("--grey");
const R = D.routing, NRESP = R.responses;

function base(extra = {}) {
  const ink = css("--ink"), muted = css("--muted"), grid = css("--grid");
  return Object.assign({
    font: {family: css("--sans"), size: narrow() ? 11 : 12, color: ink},
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    margin: {l: 8, r: 12, t: 6, b: 36},
    xaxis: {gridcolor: grid, zeroline: false, automargin: true, linecolor: css("--axis"), tickfont: {color: muted}},
    yaxis: {gridcolor: grid, zeroline: false, automargin: true, tickfont: {color: ink}, ticklabelstandoff: 8},
    hoverlabel: {bgcolor: css("--card"), bordercolor: css("--axis"), font: {color: ink, family: css("--sans")}},
    showlegend: false, bargap: 0.25,
  }, extra);
}
const CFG = {displayModeBar: false, responsive: true};
const X = (o = {}) => Object.assign({}, base().xaxis, o);
const Y = (o = {}) => Object.assign({}, base().yaxis, o);
function plot(id, h, traces, layout) { const e = $(id); Plotly.purge(e); e.style.height = h + "px"; Plotly.newPlot(e, traces, layout, CFG); }
const card = (id, title, desc, take, extra = "", src = "") =>
  `<div class="card"><h3>${title}</h3>${desc ? `<p class="d">${desc}</p>` : ""}${extra}<div id="${id}" class="chart"></div>${take ? `<p class="take">${take}</p>` : ""}${src ? `<p class="src">${src}</p>` : ""}</div>`;
const legend = items => `<div class="legend">${items.map(([l, c]) => `<span><span class="sw" style="background:${c}"></span>${l}</span>`).join("")}</div>`;
const the = c => (/^(United|Netherlands)/.test(c) ? "the " + c : c);
const shortModel = m => narrow() ? m.replace(" Preview", "").replace(" Instruct", "").replace(" Thinking", " Think.") : m;

// ---------- header ----------
function renderHeader() {
  const C = D.corpus;
  $("lede").innerHTML = `Native speakers from ${C.communities} communities wrote ${F(C.riddles)} original riddles in their own languages,
    using local nicknames, history, food and everyday life that people at home would recognise. We evaluated ${LB.length} AI model
    configurations; valid native-speaker labels cover ${p0(100 * R.human / NRESP)} of the ${F(NRESP)} recorded answers.`;
  const st = [[F(C.riddles), "original riddles"], [C.communities, "communities"], [C.languages, "languages"],
    [C.countries, "countries"], [LB.length, "model configurations"], [F(NRESP), "model answers"]];
  $("stats").innerHTML = st.map(([n, l]) => `<div class="stat"><div class="n">${n}</div><div class="l">${l}</div></div>`).join("");
  const L = D.links || {}, names = {demo: "Try the riddles", quiz: "Take the quiz", paper: "Read the paper", dataset: "Get the dataset"};
  $("links").innerHTML = Object.keys(names).filter(k => L[k]).map((k, i) =>
    `<a class="${i ? "" : "primary"}" href="${L[k]}" target="_blank" rel="noopener">${names[k]}</a>`).join("");
}

// ---------- Results ----------
function renderResults() {
  const best = LB[0], over40 = LB.filter(r => r.human_accuracy > 40).length;
  const ratio = LB.filter(r => r.exact_match > 0).map(r => r.difference_pp / r.exact_match * 100);
  $("results").innerHTML = `<h2>Results</h2>
    <p class="intro">The share of scored answers judged fully correct by rules, valid native-speaker labels, or agreeing LLM judges, for each model configuration,
    across the ${D.n_human_evaluated_communities} communities with human evaluation.</p>` +
    card("lb", "Leaderboard",
      "Light blue is what an exact string match with the reference answer already counts. Dark blue is the extra correct answers found by native speakers and the LLM judges: other valid names, spellings and phrasings.",
      `<b>${best.model}</b> leads at ${p1(best.human_accuracy)}. Only ${over40} of ${LB.length} configurations get more than 40% right.
       For configurations with exact matches, review adds ${Math.round(Math.min(...ratio))}% to ${Math.round(Math.max(...ratio))}% on top of exact matching.`,
      legend([["exact match", css("--blue-lt")], ["added by review", css("--blue-dk")]]),
      "Source: exported model-performance and human-evaluation summaries.") +
    card("rank", "Exact matching changes the ranking",
      "Each line is one model: its rank when only exact matches count (left) and its rank after native speakers and the LLM judges found the other correct answers (right). Coloured lines change rank.",
      rankTake(), legend([["moves up", css("--blue-dk")], ["moves down", css("--orange")], ["same rank", css("--grey")]]),
      "Source: exported leaderboard exact-match and reviewed-accuracy ranks.");
  const rows = [...LB].reverse();
  plot("lb", rows.length * (narrow() ? 21 : 23) + 50, [
    {type: "bar", orientation: "h", y: rows.map(r => shortModel(r.model)), x: rows.map(r => r.exact_match), marker: {color: css("--blue-lt")},
     hovertemplate: "%{y}<br>exact match %{x:.1f}%<extra></extra>"},
    {type: "bar", orientation: "h", y: rows.map(r => shortModel(r.model)), x: rows.map(r => r.difference_pp), marker: {color: css("--blue-dk")},
     text: rows.map(r => p1(r.human_accuracy)), textposition: "outside", cliponaxis: false, textfont: {color: css("--ink")},
     customdata: rows.map(r => r.human_accuracy), hovertemplate: "%{y}<br>added by review %{x:.1f} points<br>total %{customdata:.1f}%<extra></extra>"}],
    base({barmode: "stack", margin: {l: 8, r: 44, t: 4, b: 30}, xaxis: X({range: [0, 100], ticksuffix: "%"})}));
  drawRanks();
}
function exactRanks() {
  const r = {}; [...LB].sort((a, b) => b.exact_match - a.exact_match).forEach((x, i) => r[x.model] = i + 1); return r;
}
function rankTake() {
  const er = exactRanks(), moved = LB.filter(x => er[x.model] !== x.rank);
  const up = [...moved].sort((a, b) => (er[b.model] - b.rank) - (er[a.model] - a.rank))[0];
  const c = LBM[D.pair_models.claude_reasoning[1]];
  return `${moved.length} of ${LB.length} configurations change rank once the other correct answers are counted.
    ${c.model} is ${ord(er[c.model])} by exact match but ${ord(c.rank)} after review` +
    (up && up.model !== c.model ? `, and ${up.model} moves from ${ord(er[up.model])} to ${ord(up.rank)}.` : ".");
}
const ord = n => n + (n % 100 >= 11 && n % 100 <= 13 ? "th" : ({1: "st", 2: "nd", 3: "rd"})[n % 10] || "th");
function drawRanks() {
  const er = exactRanks(), n = LB.length, nm = narrow();
  const col = x => er[x.model] > x.rank ? css("--blue-dk") : er[x.model] < x.rank ? css("--orange") : css("--grey");
  const still = LB.filter(x => er[x.model] === x.rank), moved = LB.filter(x => er[x.model] !== x.rank);
  const line = x => ({type: "scatter", mode: "lines+markers", x: [0, 1], y: [er[x.model], x.rank],
    line: {color: col(x), width: er[x.model] === x.rank ? 1.5 : 2.5}, marker: {size: 7, color: col(x), line: {color: css("--card"), width: 1.5}},
    customdata: [[x.model, x.exact_match], [x.model, x.human_accuracy]],
    hovertemplate: "%{customdata[0]}<br>rank %{y}, %{customdata[1]:.1f}%<extra></extra>"});
  const labels = {type: "scatter", mode: "text", x: LB.map(() => 1.04), y: LB.map(x => x.rank), text: LB.map(x => shortModel(x.model)),
    textposition: "middle right", textfont: {size: nm ? 10 : 11, color: LB.map(x => er[x.model] === x.rank ? css("--ink2") : css("--ink"))}, hoverinfo: "skip", cliponaxis: false};
  plot("rank", n * (nm ? 19 : 21) + 60, [...still.map(line), ...moved.map(line), labels],
    base({margin: {l: 30, r: nm ? 150 : 190, t: 26, b: 10},
      xaxis: X({range: [-0.05, 1.05], tickvals: [0, 1], ticktext: ["exact match", "after review"], side: "top", showgrid: false, linecolor: "rgba(0,0,0,0)", tickfont: {color: css("--muted"), size: 12}}),
      yaxis: Y({autorange: "reversed", range: [n + 0.5, 0.5], tickvals: [1, 5, 10, 15, 20, 25], showgrid: false, tickfont: {color: css("--muted")}, title: {text: ""}})}));
}

// ---------- Cultures ----------
let cultureView = "all";
function renderCultures() {
  const com = D.communities, hard = com.slice(0, 8), easy = com.slice(-8);
  const lead = LB[0], lowres = hard.filter(r => ["low", "extremely-low"].includes(r.language_resourcedness)).length;
  const g = [...LB].sort((a, b) => (b.max_community_accuracy - b.min_community_accuracy) - (a.max_community_accuracy - a.min_community_accuracy));
  const ct = [...D.countries].sort((a, b) => a.accuracy - b.accuracy);
  const sr = [...D.subregion].sort((a, b) => a.pooled_accuracy - b.pooled_accuracy);
  $("cultures").innerHTML = `<h2>Cultures</h2>
    <p class="intro">Average scores hide large differences between communities. The same model can solve most riddles from one culture and almost none from another.</p>` +
    card("com", "Hardest and easiest communities",
      `Each bar is one community: the average accuracy of the ${LB.length} model configurations on its riddles. Colour shows how much web text exists in the language.`,
      `The eight hardest communities average ${p0(hard[0].mean_accuracy)} to ${p0(hard[7].mean_accuracy)}, and ${lowres} of them write in low or extremely low-resource languages.
       The eight easiest average ${p0(easy[0].mean_accuracy)} to ${p0(easy[7].mean_accuracy)}.`,
      `<div class="toggle" id="comtog"><button data-v="all" class="${cultureView === "all" ? "on" : ""}">All ${com.length}</button><button data-v="ends" class="${cultureView === "ends" ? "on" : ""}">Top and bottom 8</button></div>` +
      legend(RES_ORDER.map(k => [RES_LABEL[k] + " resource", resColor(k)])),
      "Source: exported community accuracy and language-resource metadata; resource level uses Common Crawl page counts.") +
    card("gap", "One model, many cultures",
      "For each model, its lowest- and highest-scoring community. A long line means its accuracy depends heavily on whose culture the riddle comes from.",
      `Even the top model, ${lead.model}, goes from <b>${p0(lead.max_community_accuracy)}</b> on ${LABEL[lead.max_accuracy_community]}
       to <b>${p0(lead.min_community_accuracy)}</b> on ${LABEL[lead.min_accuracy_community]}.`,
      legend([["lowest community", css("--red")], ["highest community", css("--blue-dk")]])) +
    card("grid", "Every model in every community",
      `Each square is one model in one community. Rows are models, best at the top. Columns are communities, hardest on the left. Pale columns are cultures that almost every model misses.`,
      gridTake(), `<p class="d" style="margin:0 0 6px">${narrow() ? "Scroll sideways to see all communities. " : ""}Hover or tap a square for details.</p>`,
      "Source: exported model-by-community accuracy matrix.") +
    card("fs", "Frontier models against small models",
      `Each dot is one community. Further right: frontier models do better there. Higher up: small models do better there.`,
      `Frontier models reach up to ${p0(Math.max(...com.map(r => r.frontier_accuracy)))} in a community; the small models never get past ${p0(Math.max(...com.map(r => r.small_accuracy)))}.`,
      legend(RES_ORDER.map(k => [RES_LABEL[k], resColor(k)])),
      `Frontier: average of ${D.frontier_models.join(", ")}. Small: average of ${D.small_models.join(", ")}.`) +
    `<div class="grid2">` +
    card("map", "By country",
      `Accuracy pooled over all models and every community in the country. Grey countries are not in the results.`,
      `Across ${ct.length} countries, accuracy ranges from <b>${p0(ct[0].accuracy)}</b> in ${the(ct[0].country)} to <b>${p0(ct[ct.length - 1].accuracy)}</b> in ${the(ct[ct.length - 1].country)}.`,
      "", `Covers the ${D.eligible_communities} communities with at least 600 human labels (leaves out ${D.excluded_from_groups.map(c => LABEL[c]).join(" and ")}).`) +
    card("reg", "By region",
      "Pooled accuracy by UN geographic subregion, with 95% intervals from resampling communities. Hover for the number of communities.",
      `${sr[sr.length - 1].group} is highest at ${p0(sr[sr.length - 1].pooled_accuracy)}; ${sr[0].group} is lowest at ${p0(sr[0].pooled_accuracy)}.`,
      "", `Same ${D.eligible_communities} communities as the map. Regions with one community have no interval.`) +
    `</div>`;
  { const g = $("grid"), w = document.createElement("div"); w.className = "scroll"; g.parentNode.insertBefore(w, g); w.appendChild(g); }
  document.querySelectorAll("#comtog button").forEach(b => b.onclick = () => {
    cultureView = b.dataset.v; document.querySelectorAll("#comtog button").forEach(x => x.classList.toggle("on", x === b)); drawCommunities();
  });
  drawCommunities();
  const gr = [...g].reverse();
  plot("gap", gr.length * (narrow() ? 21 : 23) + 50, [
    {type: "scatter", mode: "lines", x: gr.flatMap(r => [r.min_community_accuracy, r.max_community_accuracy, null]), y: gr.flatMap(r => [shortModel(r.model), shortModel(r.model), null]),
     line: {color: css("--grid"), width: 3}, hoverinfo: "skip"},
    {type: "scatter", mode: "markers", x: gr.map(r => r.min_community_accuracy), y: gr.map(r => shortModel(r.model)), marker: {color: css("--red"), size: 9, line: {color: css("--card"), width: 2}},
     text: gr.map(r => LABEL[r.min_accuracy_community]), hovertemplate: "%{y}<br>lowest: %{text}, %{x:.0f}%<extra></extra>"},
    {type: "scatter", mode: "markers", x: gr.map(r => r.max_community_accuracy), y: gr.map(r => shortModel(r.model)), marker: {color: css("--blue-dk"), size: 9, line: {color: css("--card"), width: 2}},
     text: gr.map(r => LABEL[r.max_accuracy_community]), hovertemplate: "%{y}<br>highest: %{text}, %{x:.0f}%<extra></extra>"}],
    base({xaxis: X({range: [-2, 102], ticksuffix: "%"})}));
  drawGrid();
  plot("fs", narrow() ? 320 : 380, [{type: "scatter", mode: "markers", x: com.map(r => r.frontier_accuracy), y: com.map(r => r.small_accuracy), text: com.map(r => r.label),
    marker: {size: 10, color: com.map(r => resColor(r.language_resourcedness)), line: {color: css("--card"), width: 1.5}},
    hovertemplate: "%{text}<br>frontier %{x:.1f}%<br>small %{y:.1f}%<extra></extra>"}],
    base({margin: {l: 8, r: 12, t: 6, b: 44}, xaxis: X({range: [0, 100], ticksuffix: "%", title: {text: "frontier models (average of 4)", font: {size: 12, color: css("--muted")}}}),
      yaxis: Y({range: [0, Math.ceil(Math.max(...com.map(r => r.small_accuracy)) / 10) * 10 + 5], ticksuffix: "%", title: {text: "small models (average of 5)", font: {size: 12, color: css("--muted")}}, tickfont: {color: css("--muted")}})}));
  plot("map", narrow() ? 260 : 330, [{type: "choropleth", locationmode: "country names", locations: D.countries.map(r => r.country), z: D.countries.map(r => r.accuracy),
    zmin: 0, zmax: 60, colorscale: [[0, css("--blue-xlt")], [1, css("--blue-dk")]],
    customdata: D.countries.map(r => r.communities + (r.communities === 1 ? " community" : " communities")),
    marker: {line: {color: css("--card"), width: .5}}, colorbar: {ticksuffix: "%", thickness: 8, len: .7, outlinewidth: 0, tickfont: {color: css("--muted")}},
    hovertemplate: "%{location}<br>%{z:.1f}% correct (%{customdata})<extra></extra>"}],
    base({margin: {l: 0, r: 0, t: 0, b: 0}, geo: {showframe: false, showcoastlines: false, projection: {type: "natural earth"}, showcountries: true,
      countrycolor: css("--card"), landcolor: css("--grid"), bgcolor: "rgba(0,0,0,0)", lataxis: {range: [-50, 75]}}}));
  plot("reg", sr.length * 26 + 50, [{type: "bar", orientation: "h", y: sr.map(r => r.group), x: sr.map(r => r.pooled_accuracy), marker: {color: css("--blue")},
    error_x: {type: "data", symmetric: false, array: sr.map(r => r.community_ci_high == null ? 0 : r.community_ci_high - r.pooled_accuracy),
      arrayminus: sr.map(r => r.community_ci_low == null ? 0 : r.pooled_accuracy - r.community_ci_low), color: css("--muted"), thickness: 1.2, width: 3},
    customdata: sr.map(r => r.communities), hovertemplate: "%{y}: %{x:.1f}% (%{customdata} communities)<extra></extra>"}],
    base({xaxis: X({range: [0, 70], ticksuffix: "%"})}));
}
function gridTake() {
  const cells = D.cells, hard = D.communities.slice(0, 8).map(r => r.community);
  const top5 = LB.slice(0, 5).map(r => r.model);
  const miss = cells.filter(c => top5.includes(c.model_display) && hard.includes(c.community) && c.value < 25).length;
  return `Even in the top five models, ${miss} of the ${top5.length * hard.length} model-community pairs in the eight hardest communities score below 25%.`;
}
function drawGrid() {
  const comms = D.communities.map(r => r.community), models = LB.map(r => r.model);
  const v = {}, t = {}; D.cells.forEach(c => { v[c.model_display + "|" + c.community] = c.value; t[c.model_display + "|" + c.community] = c.total; });
  const e = $("grid"), nm = narrow();
  e.style.minWidth = nm ? "760px" : "";
  plot("grid", models.length * (nm ? 15 : 17) + 120, [{type: "heatmap", x: comms.map(c => LABEL[c]), y: models.map(shortModel),
    z: models.map(m => comms.map(c => v[m + "|" + c])), customdata: models.map(m => comms.map(c => t[m + "|" + c])),
    zmin: 0, zmax: 100, xgap: 1, ygap: 1, colorscale: [[0, css("--surface")], [0.02, css("--blue-xlt")], [1, css("--blue-dk")]],
    colorbar: {ticksuffix: "%", thickness: 8, len: .8, outlinewidth: 0, tickfont: {color: css("--muted")}},
    hovertemplate: "%{y}<br>%{x}<br>%{z:.1f}% of %{customdata} scored answers<extra></extra>"}],
    base({margin: {l: 8, r: 8, t: 4, b: 110}, xaxis: X({tickangle: -60, tickfont: {size: 9, color: css("--muted")}, showgrid: false, linecolor: "rgba(0,0,0,0)"}),
      yaxis: Y({autorange: "reversed", tickfont: {size: nm ? 9.5 : 10.5, color: css("--ink")}, showgrid: false, dtick: 1})}));
}
function drawCommunities() {
  let rows = D.communities;
  if (cultureView === "ends") rows = [...rows.slice(0, 8), ...rows.slice(-8)];
  plot("com", rows.length * (narrow() ? 17 : 16) + 40, [{type: "bar", orientation: "h", y: rows.map(r => r.label), x: rows.map(r => r.mean_accuracy),
    marker: {color: rows.map(r => resColor(r.language_resourcedness))}, customdata: rows.map(r => [RES_LABEL[r.language_resourcedness], r.n_below_10]),
    hovertemplate: `%{y}<br>%{x:.1f}% average<br>%{customdata[0]} resource<br>%{customdata[1]} of ${LB.length} models below 10%<extra></extra>`}],
    base({bargap: .18, margin: {l: 8, r: 12, t: 2, b: 30}, xaxis: X({ticksuffix: "%", range: [0, 70]}), yaxis: Y({tickfont: {size: narrow() ? 10 : 11, color: css("--ink")}})}));
}

// ---------- Answers ----------
let topicMode = "fixed";
function renderAnswers() {
  const et = D.error_types, ettot = et.reduce((a, b) => a + b.responses, 0);
  const halluc = et.find(r => r.code === "0.3"), wrongEntity = et.find(r => r.code === "0.2");
  const abst = 100 * R.admits_not_knowing / NRESP;
  const H = D.hallucination.map(r => Object.assign({}, r, {acc: LBM[r.model_display].human_accuracy}));
  const hiH = [...H].sort((a, b) => b.pct_hallucinated - a.pct_hallucinated)[0], loH = [...H].sort((a, b) => a.pct_hallucinated - b.pct_hallucinated)[0];
  const tp = Object.fromEntries(D.topic_pooled.map(r => [r.topic_group, r]));
  const fixed = D.topic_pooled.filter(r => r.topic_group !== "Wildcard"), fixedAll = fixed.reduce((a, b) => a + b.accuracy * b.responses, 0) / fixed.reduce((a, b) => a + b.responses, 0);
  const fx = [...fixed].sort((a, b) => b.accuracy - a.accuracy);
  const RS = D.riddle_solves, none = RS.hist[0] || 0, all = RS.hist[LB.length] || 0;
  const halfOrMore = Object.entries(RS.hist).filter(([k]) => +k >= LB.length / 2).reduce((a, [, v]) => a + v, 0);
  const AG = D.agreement, agN = Object.entries(AG.by_n).map(([k, v]) => [+k, v]);
  const sumAg = (f, st) => agN.filter(([k]) => f(k)).reduce((a, [, v]) => a + (v[st] || 0), 0);
  const strong = k => k >= AG.models / 2, strongAll = sumAg(strong, "correct") + sumAg(strong, "wrong") + sumAg(strong, "not scored");
  const strongWrong = sumAg(strong, "wrong");
  $("answers").innerHTML = `<h2>Answers</h2>
    <p class="intro">What the ${F(NRESP)} answers look like: how often models are right, partly right or wrong, and the kinds of mistakes they make.</p>` +
    card("dist", "How each model's answers were scored",
      "Every scored answer gets 0 (wrong), 1 (admits not knowing), 2 (partly correct) or 3 (correct). The two blues add up to the leaderboard score.",
      `Models almost never say they don't know. Only ${p1(abst)} of all answers admit it, although the prompt asks models to answer "I don't know" when unsure.`,
      legend([["correct, exact match", css("--blue-dk")], ["correct, found by review", css("--blue-lt")], ["partly correct", css("--yellow")], ["admits not knowing", css("--grey")], ["wrong", css("--red")]]),
      "Source: exported scoring and error summaries, restricted to human-evaluated communities.") +
    `<div class="grid2">` +
    card("et", "Kinds of mistakes",
      `Error types cover ${F(ettot)} rule-scored or native-speaker-reviewed answers. Deterministic rules retain their assigned outcome type; LLM-jury-only answers are excluded.`,
      `The most common type is a <b>hallucinated</b> answer, made up or unrelated to the riddle: ${p1(halluc.percent)} of this rule-and-human-scored population.
       Next is the right kind of thing but the wrong one (${p1(wrongEntity.percent)}).`,
      "", `Source: the shared human-error-type summary (${F(ettot)} rule-scored or native-speaker-reviewed answers).`) +
    card("hal", "Weaker models make things up more",
      "Each dot is a model: its accuracy against the share of its human-reviewed answers labelled hallucinated.",
      `${hiH.model_display} has the highest share (${p0(hiH.pct_hallucinated)}), ${loH.model_display} the lowest (${p0(loH.pct_hallucinated)}).`,
      "", "Human-reviewed answers only, so the share leaves out exact matches and answers scored by rules or judges.") +
    `</div>` +
    card("top", "Topics",
      "Every community wrote 10 riddles on each of eight shared topics, plus 20 on two topics of its own choosing (the wildcards).",
      `${fx[0].topic_group} has the highest shared-topic accuracy (${p0(fx[0].accuracy)}) and ${fx[fx.length - 1].topic_group} the lowest (${p0(fx[fx.length - 1].accuracy)}).
       The communities' wildcard topics score ${p0(tp.Wildcard.accuracy)} against ${p0(fixedAll)} on shared topics.`,
      `<div class="toggle" id="toptog"><button data-v="fixed" class="${topicMode === "fixed" ? "on" : ""}">Overall</button><button data-v="heat" class="${topicMode === "heat" ? "on" : ""}">By model</button></div>`,
      `Pooled over all models in the ${D.eligible_communities} communities that meet the shared human-label coverage threshold.`) +
    `<div class="grid2">` +
    card("solve", "How many models solve each riddle",
      `For the ${F(RS.riddles)} riddles where all ${LB.length} configurations have a scored answer: how many of them got it right.`,
      `<b>${F(none)}</b> riddles were solved by no model at all, and ${F(all)} by every model. ${p0(100 * halfOrMore / RS.riddles)} of riddles are solved by at least half of the models.`) +
    card("agr", "When models agree, are they right?",
      `For each of the ${F(AG.riddles)} riddles, how many of the ${AG.models} configurations gave the most common answer, and whether that answer was correct.`,
      `When at least half of the models give the same answer, it is usually right, but not always: on ${F(strongWrong)} of ${F(strongAll)} such riddles they agree on a wrong answer.`,
      legend([["most common answer correct", css("--blue-dk")], ["wrong", css("--red")], ["not scored", css("--grey")]])) +
    `</div>`;
  document.querySelectorAll("#toptog button").forEach(b => b.onclick = () => {
    topicMode = b.dataset.v; document.querySelectorAll("#toptog button").forEach(x => x.classList.toggle("on", x === b)); drawTopics();
  });
  const order = LB.map(r => r.model), dm = Object.fromEntries(D.score_dist.map(r => [r.model, r]));
  const rows = [...order].reverse().map(m => dm[m]);
  const parts = [["exact", "correct, exact match", "--blue-dk"], ["review3", "correct, found by review", "--blue-lt"], ["s2", "partly correct", "--yellow"], ["s1", "admits not knowing", "--grey"], ["s0", "wrong", "--red"]];
  plot("dist", rows.length * (narrow() ? 21 : 23) + 50, parts.map(([k, l, c]) => ({type: "bar", orientation: "h", y: rows.map(r => shortModel(r.model)), x: rows.map(r => r[k]),
    marker: {color: css(c), line: {color: css("--card"), width: 1}}, hovertemplate: "%{y}<br>" + l + ": %{x:.1f}%<extra></extra>"})),
    base({barmode: "stack", bargap: .2, xaxis: X({range: [0, 100], ticksuffix: "%"})}));
  const ETL = {"0.4": "Other, wrong (with note)", "2.4": "Other, partly correct (with note)", "3.3": "Other correct (with note)"};
  const ers = [...et].sort((a, b) => a.responses - b.responses).map(r => Object.assign({}, r, {error_type: ETL[r.code] || r.error_type}));
  const grp = {"0": "--red", "1": "--grey", "2": "--yellow", "3": "--blue-dk"};
  plot("et", ers.length * 24 + 40, [{type: "bar", orientation: "h", y: ers.map(r => r.error_type), x: ers.map(r => r.percent),
    marker: {color: ers.map(r => css(grp[r.score_group]))}, customdata: ers.map(r => r.responses),
    text: ers.map(r => p1(r.percent)), textposition: "outside", cliponaxis: false, textfont: {color: css("--ink2"), size: 11},
    hovertemplate: "%{y}<br>%{customdata:,} answers (%{x:.2f}%)<extra></extra>"}],
    base({margin: {l: 8, r: 46, t: 4, b: 30}, xaxis: X({ticksuffix: "%", range: [0, 36]}), yaxis: Y({tickfont: {size: 11, color: css("--ink")}})}));
  plot("hal", narrow() ? 300 : 360, [{type: "scatter", mode: "markers", x: H.map(r => r.acc), y: H.map(r => r.pct_hallucinated), text: H.map(r => r.model_display),
    customdata: H.map(r => r.human_labels), marker: {size: 10, color: css("--red"), line: {color: css("--card"), width: 1.5}},
    hovertemplate: "%{text}<br>accuracy %{x:.1f}%<br>hallucinated %{y:.1f}% of %{customdata:,} reviewed answers<extra></extra>"}],
    base({margin: {l: 8, r: 12, t: 6, b: 44}, xaxis: X({range: [0, 85], ticksuffix: "%", title: {text: "accuracy", font: {size: 12, color: css("--muted")}}}),
      yaxis: Y({range: [0, 80], ticksuffix: "%", tickfont: {color: css("--muted")}, title: {text: "hallucinated", font: {size: 12, color: css("--muted")}}})}));
  drawTopics();
  const hk = Array.from({length: LB.length + 1}, (_, i) => i);
  plot("solve", narrow() ? 260 : 300, [{type: "bar", x: hk, y: hk.map(k => RS.hist[k] || 0), marker: {color: hk.map(k => k === 0 ? css("--red") : css("--blue"))},
    hovertemplate: "%{x} of " + LB.length + " models right: %{y:,} riddles<extra></extra>"}],
    base({bargap: .12, margin: {l: 8, r: 8, t: 6, b: 44}, xaxis: X({title: {text: "models that got it right", font: {size: 12, color: css("--muted")}}, dtick: 4}),
      yaxis: Y({tickfont: {color: css("--muted")}})}));
  const ak = agN.map(([k]) => k);
  plot("agr", narrow() ? 260 : 300, [["correct", "--blue-dk"], ["wrong", "--red"], ["not scored", "--grey"]].map(([s, c]) => ({type: "bar", x: ak, y: agN.map(([, v]) => v[s] || 0),
    marker: {color: css(c)}, hovertemplate: "%{x} models agree: %{y:,} riddles, answer " + s + "<extra></extra>"})),
    base({barmode: "stack", bargap: .12, margin: {l: 8, r: 8, t: 6, b: 44}, xaxis: X({title: {text: "models giving the most common answer", font: {size: 12, color: css("--muted")}}, dtick: 4}),
      yaxis: Y({tickfont: {color: css("--muted")}})}));
}
function drawTopics() {
  const TOP = ["History", "National Heritage", "Geography", "Politics", "Sports", "Arts", "Everyday Life", "Kid's World", "Wildcard"];
  const lab = t => t === "Wildcard" ? "Wildcards" : t;
  if (topicMode === "fixed") {
    const tp = Object.fromEntries(D.topic_pooled.map(r => [r.topic_group, r]));
    const t = [...TOP].sort((a, b) => tp[a].accuracy - tp[b].accuracy);
    plot("top", t.length * 28 + 40, [{type: "bar", orientation: "h", y: t.map(lab), x: t.map(k => tp[k].accuracy),
      marker: {color: t.map(k => k === "Wildcard" ? css("--orange") : css("--blue"))}, text: t.map(k => p1(tp[k].accuracy)), textposition: "outside", cliponaxis: false,
      textfont: {color: css("--ink2"), size: 11}, customdata: t.map(k => tp[k].responses), hovertemplate: "%{y}: %{x:.1f}% of %{customdata:,} answers<extra></extra>"}],
      base({margin: {l: 8, r: 46, t: 4, b: 30}, xaxis: X({range: [0, 50], ticksuffix: "%"})}));
  } else {
    const m = Object.fromEntries(D.topic_model.map(r => [r.model_display + "|" + r.topic_group, r.accuracy]));
    const models = LB.map(r => r.model).reverse();
    plot("top", models.length * 20 + 90, [{type: "heatmap", x: TOP.map(t => narrow() ? ({"National Heritage": "Heritage", "Everyday Life": "Everyday", "Kid's World": "Kids", "Geography": "Geogr.", "Wildcard": "Wildcards"})[t] || t : lab(t)), y: models.map(shortModel), z: models.map(md => TOP.map(t => m[md + "|" + t])),
      zmin: 0, zmax: 100, colorscale: [[0, css("--surface")], [0.01, css("--blue-xlt")], [1, css("--blue-dk")]], xgap: 2, ygap: 2,
      colorbar: {ticksuffix: "%", thickness: 8, outlinewidth: 0, tickfont: {color: css("--muted")}},
      hovertemplate: "%{y}<br>%{x}: %{z:.1f}%<extra></extra>"}],
      base({margin: {l: 8, r: 8, t: 4, b: 70}, xaxis: X({tickangle: -40, gridcolor: "rgba(0,0,0,0)"}), yaxis: Y({tickfont: {size: 10.5, color: css("--ink")}, gridcolor: "rgba(0,0,0,0)", dtick: 1})}));
  }
}

// ---------- What helps, what hurts ----------
function renderHelps() {
  const P = D.pairs, rs = D.resource, eng = D.english.filter(r => r.community !== "all").sort((a, b) => a.paired_delta - b.paired_delta);
  const all = D.english.find(r => r.community === "all"), TE = Object.fromEntries(D.tier_effects.map(r => [r.tier, r]));
  const zh = [...D.chinese].sort((a, b) => a.accuracy - b.accuracy);
  const SL = D.search_losses;
  const small = LB.filter(r => { const b = parseFloat(String(r.parameters_b).split("/").pop()); return !isNaN(b) && b <= 8; });
  const model = key => LBM[D.model_displays[key]];
  const g = model("gemma-4-31b"), o = model("Olmo-3-32B-Think"), a70 = model("swiss-ai/Apertus-70B-Instruct-2509"), a15 = model("swiss-ai/Apertus-v1.5-70B");
  const localTerm = D.local_term_effect;
  const signed = n => (n >= 0 ? "+" : "") + n.toFixed(1);
  $("helps").innerHTML = `<h2>What helps, what hurts</h2>
    <p class="intro">Comparisons that separate model settings, language and culture. Each one compares like with like: the same model with a setting switched on or off, the same riddles in two languages, or the same language in different places.</p>` +
    card("pairs", "Model configuration comparisons",
      "Each pair compares two configurations of the same model. Unspecified reasoning settings are not assumed to mean that reasoning was disabled.",
      `${D.pair_models.claude_reasoning[1]} scores ${signed(P.claude_reasoning[1] - P.claude_reasoning[0])} points relative to ${D.pair_models.claude_reasoning[0]}.
       ${D.pair_models.deepseek_reasoning[1]} scores ${signed(P.deepseek_reasoning[1] - P.deepseek_reasoning[0])} points relative to ${D.pair_models.deepseek_reasoning[0]}.
       ${D.pair_models.gpt_search[1]} scores ${signed(P.gpt_search[1] - P.gpt_search[0])} points relative to ${D.pair_models.gpt_search[0]}.
       Among human-labelled answers that were correct without search but wrong with it, ${p0(SL.shares["0.2"] || 0)} are the right kind of thing but the wrong one, and ${p0(SL.shares["0.3"] || 0)} are hallucinated.`,
      legend([["baseline configuration", css("--grey")], ["comparison configuration", css("--blue-dk")]]),
      "Source: exported paired configuration accuracy and search-loss summaries.") +
    card("eng", "English translation comparison",
      `${D.english_models.length} model configurations (${D.english_models.join(", ")}) answered English translations from ${eng.length} communities. Riddles whose translation gives the answer away are left out.`,
      `Accuracy goes from <b>${p1(all.paired_original)}</b> to <b>${p1(all.paired_english)}</b> (${signed(all.paired_delta)} points, 95% interval ${all.paired_delta_low.toFixed(1)} to ${all.paired_delta_high.toFixed(1)}),
       on ${F(Math.round(all.responses / D.english_models.length))} riddles. The same configurations score ${p1(D.english_high_resource_accuracy)} on human-evaluated high-resource communities.`,
      legend([["original language", css("--orange")], ["English translation", css("--blue-dk")]]),
      "Source: the shared English-condition summary. English answers were scored by rules and the LLM judges only.") +
    `<div class="grid2">` +
    card("res", "Less web text in the language, fewer right answers",
      "Pooled accuracy by how many Common Crawl pages exist in each community's language, with 95% intervals from resampling communities.",
      `Accuracy falls from <b>${p1(rs[0].pooled_accuracy)}</b> for high-resource languages to <b>${p1(rs[3].pooled_accuracy)}</b> for extremely low-resource ones.
       Accounting for the model, riddle and topic, the odds of a right answer are about ${Math.round(1 / TE.medium.odds_ratio)}, ${Math.round(1 / TE.low.odds_ratio)} and ${Math.round(1 / TE["extremely-low"].odds_ratio)} times higher in high-resource languages than in medium, low and extremely low-resource ones.`,
      "", "Source: exported resource-level accuracy and fitted tier effects. High: 10M or more pages; medium: 100K to 10M; low: 10K to 100K; extremely low: under 10K or not detected.") +
    card("zh", "Same language, different culture",
      `${zh.length} communities write in Chinese. On the same ${zh[0].shared_models} models, their accuracy differs widely.`,
      `${zh[zh.length - 1].label} reaches ${p0(zh[zh.length - 1].accuracy * 100)} and ${zh[0].label} reaches ${p0(zh[0].accuracy * 100)}.
       The language is shared, but these communities have different scores.`,
      "", `Source: the shared natural-experiment summary. Averages over ${zh[0].shared_models} models scored in all ${zh.length} communities.`) +
    `</div>` +
    `<div class="grid2">` +
    card("size", "Bigger is usually better, but not always",
      "Accuracy against model size, for models with a published size (total parameters, log scale).",
      `Models of 8B parameters or fewer reach at most ${p1(Math.max(...small.map(r => r.human_accuracy)))}. ${g.model} scores ${p1(g.human_accuracy)}, compared with ${p1(o.human_accuracy)} for ${o.model}.
       The two larger Apertus 70B configurations score ${p1(a70.human_accuracy)} and ${p1(a15.human_accuracy)}.`,
      "", "Source: exported model-performance and parameter metadata. Proprietary models do not publish their size and are not shown.") +
    `<div class="card"><h3>Words with no English name</h3><p class="d">Some answers are local terms without a common English name. The fitted riddle-feature model compares their odds of a fully correct answer with the reference answer category, accounting for other modelled features.</p>
      <div class="big">${localTerm.odds_ratio.toFixed(2)}</div><p class="d" style="margin-top:6px">odds ratio for a right answer when the answer is a local term (95% interval ${localTerm.ci_low.toFixed(2)} to ${localTerm.ci_high.toFixed(2)}).</p>
      <p class="src">Source: the released aggregate riddle-feature effect summary.</p></div>` +
    `</div>`;
  const pr = [["claude_reasoning", P.claude_reasoning], ["deepseek_reasoning", P.deepseek_reasoning], ["gpt_search", P.gpt_search]].map(([key, values]) => [...D.pair_models[key], values]);
  const yl = pr.map(([m, s]) => `${m}<br><span style="font-size:11px">${s}</span>`);
  plot("pairs", 200, [
    {type: "scatter", mode: "lines", x: pr.flatMap(([, , v]) => [v[0], v[1], null]), y: yl.flatMap(y => [y, y, null]), line: {color: css("--grid"), width: 4}, hoverinfo: "skip"},
    {type: "scatter", mode: "markers", x: pr.map(([, , v]) => v[0]), y: yl, marker: {size: 12, color: css("--grey"), line: {color: css("--card"), width: 2}}, hovertemplate: "baseline: %{x:.1f}%<extra></extra>"},
    {type: "scatter", mode: "markers+text", x: pr.map(([, , v]) => v[1]), y: yl, marker: {size: 12, color: css("--blue-dk"), line: {color: css("--card"), width: 2}},
     text: pr.map(([, , v]) => (v[1] - v[0] > 0 ? "+" : "") + (v[1] - v[0]).toFixed(1)), textposition: pr.map(([, , v]) => v[1] > v[0] ? "middle right" : "middle left"),
     textfont: {color: css("--ink2"), size: 12}, hovertemplate: "comparison: %{x:.1f}%<extra></extra>"}],
    base({xaxis: X({range: [0, 100], ticksuffix: "%"})}));
  plot("eng", eng.length * 40 + 50, [
    {type: "scatter", mode: "lines", x: eng.flatMap(r => [r.paired_original, r.paired_english, null]), y: eng.flatMap(r => [r.label, r.label, null]), line: {color: css("--grid"), width: 4}, hoverinfo: "skip"},
    {type: "scatter", mode: "markers", x: eng.map(r => r.paired_original), y: eng.map(r => r.label), marker: {size: 12, color: css("--orange"), line: {color: css("--card"), width: 2}}, hovertemplate: "%{y}<br>original %{x:.1f}%<extra></extra>"},
    {type: "scatter", mode: "markers", x: eng.map(r => r.paired_english), y: eng.map(r => r.label), marker: {size: 12, color: css("--blue-dk"), line: {color: css("--card"), width: 2}},
     customdata: eng.map(r => r.paired_delta), hovertemplate: "%{y}<br>English %{x:.1f}% (%{customdata:+.1f} points)<extra></extra>"}],
    base({xaxis: X({range: [0, 60], ticksuffix: "%"})}));
  plot("res", 280, [{type: "bar", x: rs.map(r => RES_LABEL[r.group]), y: rs.map(r => r.pooled_accuracy), marker: {color: rs.map(r => resColor(r.group))},
    error_y: {type: "data", symmetric: false, array: rs.map(r => r.community_ci_high - r.pooled_accuracy), arrayminus: rs.map(r => r.pooled_accuracy - r.community_ci_low), color: css("--muted"), thickness: 1.2, width: 5},
    customdata: rs.map(r => [r.communities, r.scored_responses]), hovertemplate: "%{x}: %{y:.1f}%<br>%{customdata[0]} communities, %{customdata[1]:,} answers<extra></extra>"}],
    base({yaxis: Y({range: [0, 50], ticksuffix: "%", tickfont: {color: css("--muted")}})}));
  plot("zh", 210, [{type: "bar", orientation: "h", y: zh.map(r => r.label), x: zh.map(r => 100 * r.accuracy), marker: {color: css("--blue")},
    text: zh.map(r => p0(100 * r.accuracy)), textposition: "outside", cliponaxis: false, textfont: {color: css("--ink2")},
    hovertemplate: "%{y}: %{x:.1f}%<extra></extra>"}], base({margin: {l: 8, r: 40, t: 4, b: 30}, xaxis: X({range: [0, 60], ticksuffix: "%"})}));
  const sized = LB.filter(r => r.parameters_b !== "--").map(r => Object.assign({}, r, {size: parseFloat(String(r.parameters_b).split("/").pop())}));
  plot("size", narrow() ? 300 : 340, [{type: "scatter", mode: "markers", x: sized.map(r => r.size), y: sized.map(r => r.human_accuracy), text: sized.map(r => r.model),
    marker: {size: 10, color: sized.map(r => [g.model, o.model].includes(r.model) ? css("--orange") : css("--blue")), line: {color: css("--card"), width: 1.5}},
    customdata: sized.map(r => r.parameters_b), hovertemplate: "%{text}<br>%{customdata}B parameters<br>%{y:.1f}% correct<extra></extra>"}],
    base({margin: {l: 8, r: 12, t: 6, b: 44}, xaxis: X({type: "log", title: {text: "parameters (billions, log scale)", font: {size: 12, color: css("--muted")}}}),
      yaxis: Y({range: [0, 70], ticksuffix: "%", tickfont: {color: css("--muted")}})}));
}

// ---------- How it's built ----------
function renderBuilt() {
  const cv = [...D.coverage].sort((a, b) => b.human / b.answers - a.human / a.answers);
  const noHuman = cv.filter(r => !r.in_results);
  const jf = [...D.judge_fill].sort((a, b) => a.change_pp - b.change_pp), big = jf[0];
  const C = D.corpus;
  $("built").innerHTML = `<h2>How the benchmark is built</h2>
    <p class="intro">Cultural Riddles is written, checked and scored by the communities themselves, through the Cohere Labs Community.</p>
    <ol class="steps">
      <li><b>Write</b><span>Native speakers wrote ${F(C.riddles)} riddles from scratch, 100 per community: ten on each of eight shared topics and ten on each of two topics the community chose. No riddle was translated from another community.</span></li>
      <li><b>Check</b><span>At least one native speaker other than the author checked every riddle: it must be solvable, its clues clear and its answer correct.</span></li>
      <li><b>Answer</b><span>${LB.length} model configurations produced ${F(NRESP)} answers with the same English prompt that names the country and topic. The coverage chart shows which communities have scored responses.</span></li>
      <li><b>Score</b><span>Rules score the clear cases. Native speakers label the rest, and two LLM judges score answers that still have no valid human label. A judge score counts only when both judges agree.</span></li>
    </ol>` +
    card("route", "Who scored the answers",
      `How the ${F(NRESP)} answers were scored.`,
      `Native speakers labelled <b>${p1(100 * R.human / NRESP)}</b> of all answers. Rules scored ${p1(100 * R.rules / NRESP)}, including ${p1(100 * R.exact_match / NRESP)} exact matches.
       The judges agreed on ${p1(100 * R.judges / NRESP)}; the remaining ${p1(100 * R.unscored / NRESP)}, where the judges disagreed or gave no valid verdict, are left out of accuracy.`,
      legend([["rules", css("--blue-lt")], ["native speakers", css("--blue-dk")], ["LLM judges (both agree)", css("--orange")], ["not scored", css("--grey")]]),
      "Source: exported routing counts from the rebuilt scored responses.") +
    card("cov", "Community by community",
      "The share of each community's answers scored by rules, native speakers and the LLM judges. Communities are sorted by the share labelled by native speakers.",
      `${noHuman.length} communities (${noHuman.map(r => r.label).join(", ")}) have no valid human labels. Their answers were scored by rules and judges only, so they are left out of the community results.`,
      `<details><summary>Show all ${cv.length} communities</summary>${legend([["rules", css("--blue-lt")], ["native speakers", css("--blue-dk")], ["LLM judges", css("--orange")], ["not scored", css("--grey")]])}<div id="covwrap"></div></details>`) +
    card("jf", "What changed when the judges filled the gaps",
      "Accuracy before and after adding the answers on which both LLM judges agree. Before, those answers were left out. None of them matched the reference exactly, so leaving them out made some models look better than they are.",
      `The biggest change is ${big.model_display}, ${big.change_pp.toFixed(1)} points. ${jf.filter(r => Math.abs(r.change_pp) < 0.5).length} of ${jf.length} configurations move by less than half a point.`,
      legend([["before", css("--grey")], ["after", css("--blue-dk")]]),
      "Source: the analysis pipeline's judge_fill_effect table, recomputed from the human-evaluation data.") +
    `<div class="card"><h3>All model configurations</h3><p class="d">Access: P proprietary, OW open-weight, OS open-source. Parameters in billions, active/total for mixture-of-experts models.</p>
     <div class="scroll"><table><thead><tr><th>#</th><th class="l">Model</th><th class="l">Access</th><th>Params (B)</th><th class="l">Reasoning</th><th>Total</th><th>Exact</th><th>Review</th></tr></thead>
     <tbody>${LB.map(r => `<tr><td>${r.rank}</td><td class="l">${r.model}</td><td class="l">${r.access_abbreviation}</td><td>${r.parameters_b === "--" ? "-" : r.parameters_b}</td>
       <td class="l">${({"N/A": "N/A", "default": "Default", "none": "None", "enabled": "Enabled", "adaptive": "Adaptive", "high": "High"})[r.reasoning] || r.reasoning}</td><td><b>${r.human_accuracy.toFixed(1)}</b></td><td>${r.exact_match.toFixed(1)}</td><td>${r.difference_pp.toFixed(1)}</td></tr>`).join("")}</tbody></table></div>
     <p class="src">Source: exported model-performance summary and canonical model mapping.</p></div>`;
  const parts = [["rules", "rules", "--blue-lt"], ["human", "native speakers", "--blue-dk"], ["judges", "LLM judges", "--orange"], ["unscored", "not scored", "--grey"]];
  plot("route", 90, parts.map(([k, l, c]) => ({type: "bar", orientation: "h", y: [""], x: [100 * R[k] / NRESP], marker: {color: css(c), line: {color: css("--card"), width: 2}},
    text: [100 * R[k] / NRESP > 8 ? p0(100 * R[k] / NRESP) : ""], textposition: "inside", insidetextanchor: "middle", textfont: {color: k === "human" ? "#fff" : css("--ink")},
    hovertemplate: l + ": " + F(R[k]) + " answers (%{x:.2f}%)<extra></extra>"})),
    base({barmode: "stack", margin: {l: 4, r: 4, t: 4, b: 26}, xaxis: X({range: [0, 100], ticksuffix: "%"}), yaxis: {visible: false}}));
  let covDrawn = false;
  document.querySelector("#built details").addEventListener("toggle", e => {
    if (!e.target.open || covDrawn) return; covDrawn = true;
    $("covwrap").innerHTML = `<div id="covplot"></div>`;
    const rows = [...cv].reverse();
    plot("covplot", rows.length * 16 + 40, parts.map(([k, l, c]) => ({type: "bar", orientation: "h", name: l, y: rows.map(r => r.label), x: rows.map(r => 100 * r[k] / r.answers),
      marker: {color: css(c)}, hovertemplate: "%{y}<br>" + l + ": %{x:.1f}%<extra></extra>"})),
      base({barmode: "stack", bargap: .18, xaxis: X({range: [0, 100], ticksuffix: "%"}), yaxis: Y({tickfont: {size: narrow() ? 10 : 11, color: css("--ink")}})}));
  });
  const jr = [...D.judge_fill].sort((a, b) => a.accuracy_after - b.accuracy_after);
  plot("jf", jr.length * (narrow() ? 21 : 23) + 50, [
    {type: "scatter", mode: "lines", x: jr.flatMap(r => [r.accuracy_before, r.accuracy_after, null]), y: jr.flatMap(r => [shortModel(r.model_display), shortModel(r.model_display), null]), line: {color: css("--grid"), width: 3}, hoverinfo: "skip"},
    {type: "scatter", mode: "markers", x: jr.map(r => r.accuracy_before), y: jr.map(r => shortModel(r.model_display)), marker: {color: css("--grey"), size: 9}, hovertemplate: "%{y}<br>before %{x:.1f}%<extra></extra>"},
    {type: "scatter", mode: "markers", x: jr.map(r => r.accuracy_after), y: jr.map(r => shortModel(r.model_display)), marker: {color: css("--blue-dk"), size: 9},
     customdata: jr.map(r => r.change_pp), hovertemplate: "%{y}<br>after %{x:.1f}% (%{customdata:+.1f} points)<extra></extra>"}],
    base({xaxis: X({range: [0, 100], ticksuffix: "%"})}));
}

// ---------- footer, nav, theme ----------
function renderFooter() {
  $("foot").innerHTML = `Cultural Riddles was written, checked and scored by native-speaker volunteers in the Cohere Labs Community.
    Accuracy is the share of scored answers rated fully correct (3 on a 0 to 3 scale). Unless a chart says otherwise, results cover the
    ${D.n_human_evaluated_communities} communities with human evaluation and ${LB.length} model configurations. Every number on this page is computed by the
    project's shared analysis loaders and checked for consistency across the exported views before each build.`;
}
function renderAll() { renderHeader(); renderResults(); renderCultures(); renderAnswers(); renderHelps(); renderBuilt(); renderFooter(); }
function setThemeLabel() {
  const t = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  $("theme").textContent = t === "dark" ? "Light" : "Dark";
}
$("theme").onclick = () => {
  const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  document.documentElement.dataset.theme = cur === "dark" ? "light" : "dark";
  try { localStorage.setItem("cr-theme", document.documentElement.dataset.theme); } catch (e) {}
  setThemeLabel(); renderAll();
};
try { const s = localStorage.getItem("cr-theme"); if (s) document.documentElement.dataset.theme = s; } catch (e) {}
setThemeLabel();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (!document.documentElement.dataset.theme) { setThemeLabel(); renderAll(); } });
let lastNarrow = narrow();
window.addEventListener("resize", () => { if (narrow() !== lastNarrow) { lastNarrow = narrow(); renderAll(); } });
const secs = ["results", "cultures", "answers", "helps", "built"];
const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) document.querySelectorAll("#nav a").forEach(a => a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id)); }),
  {rootMargin: "-40% 0px -55% 0px"});
secs.forEach(s => io.observe($(s)));
renderAll();
</script></body></html>
"""

if __name__ == "__main__":
    main()
