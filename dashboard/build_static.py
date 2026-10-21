"""Build the static Cultural Riddles dashboard (index.html).

Reads ``data/derived/dashboard/dashboard_data.json``, validates relationships
between the exported views, and bakes the data into ``template.html`` to make
one page (Plotly and fonts from a CDN). Building requires only Python's
standard library; serving the result requires no Python process or server.

    uv run python -m dashboard.build_static

Links show up on the page only when set below.
"""

from __future__ import annotations

import json
from pathlib import Path

from dashboard import compute as C

HERE = Path(__file__).resolve().parent
OUT = HERE / "index.html"
TEMPLATE = HERE / "template.html"

LINKS = {
    "paper": "",
    "dataset": "https://huggingface.co/datasets/CohereLabsCommunity/CulturalRiddles",
    "code": "https://github.com/Cohere-Labs-Community/CulturalRiddles",
}


def main() -> None:
    data = C.load()
    C.verify(data)
    data["links"] = {k: v for k, v in LINKS.items() if v}
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.read_text(encoding="utf-8")
    OUT.write_text(page.replace("__DATA__", payload), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
