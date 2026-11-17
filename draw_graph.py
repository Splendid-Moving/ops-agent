"""
Draws the agent's graph and saves it as a picture (docs/graph.png) plus a page that shows it (docs/graph.md).

The compiled LangGraph graph describes itself, so the picture always matches the real wiring
in agent/graph.py. The PNG is rendered by the free mermaid.ink service (only node names are sent).
Re-run after changing the graph: `uv run python draw_graph.py`.
"""

import logging
from pathlib import Path

logging.disable(logging.INFO)

from agent.graph import graph  # noqa: E402

DOCS_FOLDER = Path(__file__).parent / "docs"
IMAGE_FILE = DOCS_FOLDER / "graph.png"
PAGE_FILE = DOCS_FOLDER / "graph.md"

HEADER = """# Ops agent graph

Generated from `agent/graph.py` by `draw_graph.py`. Do not edit by hand.

- **Solid arrow**: always goes to the next step.
- **Dotted arrow**: a decision. The step picks one route (or several, for the
  parallel booking actions after `execute`).
- `ask_missing` and `confirm` pause and wait for a human reply before continuing.

![Ops agent graph](graph.png)

<details>
<summary>Diagram source (Mermaid)</summary>

"""

FOOTER = "\n</details>\n"


def main() -> None:
    drawable_graph = graph.get_graph()
    DOCS_FOLDER.mkdir(exist_ok=True)

    try:
        IMAGE_FILE.write_bytes(drawable_graph.draw_mermaid_png())
        print(f"Saved {IMAGE_FILE}")
    except Exception as error:
        print(f"Could not render the PNG (is the internet reachable?): {error}")

    mermaid_diagram = drawable_graph.draw_mermaid()
    PAGE_FILE.write_text(HEADER + "```mermaid\n" + mermaid_diagram + "```\n" + FOOTER)
    print(f"Saved {PAGE_FILE}")


if __name__ == "__main__":
    main()
