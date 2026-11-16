"""
Draws the agent's graph and saves it to docs/graph.md.

It asks the compiled LangGraph graph to describe itself as a Mermaid diagram,
so the picture always matches the real wiring in agent/graph.py. GitHub renders
the diagram automatically. Re-run after changing the graph: `uv run python draw_graph.py`.
"""

import logging
from pathlib import Path

logging.disable(logging.INFO)

from agent.graph import graph  # noqa: E402

OUTPUT_FILE = Path(__file__).parent / "docs" / "graph.md"

HEADER = """# Ops agent graph

Generated from `agent/graph.py` by `draw_graph.py`. Do not edit by hand.

- **Solid arrow**: always goes to the next step.
- **Dotted arrow**: a decision. The step picks one route (or several, for the
  parallel booking actions after `execute`).
- `ask_missing` and `confirm` pause and wait for a human reply before continuing.

"""


def main() -> None:
    mermaid_diagram = graph.get_graph().draw_mermaid()
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    OUTPUT_FILE.write_text(HEADER + "```mermaid\n" + mermaid_diagram + "```\n")
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
