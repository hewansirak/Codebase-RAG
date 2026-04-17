"""Interactive repo map: a treemap of every file, with the files retrieved for an answer highlighted."""

from __future__ import annotations

from collections import defaultdict

import plotly.graph_objects as go

from .retrieval import Hit
from .store import Index

HIT_COLOR = "#ff6b35"
IDLE_COLOR = "#2b3a55"


def map_data(index: Index, hits: list[Hit]) -> dict[str, list]:
    """Treemap ids/labels/parents/values/colours/hover text for one repository."""
    lines_per_file: dict[str, int] = defaultdict(int)
    for c in index.chunks:
        lines_per_file[c.path] += c.end_line - c.start_line + 1
    retrieved: dict[str, list[str]] = defaultdict(list)
    for rank, hit in enumerate(hits, start=1):
        if hit.chunk.repo == index.repo:
            c = hit.chunk
            retrieved[c.path].append(f"#{rank} {c.symbol} (L{c.start_line}-{c.end_line})")

    ids, labels, parents, values, colors, hover = [index.repo], [index.repo], [""], [0], [IDLE_COLOR], [""]
    seen = {index.repo}
    for path in sorted(lines_per_file):
        parts = path.split("/")
        for depth in range(1, len(parts)):
            folder = index.repo + "/" + "/".join(parts[:depth])
            if folder not in seen:
                seen.add(folder)
                parent = index.repo + ("/" + "/".join(parts[: depth - 1]) if depth > 1 else "")
                ids.append(folder), labels.append(parts[depth - 1]), parents.append(parent)
                values.append(0), colors.append(IDLE_COLOR), hover.append("")
        parent = index.repo + ("/" + "/".join(parts[:-1]) if len(parts) > 1 else "")
        ids.append(f"{index.repo}/{path}"), labels.append(parts[-1]), parents.append(parent)
        values.append(lines_per_file[path])
        colors.append(HIT_COLOR if path in retrieved else IDLE_COLOR)
        hover.append("<br>".join(retrieved[path]) if path in retrieved else f"{lines_per_file[path]} lines")
    return {"ids": ids, "labels": labels, "parents": parents, "values": values, "colors": colors, "hover": hover}


def build_repo_map(index: Index, hits: list[Hit]) -> go.Figure:
    data = map_data(index, hits)
    fig = go.Figure(
        go.Treemap(
            ids=data["ids"],
            labels=data["labels"],
            parents=data["parents"],
            values=data["values"],
            marker={"colors": data["colors"]},
            hovertext=data["hover"],
            hoverinfo="text+label",
            branchvalues="remainder",
        )
    )
    fig.update_layout(margin={"t": 10, "l": 0, "r": 0, "b": 0}, height=460)
    return fig
