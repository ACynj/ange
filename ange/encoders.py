"""The eight graph encodings under study.

Every encoder maps a MRGraph to text. The text is embedded into a fixed
task template (see prompts.py) so that ONLY the graph representation varies.

Baselines (role carried by POSITION / prose, predicted to fail by Thm 1):
  edge_list      "n3 r1 n17" one triple per line
  adjacency_list grouped by source node
  incident_list  grouped by target node
  nl             natural-language sentences (talk-like-a-graph style)
  json           JSON array of triples
  sog            lossless-impossible single-line digest (lossy by design)
  dfs_serial     reversible frequency-aware DFS serialization (Guo'26 style)

Ours (role carried by CONTENT tokens, Thm 2 construction):
  pointer        G_PG -- every edge is a self-contained addressed record
                 "EDGE rel=r1 src=n3 tgt=n17" and the query uses the SAME
                 template, so answering = one content-addressed lookup.
"""
from __future__ import annotations

from typing import List

from .graph import MRGraph


def _node_line(g: MRGraph) -> str:
    return "Nodes: " + ", ".join(g.nodes())


def _rel_line(g: MRGraph) -> str:
    return "Relations: " + ", ".join(g.rels)


# ---------------------------------------------------------------- baselines

def enc_edge_list(g: MRGraph) -> str:
    lines = [_node_line(g), _rel_line(g), "Edges (format: source relation target):"]
    lines += [f"{s} {r} {t}" for s, r, t in g.edges]
    return "\n".join(lines)


def enc_adjacency_list(g: MRGraph) -> str:
    by_src: dict = {}
    for s, r, t in g.edges:
        by_src.setdefault(s, {}).setdefault(r, []).append(t)
    lines = [_node_line(g), _rel_line(g), "Outgoing edges grouped by node:"]
    for v in g.nodes():
        if v in by_src:
            parts = [f"{r} -> {', '.join(sorted(ts))}" for r, ts in sorted(by_src[v].items())]
            lines.append(f"{v}: " + "; ".join(parts))
    return "\n".join(lines)


def enc_incident_list(g: MRGraph) -> str:
    by_tgt: dict = {}
    for s, r, t in g.edges:
        by_tgt.setdefault(t, {}).setdefault(r, []).append(s)
    lines = [_node_line(g), _rel_line(g), "Incoming edges grouped by node:"]
    for v in g.nodes():
        if v in by_tgt:
            parts = [f"{r} <- {', '.join(sorted(ss))}" for r, ss in sorted(by_tgt[v].items())]
            lines.append(f"{v}: " + "; ".join(parts))
    return "\n".join(lines)


def enc_nl(g: MRGraph) -> str:
    lines = [_node_line(g), _rel_line(g), "Description:"]
    for s, r, t in g.edges:
        lines.append(f"{s} points to {t} through the relation {r}.")
    return "\n".join(lines)


def enc_json(g: MRGraph) -> str:
    import json
    return json.dumps({"nodes": g.nodes(), "relations": g.rels,
                       "edges": [{"src": s, "rel": r, "tgt": t} for s, r, t in g.edges]},
                      indent=0, separators=(", ", ": "))


def enc_sog(g: MRGraph) -> str:
    """Lossy structural digest -- a whole graph in one line (SOG-token style)."""
    from collections import Counter
    deg = Counter()
    for s, r, t in g.edges:
        deg[s] += 1
    degs = " ".join(f"{v}:{deg.get(v, 0)}" for v in g.nodes())
    return (f"[GRAPH n={g.n_nodes} m={len(g.edges)} rels={len(g.rels)} "
            f"degs({degs})]")


def enc_dfs_serial(g: MRGraph) -> str:
    """Reversible frequency-aware DFS serialization (Guo'26-style).

    Serializes the undirected skeleton by DFS, emitting 'visit' tokens for new
    nodes and 'back' tokens for cross/visited edges; fully reversible.
    """
    import heapq
    adj: dict = {}
    for s, r, t in g.edges:
        adj.setdefault(s, []).append((t, r, "fwd"))
        adj.setdefault(t, []).append((s, r, "rev"))
    # order neighbours by degree (frequency-aware), tie-break by name
    deg = {v: len(adj.get(v, [])) for v in g.nodes()}
    def nb_key(v_r_d):
        v = v_r_d[0]
        return (-deg[v], v)
    for v in adj:
        adj[v].sort(key=nb_key)
    visited = set()
    tokens: List[str] = [f"R={','.join(g.rels)}"]
    stack_entry = sorted(g.nodes(), key=lambda v: (-deg.get(v, 0), v))
    for root in stack_entry:
        if root in visited:
            continue
        visited.add(root)
        tokens.append(f"V:{root}")
        stack = [(root, iter(adj.get(root, [])))]
        while stack:
            node, it = stack[-1]
            advanced = False
            for nxt, rel, direction in it:
                tag = f"{rel}{'+' if direction == 'fwd' else '-'}"
                if nxt not in visited:
                    visited.add(nxt)
                    tokens.append(f"{tag}>{nxt}")
                    stack.append((nxt, iter(adj.get(nxt, []))))
                    advanced = True
                    break
                else:
                    tokens.append(f"{tag}*{nxt}")
                    advanced = True
                    break
            if not advanced:
                stack.pop()
    return " ".join(tokens)


# ------------------------------------------------------------------- ours

def enc_pointer(g: MRGraph, query: tuple | None = None) -> str:
    """G_PG (Thm 2 construction).

    - definition block (skippable): nodes and relations
    - flat record block: one self-contained addressed record per edge,
      'EDGE rel=<r> src=<u> tgt=<v>' -- structural roles are CONTENT tokens,
      never positions; the record is position-invariant and order-invariant
    - optional query record using the SAME template with a hole:
      'EDGE rel=<r> src=<u> tgt=?'
    """
    lines = [_node_line(g), _rel_line(g), "Edge records:"]
    lines += [f"EDGE rel={r} src={s} tgt={t}" for s, r, t in g.edges]
    if query is not None:
        qsrc, qrel = query
        lines.append(f"QUERY-TEMPLATE: EDGE rel={qrel} src={qsrc} tgt=?")
    return "\n".join(lines)


def enc_pointer2(g: MRGraph, query: tuple | None = None) -> str:
    """G_PG-v2 (compact): roles carried by directional content tokens.

    Each edge = one self-contained record 'n3 -r1-> n17': the relation token
    '-r1->' itself encodes src-side/tgt-side (direction is CONTENT, not
    position). ~7 tokens/edge vs ~12 in v1, close to edge_list (~6).
    Query uses the same directional template with a hole.
    """
    lines = [_node_line(g), _rel_line(g), "Edges (u -rel-> v):"]
    lines += [f"{s} -{r}-> {t}" for s, r, t in g.edges]
    if query is not None:
        qsrc, qrel = query
        lines.append(f"QUERY: {qsrc} -{qrel}-> ?")
    return "\n".join(lines)


def triangle_partners(g: MRGraph):
    """Second-order (2-WL initial-color) information: for each node, the set
    of partners with which it shares a triangle (edge-level, any relation)."""
    from collections import defaultdict
    und = defaultdict(set)
    for s, r, t in g.edges:
        und[s].add(t); und[t].add(s)
    tri = defaultdict(set)
    for s, r, t in g.edges:
        for x in und[t]:
            if x == s:
                continue
            if s in und[x]:  # s-t-x closes a triangle
                tri[s].add(x); tri[t].add(x); tri[x].add(s)
    return tri


def enc_pointer2pg(g: MRGraph, query: tuple | None = None) -> str:
    """G_PG-2: compact pointer records + explicit triangle records.

    The TRI lines expose exactly the 2-WL initial colors that 1-hop records
    cannot express (e.g., C6 has none, 2xC3 has six), so Thm.-3-style
    alignment extends one WL level (k layers recover 2-WL)."""
    lines = [_node_line(g), _rel_line(g), "Edges (u -rel-> v):"]
    lines += [f"{s} -{r}-> {t}" for s, r, t in g.edges]
    tri = triangle_partners(g)
    lines.append("Triangles (TRI v -> partners):")
    lines += [f"TRI {v} -> {{{', '.join(sorted(tri[v]))}}}" for v in sorted(tri)]
    if query is not None:
        qsrc, qrel = query
        lines.append(f"QUERY: {qsrc} -{qrel}-> ?")
    return "\n".join(lines)


ENCODINGS = {
    "edge_list": enc_edge_list,
    "adjacency_list": enc_adjacency_list,
    "incident_list": enc_incident_list,
    "nl": enc_nl,
    "json": enc_json,
    "sog": enc_sog,
    "dfs_serial": enc_dfs_serial,
    "pointer": enc_pointer,
    "pointer2": enc_pointer2,
    "pointer2pg": enc_pointer2pg,
}

# subset used in the smoke test
SMOKE_ENCODINGS = ["edge_list", "adjacency_list", "nl", "sog", "dfs_serial", "pointer"]
