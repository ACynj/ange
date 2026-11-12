"""E3: SFT data for LoRA fine-tuning under two encodings.

Same graphs, same queries, same answers -- ONLY the encoding differs
(pointer vs edge_list). Gold CoT is synthesized programmatically so the
supervision is identical in content across the two arms.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

from .graph import MRGraph, gen_graph
from .encoders import ENCODINGS, enc_pointer, enc_pointer2, enc_pointer2pg
from .probes import HEADER, FOOT_BOOL, FOOT_LIST


def _path(g: MRGraph, src: str, rel: str, dst: str, k: int) -> Optional[List[str]]:
    """Shortest path (<=k) along `rel` edges, or None."""
    from collections import deque
    q = deque([(src, [src])])
    seen = {src}
    while q:
        u, p = q.popleft()
        if len(p) - 1 > k:
            continue
        if u == dst and len(p) > 1:
            return p
        for s, r, t in g.edges:
            if s == u and r == rel and t not in seen:
                seen.add(t)
                q.append((t, p + [t]))
    return None


def reach_sample(g: MRGraph, rng: random.Random, k: int, prefer: str = "any") -> Optional[Tuple[str, str, str]]:
    """Returns (question, gold_answer, meta(src,rel,tgt)) or None.
    prefer='YES'/'NO' forces the label (for balance); 'any' samples freely."""
    src = rng.choice(g.nodes()); rel = rng.choice(g.rels)
    reach = g.reach(src, rel, k)
    cands = [t for t in g.nodes() if t != src]
    if not cands:
        return None
    if prefer == "YES":
        if not reach:
            return None
        tgt = rng.choice(sorted(reach))
    elif prefer == "NO":
        negs = [t for t in cands if t not in reach]
        if not negs:
            return None
        tgt = rng.choice(negs)
    else:
        tgt = rng.choice(cands)
    # balance: keep if it differs from the running majority is handled by caller
    q = (f"Starting from node {src} and following only {rel} edges in their "
         f"forward direction, is node {tgt} reachable in at most {k} step"
         f"{'s' if k > 1 else ''}? ")
    if tgt in reach:
        p = _path(g, src, rel, tgt, k)
        chain = " -> ".join(p)
        a = (f"Following {rel} edges from {src}: {chain}. {tgt} is reached in "
             f"{len(p)-1} step(s), which is within {k}.\nANSWER: YES")
    else:
        nodes_reached = ", ".join(sorted(reach)) if reach else "none"
        a = (f"Following {rel} edges from {src}, the nodes reachable in at most "
             f"{k} steps are: {nodes_reached}. {tgt} is not among them.\nANSWER: NO")
    return q, a, (src, rel, tgt)


def nbr_sample(g: MRGraph, rng: random.Random) -> Optional[Tuple[str, str, str]]:
    src = rng.choice(g.nodes()); rel = rng.choice(g.rels)
    ans = g.out_neighbors(src, rel)
    if not (1 <= len(ans) <= 5):
        return None
    q = f"List ALL nodes that are targets of {rel} edges starting from node {src}. "
    ev = "; ".join(f"{src} {rel} {t}" for t in ans)
    a = f"{rel} edges from {src}: {ev}.\nANSWER: {', '.join(ans)}"
    return q, a, (src, rel, None)


def build_prompt(g: MRGraph, q: str, task: str, encoding: str, meta) -> str:
    if encoding == "pointer":
        gt = enc_pointer(g, query=(meta[0], meta[1]))
    elif encoding == "pointer2":
        gt = enc_pointer2(g, query=(meta[0], meta[1]))
    elif encoding == "pointer2pg":
        gt = enc_pointer2pg(g, query=(meta[0], meta[1]))
    else:
        gt = ENCODINGS[encoding](g)
    foot = FOOT_BOOL if task.startswith("reach") else FOOT_LIST
    return HEADER + gt + "\n\nQuestion: " + q + foot


def gen_split(seed_lo: int, seed_hi: int, per_graph: int, n: int = 20,
              n_rels: int = 4, avg_deg: float = 1.6) -> List[dict]:
    """Generate (graph, task, q, answer) samples; deterministic by seed range."""
    out: List[dict] = []
    for sd in range(seed_lo, seed_hi):
        rng = random.Random(sd * 7919)
        g = gen_graph(sd, n, n_rels, avg_deg)
        made = 0
        want_yes = True
        while made < per_graph:
            k = rng.choice([1, 2, 3])
            if rng.random() < 0.5:
                s = reach_sample(g, rng, k, prefer=("YES" if want_yes else "NO"))
                if s is None:
                    s = reach_sample(g, rng, k, prefer=("NO" if want_yes else "YES"))
                if s is None:
                    continue
                want_yes = not want_yes
                q, a, meta = s
                out.append({"graph": g, "task": f"reach{k}", "q": q, "a": a, "meta": meta})
            else:
                s = nbr_sample(g, rng)
                if s is None:
                    continue
                q, a, meta = s
                out.append({"graph": g, "task": "nbr", "q": q, "a": a, "meta": meta})
            made += 1
    return out


def make_chat(split: List[dict], encoding: str) -> List[dict]:
    return [{"messages": [
        {"role": "user", "content": build_prompt(s["graph"], s["q"], s["task"], encoding, s["meta"])},
        {"role": "assistant", "content": s["a"]}], "task": s["task"]} for s in split]


# ------------------------- iso training support -------------------------
from .graph import isomorphic_pair, noniso_pair_wl_distinguishable, noniso_pair_wl_indistinguishable

ISO_Q = ("Are graph A and graph B isomorphic, i.e. identical up to a renaming "
         "of nodes (relations keep their names)? Keep your reasoning under 6 "
         "short lines. ")


def iso_sample(rng, n=12, want="pos"):
    """One iso training sample. Returns dict with graphs + gold."""
    if want == "pos":
        ga, gb = isomorphic_pair(rng.randrange(10**6), n=n)
        gold = "YES"
        cot = ("Rename nodes of B to match A: the edge multisets correspond "
               "one-to-one after renaming.\nANSWER: YES")
    else:
        got = noniso_pair_wl_distinguishable(rng.randrange(10**6), n=n)
        if got is None:
            return None
        ga, gb, _ = got
        gold = "NO"
        cot = ("A has a directed 3-cycle motif where B has a transitive "
               "triple; no renaming can map a cycle onto a chain.\nANSWER: NO")
    return {"ga": ga, "gb": gb, "gold": gold, "cot": cot, "task": "iso"}


def iso_insep_sample(rng):
    got = noniso_pair_wl_indistinguishable(rng.randrange(10**6))
    ga, gb, _ = got
    cot = ("Count TRI records: graph A has none, graph B has six nodes with "
           "triangle partners. Different triangle multisets.\nANSWER: NO")
    return {"ga": ga, "gb": gb, "gold": "NO", "cot": cot, "task": "iso"}


def build_prompt_iso(ga, gb, encoding):
    """Two-graph prompt for iso training/eval."""
    ta = enc_pointer2pg(ga) if encoding == "pointer2pg" else (
        enc_pointer2(ga) if encoding == "pointer2" else ENCODINGS[encoding](ga))
    tb = enc_pointer2pg(gb) if encoding == "pointer2pg" else (
        enc_pointer2(gb) if encoding == "pointer2" else ENCODINGS[encoding](gb))
    return (HEADER + f"GRAPH A:\n{ta}\n\nGRAPH B:\n{tb}\n\nQuestion: "
            + ISO_Q + FOOT_BOOL)


def gen_split_with_iso(seed_lo, seed_hi, per_graph, iso_frac=0.34, n=20):
    """Mixed split: reach/nbr + iso pairs (pos & wl-sep alternating)."""
    import random as _r
    rng = _r.Random(seed_lo * 31)
    out = gen_split(seed_lo, seed_hi, per_graph, n=n)
    # append iso samples: ~1/3 of total
    n_iso = int(len(out) * iso_frac / (1 - iso_frac))
    made, want_yes = 0, True
    while made < n_iso:
        if want_yes:
            s = iso_sample(rng, want="pos")
        else:
            s = iso_sample(rng, want="sep")
        if s is None:
            continue
        out.append({"graph": s["ga"], "task": "iso_" + ("pos" if s["gold"] == "YES" else "sep"),
                    "q": ISO_Q, "a": s["cot"], "meta": (None, None, None), "graph_b": s["gb"],
                    "gold_bool": s["gold"]})
        want_yes = not want_yes
        made += 1
    # sprinkle a few insep pairs (the boundary case)
    for _ in range(max(6, n_iso // 12)):
        s = iso_insep_sample(rng)
        out.append({"graph": s["ga"], "task": "iso_insep", "q": ISO_Q, "a": s["cot"],
                    "meta": (None, None, None), "graph_b": s["gb"], "gold_bool": "NO"})
    rng.shuffle(out)
    return out
