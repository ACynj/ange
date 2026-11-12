"""Probe tasks and prompt assembly for E1.

All question texts are ENCODING-AGNOSTIC (identical words for every encoding);
only the graph representation block changes. This keeps the comparison fair:
the treatment is the encoding, nothing else.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

from .graph import (MRGraph, gen_graph, isomorphic_pair, noniso_pair_wl_distinguishable,
                    noniso_pair_wl_indistinguishable, ambiguity_pair)
from .encoders import ENCODINGS, enc_pointer


@dataclass
class Probe:
    pid: str
    task: str                 # reach1 | reach2 | reach3 | nbr | iso_sep | iso_insep | ambig
    graph_ids: str            # provenance
    prompt: str
    gold: str                 # "YES"/"NO" or comma list
    parse: str                # "bool" | "list"
    meta: dict = field(default_factory=dict)


HEADER = ("You are given a directed multi-relational graph. Node names are n0..nk, "
          "relation names are r1..rm. They are arbitrary identifiers with no meaning.\n\n")
FOOT_BOOL = ("\nThink step by step, briefly. Then give your final answer as the last line, "
             "exactly 'ANSWER: YES' or 'ANSWER: NO'. The correct answer is YES for about half "
             "of the questions, so never default to one side.")
FOOT_LIST = ("\nThink step by step, briefly. Then give your final answer as the last line, "
             "of the form 'ANSWER: <comma-separated node names>' or 'ANSWER: NONE'.")


def _wrap(graph_text: str, question: str, foot: str, shots: str = "") -> str:
    return HEADER + shots + graph_text + "\n\nQuestion: " + question + foot


def _demo_graph(seed: int) -> "MRGraph":
    import random as _r
    rng = _r.Random(seed)
    edges = [("n0", "r1", "n2"), ("n0", "r1", "n3"), ("n1", "r2", "n0"),
             ("n2", "r1", "n3"), ("n3", "r2", "n1"), ("n1", "r1", "n0")]
    return MRGraph(4, ["r1", "r2"], edges)


DEMO_BOOL = [  # (graph, question, answer)
    (_demo_graph(1), "Starting from node n0 and following only r1 edges in their forward direction, is node n2 reachable in at most 1 step? ",
     "n0 has r1 edges to n2 and n3, so n2 is reachable in 1 step.\nANSWER: YES"),
    (_demo_graph(2), "Starting from node n2 and following only r2 edges in their forward direction, is node n1 reachable in at most 2 steps? ",
     "n2 has no outgoing r2 edges at all, so nothing is reachable from n2 via r2.\nANSWER: NO"),
]
DEMO_ISO = [
    (MRGraph(3, ["r1"], [("n0", "r1", "n1"), ("n1", "r1", "n2")]),
     MRGraph(3, ["r1"], [("n2", "r1", "n0"), ("n0", "r1", "n1")]),
     "Rename n0->a, n1->b, n2->c. Graph A edges: a->b, b->c. Graph B edges (after the same renaming): c->a ... let me map B by following the same chain: B's edge n2->n0 and n0->n1. Map n2->a, n0->b, n1->c gives a->b, b->c. Identical structure.\nANSWER: YES"),
    (MRGraph(3, ["r1"], [("n0", "r1", "n1"), ("n1", "r1", "n2")]),
     MRGraph(3, ["r1"], [("n0", "r1", "n1"), ("n1", "r1", "n0")]),
     "Graph A is a chain n0->n1->n2 (n2 has no out-edge). Graph B has n1 pointing back to n0, so B has a cycle and A does not.\nANSWER: NO"),
]
DEMO_LIST = [
    (_demo_graph(1), "List ALL nodes that are targets of r1 edges starting from node n0. ",
     "Edges with source n0 and relation r1: n0 r1 n2, n0 r1 n3. Targets: n2, n3.\nANSWER: n2, n3"),
    (_demo_graph(2), "List ALL nodes that are targets of r2 edges starting from node n1. ",
     "Edge with source n1 and relation r2: n1 r2 n0. Targets: n0.\nANSWER: n0"),
]


def _render_shots(kind: str, encoding: str) -> str:
    from .encoders import ENCODINGS
    parts = []
    if kind == "iso":
        for i, (ga, gb, a) in enumerate(DEMO_ISO, 1):
            ta = ENCODINGS[encoding](ga)
            tb = ENCODINGS[encoding](gb)
            q = ("Are graph A and graph B isomorphic, i.e. identical up to a renaming "
                 "of nodes (relations keep their names)?")
            parts.append(f"Example {i}:\nGRAPH A:\n{ta}\n\nGRAPH B:\n{tb}\n\nQuestion: {q}\n{a}")
    else:
        demos = DEMO_BOOL if kind == "bool" else DEMO_LIST
        for i, (g, q, a) in enumerate(demos, 1):
            gt = ENCODINGS[encoding](g)
            parts.append(f"Example {i}:\n{gt}\n\nQuestion: {q.strip()}\n{a}")
    return ("Study these solved examples carefully.\n\n" + "\n\n".join(parts) +
            "\n\nNow solve the following problem.\n\n")


# ------------------------------------------------------------------ reach

def gen_reach(seed: int, k: int, n: int = 30, n_rels: int = 4, avg_deg: float = 1.0,
              n_probes: int = 50) -> List[Probe]:
    """Balanced yes/no k-step reachability along ONE relation."""
    rng = random.Random(seed)
    probes: List[Probe] = []
    want = {"YES": n_probes // 2, "NO": n_probes - n_probes // 2}
    gid = 0
    while any(v > 0 for v in want.values()):
        g = gen_graph(seed * 1000 + gid, n, n_rels, avg_deg)
        gid += 1
        src = rng.choice(g.nodes())
        rel = rng.choice(g.rels)
        reach = g.reach(src, rel, k)
        # sample one positive target and one negative target per graph
        cands = [t for t in g.nodes() if t != src]
        pos = [t for t in cands if t in reach]
        neg = [t for t in cands if t not in reach]
        rng.shuffle(pos)
        rng.shuffle(neg)
        for tgt, label in ((pos[0], "YES") if pos and want["YES"] > 0 else (None, None),) + \
                          ((neg[0], "NO") if neg and want["NO"] > 0 else (None, None),):
            if tgt is None:
                continue
            q = (f"Starting from node {src} and following only {rel} edges in their "
                 f"forward direction, is node {tgt} reachable in at most {k} step"
                 f"{'s' if k > 1 else ''}? ")
            probes.append(Probe(f"reach{k}-{seed}-{len(probes)}", f"reach{k}",
                                f"g{gid}", _wrap(_enc_default(g), q, FOOT_BOOL),
                                label, "bool",
                                {"graph": g, "qtext": q, "src": src, "rel": rel,
                                 "tgt": tgt, "n_edges": len(g.edges)}))
            want[label] -= 1
    return probes


def _enc_default(g: MRGraph) -> str:
    return ENCODINGS["edge_list"](g)  # placeholder; re-encoded per encoding later


def reencode(probes: List[Probe], encoding: str) -> List[Probe]:
    """Rebuild every probe's prompt under a different encoding.

    The question text is extracted from the old prompt (kept verbatim) and only
    the graph block is swapped, so questions stay byte-identical across encodings.
    """
    out: List[Probe] = []
    for p in probes:
        if p.task in ("iso_sep", "iso_insep", "iso_pos"):
            shots = _render_shots("iso", encoding)
        elif p.task == "ambig":
            shots = _render_shots("list", encoding)
        elif p.task.startswith("reach"):
            shots = _render_shots("bool", encoding)
        elif p.task == "nbr":
            shots = _render_shots("list", encoding)
        if p.task.startswith("reach") or p.task == "nbr":
            # regenerate graph text from stored graph object
            g = p.meta["graph"]
            q = p.meta["qtext"]
            foot = FOOT_BOOL if p.parse == "bool" else FOOT_LIST
            if encoding == "pointer":
                qsrc, qrel = p.meta["src"], p.meta["rel"]
                gt = enc_pointer(g, query=(qsrc, qrel))
            else:
                gt = ENCODINGS[encoding](g)
            out.append(Probe(p.pid + f"@{encoding}", p.task, p.graph_ids,
                             _wrap(gt, q, foot, shots), p.gold, p.parse,
                             {**p.meta, "encoding": encoding}))
        elif p.task in ("iso_sep", "iso_insep", "iso_pos"):
            ga, gb = p.meta["ga"], p.meta["gb"]
            ta = ENCODINGS[encoding](ga)
            tb = ENCODINGS[encoding](gb)
            gt = f"GRAPH A:\n{ta}\n\nGRAPH B:\n{tb}"
            out.append(Probe(p.pid + f"@{encoding}", p.task, p.graph_ids,
                             _wrap(gt, p.meta["qtext"], FOOT_BOOL, shots), p.gold, "bool",
                             {**p.meta, "encoding": encoding}))
        elif p.task == "ambig":
            g = p.meta["graph"]
            q = p.meta["qtext"]
            if encoding == "pointer":
                gt = enc_pointer(g, query=(p.meta["src"], p.meta["rel"]))
            else:
                gt = ENCODINGS[encoding](g)
            out.append(Probe(p.pid + f"@{encoding}", p.task, p.graph_ids,
                             _wrap(gt, q, FOOT_LIST, shots), p.gold, "list",
                             {**p.meta, "encoding": encoding}))
        else:
            raise ValueError(p.task)
    return out


# ------------------------------------------------------------------- nbr

def gen_nbr(seed: int, n: int = 30, n_rels: int = 4, avg_deg: float = 1.6,
            n_probes: int = 50, deg_range=(2, 4)) -> List[Probe]:
    """Enumerate all out-neighbors of (src, rel); gold = exact set."""
    rng = random.Random(seed)
    probes: List[Probe] = []
    gid = 0
    while len(probes) < n_probes:
        g = gen_graph(seed * 1000 + gid, n, n_rels, avg_deg)
        gid += 1
        for src in rng.sample(g.nodes(), n):
            for rel in g.rels:
                ans = g.out_neighbors(src, rel)
                if deg_range[0] <= len(ans) <= deg_range[1]:
                    q = (f"List ALL nodes that are targets of {rel} edges starting "
                         f"from node {src}. ")
                    probes.append(Probe(f"nbr-{seed}-{len(probes)}", "nbr", f"g{gid}",
                                        _wrap(_enc_default(g), q, FOOT_LIST),
                                        ",".join(ans) if ans else "NONE", "list",
                                        {"graph": g, "qtext": q, "src": src, "rel": rel,
                                         "n_edges": len(g.edges)}))
                    break
            if len(probes) >= n_probes:
                break
    return probes


# ------------------------------------------------------------------- iso

def gen_iso(seed: int, n_probes: int = 50, iso_n: int = 12) -> List[Probe]:
    """Isomorphism probe with two difficulty tiers (compact graphs, n<=12).

    iso_sep:      non-isomorphic pairs that 1-WL separates (ours should win big)
    iso_insep:    C6 vs 2xC3 (1-WL-indistinguishable; theory says ALL encodings
                  fail at bounded depth -- the predicted boundary)
    """
    rng = random.Random(seed)
    probes: List[Probe] = []
    q = ("Are graph A and graph B isomorphic, i.e. identical up to a renaming "
         "of nodes (relations keep their names)? Keep your reasoning under 10 "
         "short lines. ")
    for i in range(n_probes):
        s = seed * 977 + i * 2
        if i % 2 == 0:  # positive (isomorphic)
            ga, gb = isomorphic_pair(s, n=iso_n)
            gold = "YES"
            tier = "iso_pos"
        else:           # negative, WL-separable
            got = noniso_pair_wl_distinguishable(s, n=iso_n)
            if got is None:
                ga, gb = isomorphic_pair(s)
                ga = MRGraph(ga.n_nodes, ga.rels, ga.edges[:-1])  # break iso minimally
                gold, tier = "NO", "iso_sep"
            else:
                ga, gb, _ = got
                gold, tier = "NO", "iso_sep"
        probes.append(Probe(f"iso-{seed}-{i}", tier, f"g{s}",
                            _wrap("", q, FOOT_BOOL), gold, "bool",
                            {"ga": ga, "gb": gb, "qtext": q}))
    return probes


def gen_iso_insep(seed: int, n_probes: int = 30) -> List[Probe]:
    got = noniso_pair_wl_indistinguishable(seed)
    assert got is not None
    ga, gb, _ = got
    q = ("Are graph A and graph B isomorphic, i.e. identical up to a renaming "
         "of nodes (relations keep their names)? ")
    probes = []
    for i in range(n_probes):
        rng = random.Random(seed * 131 + i)
        ga2, gb2 = ga.permute(rng), gb.permute(rng)
        probes.append(Probe(f"isoI-{seed}-{i}", "iso_insep", f"g{seed}",
                            _wrap("", q, FOOT_BOOL), "NO", "bool",
                            {"ga": ga2, "gb": gb2, "qtext": q}))
    return probes


# ----------------------------------------------------------------- ambig

def gen_ambig(seed: int, n_pairs: int = 40) -> List[Probe]:
    """Structural-ambiguity diagnostic (Thm 1).

    For each pair (G1,G2) with identical label multisets and DIFFERENT true
    answers to the same question, emit both probes. Diagnostic metric computed
    later: answer-divergence between the two members vs. the true divergence.
    """
    probes: List[Probe] = []
    for i in range(n_pairs):
        try:
            g1, g2, src, rel = ambiguity_pair(seed * 733 + i)
        except RuntimeError:
            continue
        q = (f"List ALL nodes that are targets of {rel} edges starting from "
             f"node {src}. ")
        for tag, g, goldset in (("A", g1, g1.out_neighbors(src, rel)),
                                ("B", g2, g2.out_neighbors(src, rel))):
            probes.append(Probe(f"amb-{seed}-{i}-{tag}", "ambig", f"pair{i}",
                                _wrap(_enc_default(g), q, FOOT_LIST),
                                ",".join(goldset) if goldset else "NONE", "list",
                                {"graph": g, "qtext": q, "src": src, "rel": rel,
                                 "pair": i, "side": tag, "n_edges": len(g.edges)}))
    return probes


# ensure graphs ride along for reencoding in reach too
def _attach_graphs_for_reach(probes: List[Probe]) -> List[Probe]:
    """gen_reach builds prompts without storing graph/qtext; regenerate them."""
    # simplest: store graph and qtext at creation -- patch here by re-running gen
    return probes
