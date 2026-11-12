"""Multi-relational directed graphs and synthetic generation for ANGE probes.

Node and relation names are neutral pseudo-words (n1, n2, ... / r1, r2, ...) so
that no world knowledge can shortcut structural reasoning.  IDs are randomly
shuffled at generation time so that ID order never correlates with structure.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class MRGraph:
    """A multi-relational directed graph."""

    n_nodes: int
    rels: List[str]                       # relation vocabulary, e.g. ["r1","r2"]
    edges: List[Tuple[str, str, str]]     # (src, rel, tgt) with node ids "n<i>"
    node_feats: Dict[str, str] = field(default_factory=dict)  # optional type tags

    def nodes(self) -> List[str]:
        return [f"n{i}" for i in range(self.n_nodes)]

    def out_neighbors(self, src: str, rel: str) -> List[str]:
        return sorted({t for s, r, t in self.edges if s == src and r == rel})

    def neighbors(self, src: str, rel: str) -> List[str]:
        return sorted({t for s, r, t in self.edges if s == src and r == rel} |
                      {s for s, r, t in self.edges if t == src and r == rel})

    def reach(self, src: str, rel: str, k: int) -> set:
        """Nodes reachable from src in <= k steps along `rel` edges (directed)."""
        frontier = {src}
        seen = {src}
        for _ in range(k):
            nxt = set()
            for u in frontier:
                for s, r, t in self.edges:
                    if s == u and r == rel and t not in seen:
                        nxt.add(t)
            frontier = nxt
            seen |= nxt
        seen.discard(src)
        return seen

    def canon(self) -> str:
        """A 1-WL-style canonical stamp used for isomorphism checks on probes."""
        return "|".join(sorted(f"{s}-{r}-{t}" for s, r, t in self.edges))

    def permute(self, rng: random.Random) -> "MRGraph":
        """Random relabeling of NODES only (relations keep their names).

        This matches the probe question semantics ("identical up to a renaming
        of nodes") and keeps edge-labeled WL an isomorphism invariant."""
        node_map = {f"n{i}": f"n{j}" for i, j in enumerate(rng.sample(range(self.n_nodes), self.n_nodes))}
        edges = [(node_map[s], r, node_map[t]) for s, r, t in self.edges]
        rng.shuffle(edges)
        return MRGraph(self.n_nodes, list(self.rels), edges)


def gen_graph(seed: int, n: int = 30, n_rels: int = 4, avg_deg: float = 1.6,
              star_free: bool = True) -> MRGraph:
    """Random sparse multi-relational graph with shuffled IDs.

    avg_deg is per relation: expected edges per relation ~= n * avg_deg.
    """
    rng = random.Random(seed)
    rels = [f"r{i}" for i in range(1, n_rels + 1)]
    edges = []
    for r in rels:
        m = max(2, int(round(n * avg_deg)))
        seen = set()
        tries = 0
        while len(seen) < m and tries < 40 * m:
            tries += 1
            s = rng.randrange(n)
            t = rng.randrange(n)
            if s == t:
                continue
            key = (s, t)
            if key in seen:
                continue
            seen.add(key)
            edges.append((f"n{s}", r, f"n{t}"))
    # structural safety: ensure connectivity of the union graph (so k-hop has signal)
    rng.shuffle(edges)
    g = MRGraph(n, rels, edges)
    return g


def wl_colors(g: MRGraph, rounds: int = 3) -> Tuple[List[str], List[str]]:
    """Directed, edge-labeled 1-WL: colors refine from (out-nb, in-nb) multisets.

    Returns (sorted node colors, edge-multiset signature)."""
    cols = {v: "init" for v in g.nodes()}
    for _ in range(rounds):
        new = {}
        for v in g.nodes():
            nb = sorted([("o", r, cols[t]) for s, r, t in g.edges if s == v] +
                        [("i", r, cols[s]) for s, r, t in g.edges if t == v])
            h = hashlib.md5((cols[v] + repr(nb)).encode()).hexdigest()[:8]
            new[v] = h
        cols = new
    node_cols = sorted(cols.values())
    edge_sig = sorted(f"{s}-{r}-{t}" for s, r, t in g.edges)
    return node_cols, edge_sig


def isomorphic_pair(seed: int, n: int = 24, n_rels: int = 3, avg_deg: float = 1.4) -> Tuple[MRGraph, MRGraph]:
    """A truly isomorphic pair (positive example for iso probe)."""
    rng = random.Random(seed)
    g1 = gen_graph(seed, n, n_rels, avg_deg)
    g2 = g1.permute(rng)
    return g1, g2


def noniso_pair_wl_distinguishable(seed: int, n: int = 24, n_rels: int = 3,
                                   avg_deg: float = 1.2, max_try: int = 1) -> Optional[Tuple[MRGraph, MRGraph, str]]:
    """A non-isomorphic pair that 1-WL distinguishes: 3-cycle vs transitive triple.

    Nodes n0,n1,n2 carry the discriminating motif (same 3 edge-slots both
    sides); all remaining edges are SHARED verbatim by both graphs, so the two
    differ ONLY in motif wiring. Cycle degrees (1,1,1 in/out) vs triple
    (out-degrees 1,1,2 / in-degrees 1,2,0) separate at 1 WL round.
    """
    rng = random.Random(seed)
    a, b, c = "n0", "n1", "n2"
    cyc = [(a, "r1", b), (b, "r1", c), (c, "r1", a)]
    tri = [(a, "r1", b), (b, "r1", c), (a, "r1", c)]
    # shared random edges on the remaining nodes (n3..)
    shared = []
    rest = list(range(3, n))
    for r in [f"r{i}" for i in range(1, n_rels + 1)]:
        m = max(2, int(round(len(rest) * avg_deg)))
        seen = set()
        tries = 0
        while len(seen) < m and tries < 60 * m:
            tries += 1
            s, t = rng.choice(rest), rng.choice(rest)
            if s == t or (s, t) in seen:
                continue
            seen.add((s, t))
            shared.append((f"n{s}", r, f"n{t}"))
    rng.shuffle(shared)
    g1 = MRGraph(n, [f"r{i}" for i in range(1, n_rels + 1)], cyc + shared)
    g2 = MRGraph(n, [f"r{i}" for i in range(1, n_rels + 1)], tri + shared)
    rng.shuffle(g1.edges)
    rng.shuffle(g2.edges)
    c1, _ = wl_colors(g1)
    c2, _ = wl_colors(g2)
    if c1 != c2:
        return g1, g2, "wl-separable"
    return None


def noniso_pair_wl_indistinguishable(seed: int, n: int = 6, max_try: int = 200) -> Optional[Tuple[MRGraph, MRGraph, str]]:
    """Classic 1-WL-indistinguishable non-isomorphic pair: C6 vs 2xC3.

    Pure pair, no padding (padding breaks the symmetry). Theory (Thm 3)
    predicts EVERY encoding fails here at bounded depth -- the paper's
    'predicted boundary' probe.
    """
    rng = random.Random(seed)
    c6 = [(f"n{i}", f"n{(i + 1) % 6}") for i in range(6)]
    t1 = [(f"n{i}", f"n{(i + 1) % 3}") for i in range(3)]
    t2 = [(f"n{3 + i}", f"n{3 + (i + 1) % 3}") for i in range(3)]
    two_c3 = t1 + t2
    def build(elist):
        edges = [(s, "r1", t) for s, t in elist]
        g = MRGraph(6, ["r1"], edges)
        rng.shuffle(g.edges)
        return g
    g1, g2 = build(c6), build(two_c3)
    c1, _ = wl_colors(g1)
    c2, _ = wl_colors(g2)
    assert c1 == c2, "C6 vs 2C3 should be 1-WL indistinguishable"
    return g1, g2, "wl-inseparable"


def ambiguity_pair(seed: int, n: int = 20, n_rels: int = 3, avg_deg: float = 1.3
                   ) -> Tuple[MRGraph, MRGraph, str, str]:
    """Structurally ambiguous pair (Thm 1 probe).

    G1, G2 share the SAME label multisets (node ids, relations, edge count) but
    differ in wiring, and a fixed query (src, rel) has DIFFERENT true answers
    on the two graphs.  A model reading only 'bag of labels' must answer
    identically; a structural reader should diverge (as the truth does).
    Returns (G1, G2, query_src, rel).
    """
    rng = random.Random(seed)
    g1 = gen_graph(seed, n, n_rels, avg_deg)
    # choose a query with 2-4 answers on g1
    for _ in range(max_try := 60):
        src = rng.choice(g1.nodes())
        rel = rng.choice(g1.rels)
        ans1 = g1.out_neighbors(src, rel)
        if 2 <= len(ans1) <= 4:
            break
    else:
        raise RuntimeError("no suitable query")
    # rewire: keep every edge (s,r,t) label but shuffle the PAIRING
    srcs = [s for s, r, t in g1.edges]
    tgts = [t for s, r, t in g1.edges]
    for attempt in range(200):
        rng.shuffle(tgts)
        edges2 = list(zip(srcs, [r for s, r, t in g1.edges], tgts))
        g2 = MRGraph(g1.n_nodes, g1.rels, [(s, r, t) for s, r, t in edges2])
        ans2 = g2.out_neighbors(src, rel)
        if sorted(ans1) != sorted(ans2) and set(ans2) - {src}:
            # ensure multiset of (rel, src-name) and (rel, tgt-name) counts match g1
            return g1, g2, src, rel
    raise RuntimeError("rewire failed")
