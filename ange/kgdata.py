"""E2: real-KG query tasks from BetaE-suite data already on disk.

Tasks (both multiple-choice, 10 candidates, letter answer):
  lp_mc    tail prediction (h, r, ?) from test.txt
  2hop_mc  2p queries (h, r1, ?x, r2, ?) from test-queries.pkl hard answers

Context = entity-centric subgraph around the query head (1-hop first, then
2-hop capped), re-encoded under each of the 8 encodings.
"""
from __future__ import annotations

import pickle
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

DATA = "/root/git/ULTRA/query-datasets/"


def _load(ds: str):
    base = DATA + ds + "/"
    ent2id = pickle.load(open(base + "ent2id.pkl", "rb"))
    rel2id = pickle.load(open(base + "rel2id.pkl", "rb"))
    id2ent = {v: k for k, v in ent2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}
    edges = []
    for line in open(base + "train.txt"):
        h, r, t = line.split()
        edges.append((int(h), int(r), int(t)))
    test = [tuple(map(int, l.split())) for l in open(base + "test.txt")]
    return id2ent, id2rel, edges, test


class KG:
    def __init__(self, ds: str):
        self.ds = ds
        self.id2ent, self.id2rel, self.edges, self.test = _load(ds)
        self.out_: Dict[int, List[Tuple[int, int]]] = defaultdict(list)   # h -> (r, t)
        self.in_: Dict[int, List[Tuple[int, int]]] = defaultdict(list)    # t -> (r, h)
        for h, r, t in self.edges:
            self.out_[h].append((r, t))
            self.in_[t].append((r, h))
        self.tails_by_rel = defaultdict(list)
        for h, r, t in self.edges:
            self.tails_by_rel[r].append(t)

    def n(self, eid: int) -> str:
        return self.id2ent[eid]

    def rel(self, rid: int) -> str:
        s = self.id2rel[rid]
        return s.replace("+", "f_").replace("-", "b_").replace("/", "_").replace(".", "_").replace(":", "_")

    def egocentric(self, seed: int, max_edges: int = 160) -> List[Tuple[int, int, int]]:
        seen = set()
        res = []
        frontier = [seed]
        visited = set()
        for hop in (1, 2):
            nxt = []
            for u in frontier:
                if u in visited:
                    continue
                visited.add(u)
                for r, t in self.out_.get(u, []):
                    e = (u, r, t)
                    if e not in seen:
                        seen.add(e)
                        res.append(e)
                        nxt.append(t)
                for r, h in self.in_.get(u, []):
                    e = (h, r, u)
                    if e not in seen:
                        seen.add(e)
                        res.append(e)
                        nxt.append(h)
            frontier = nxt
            if len(res) >= max_edges // 2:
                break
        random.Random(seed).shuffle(res)
        return res[:max_edges]


def to_mrtext(kg: KG, edges: List[Tuple[int, int, int]]):
    """ID triples -> text triples with readable relation names and node aliases.

    Entities get short aliases e0..ek (in order of first appearance) since FB
    MIDs are opaque; the alias map is part of the encoding (same for all)."""
    alias: Dict[int, str] = {}
    rel_alias: Dict[int, str] = {}
    out = []
    for h, r, t in edges:
        for x in (h, t):
            if x not in alias:
                alias[x] = f"e{len(alias)}"
        if r not in rel_alias:
            rel_alias[r] = kg.rel(r)
        out.append((alias[h], rel_alias[r], alias[t]))
    alias_names = {alias[x]: kg.n(x) for x in alias}
    return out, alias_names


def gen_lp_mc(kg: KG, n: int, seed: int = 5, deg_lo: int = 15, deg_hi: int = 140,
              n_cand: int = 10, max_edges: int = 160) -> List[dict]:
    """Tail prediction MC. Negative candidates: tails of the same relation
    (plausible distractors, not trivially wrong)."""
    rng = random.Random(seed)
    probes = []
    tests = kg.test[:]
    rng.shuffle(tests)
    for h, r, t in tests:
        if not (deg_lo <= len(kg.out_.get(h, [])) + len(kg.in_.get(h, [])) <= deg_hi):
            continue
        if t not in kg.id2ent:
            continue
        pool = [x for x in kg.tails_by_rel.get(r, []) if x != t and x in kg.id2ent]
        if len(pool) < n_cand - 1:
            continue
        negs = rng.sample(pool, n_cand - 1)
        cands = [t] + negs
        rng.shuffle(cands)
        gold_letter = "ABCDEFGHIJ"[cands.index(t)]
        edges = kg.egocentric(h, max_edges)
        probes.append({"kind": "lp_mc", "head": h, "rel": r, "gold_ent": t,
                       "cands": cands, "gold": gold_letter, "edges": edges})
        if len(probes) >= n:
            break
    return probes


def question_lp(kg: KG, p: dict) -> Tuple[str, str]:
    from .encoders import enc_pointer
    triples, alias_names = to_mrtext(kg, p["edges"])
    cand_lines = "\n".join(f"  {ltr}. {kg.n(c)}" for ltr, c in zip("ABCDEFGHIJ", p["cands"]))
    q = (f"Which entity is the correct target of the relation '{kg.rel(p['rel'])}' "
         f"starting from entity '{kg.n(p['head'])}' (shown as "
         f"'{p['head']}')?\nChoose exactly one letter among A-J.\nOptions:\n{cand_lines}\n"
         f"End your reply with: 'ANSWER: <letter>'.")
    return q, alias_names


DEMO_TRIPLES = [("e0", "located_in", "e1"), ("e2", "located_in", "e1"),
                ("e0", "works_with", "e2"), ("e1", "part_of", "e3")]
DEMO_ALIASES = {"e0": "alice", "e1": "paris", "e2": "bob", "e3": "france"}


def build_prompt(kg: KG, p: dict, encoding: str) -> str:
    triples, alias_names = to_mrtext(kg, p["edges"])
    g = _MiniGraph(triples, alias_names)
    from .encoders import ENCODINGS
    from .encoders import enc_pointer
    inv_alias = {name: a for a, name in alias_names.items()}
    head_alias = inv_alias.get(kg.n(p["head"]))
    rel_txt = kg.rel(p["rel"])
    if encoding == "pointer" and head_alias is not None:
        gt = enc_pointer(g, query=(head_alias, rel_txt))
    else:
        gt = ENCODINGS[encoding](g)
    hdr = ("You are given a subgraph of a knowledge graph. Nodes are e0..ek; the "
           "alias list after the graph gives their real names. Relation names are "
           "readable.\n\n")
    alias_tbl = "\nAliases:\n" + "\n".join(f"{a} = {n}" for a, n in alias_names.items())
    # 1-shot demo rendered in the SAME encoding
    dg = _MiniGraph(DEMO_TRIPLES, DEMO_ALIASES)
    if encoding == "pointer":
        dt = enc_pointer(dg, query=("e0", "works_with"))
    else:
        dt = ENCODINGS[encoding](dg)
    dtbl = "\nAliases:\n" + "\n".join(f"{a} = {n}" for a, n in DEMO_ALIASES.items())
    demo = ("Example:\n" + dt + dtbl + "\n\nQuestion: Which entity is the correct "
            "target of the relation 'works_with' starting from entity 'alice'?\n"
            "Choose exactly one letter among A-C.\nOptions:\n  A. paris\n  B. bob\n  C. france\n"
            "The 'works_with' edges from e0 (alice) point to e2 = bob.\nANSWER: B\n\n"
            "Now solve:\n\n")
    return hdr + demo + gt + alias_tbl + "\n\nQuestion: " + _qtext(kg, p)


def _qtext(kg: KG, p: dict) -> str:
    if p["kind"] == "lp_mc":
        cand_lines = "\n".join(f"  {ltr}. {kg.n(c)}" for ltr, c in zip("ABCDEFGHIJ", p["cands"]))
        return (f"Which entity is the correct target of the relation '{kg.rel(p['rel'])}' "
                f"starting from entity '{kg.n(p['head'])}'?\nChoose exactly one letter "
                f"among A-J.\nOptions:\n{cand_lines}\nEnd your reply with: 'ANSWER: <letter>'.")
    raise ValueError(p["kind"])


class _MiniGraph:
    """Adapter so the 8 encoders can consume text triples + aliases."""
    def __init__(self, triples, alias_names):
        self.edges = triples
        self.alias_names = alias_names
        self.rels = sorted({r for _, r, _ in triples})
        self.n_nodes = len(alias_names)

    def nodes(self):
        return sorted(self.alias_names, key=lambda s: int(s[1:]))


# patch encoders to use alias names via node_feats: simplest is monkeypatch-free
# -- we instead render nodes() as aliases and append an alias table line in
# encoders via MRGraph.node_feats. For simplicity we inject the alias table by
# wrapping: encoders already print nodes(); we add alias table as extra header.
def _alias_block(g):
    if not hasattr(g, "alias_names") or not g.alias_names:
        return ""
    return "\n".join(f"{a} = {name}" for a, name in g.alias_names.items())


def enc_with_aliases(base_fn, g):
    return base_fn(g) + ("\nAliases:\n" + _alias_block(g) if _alias_block(g) else "")
