#!/usr/bin/env python
"""P3: LoRA fine-tuning on real-KG (NELL) multiple-choice queries.

Train arms (7B): edge_list vs pointer2, identical data.
Test: E2 probe set (seed=5 heads) + a fresh seed=6 set; train heads disjoint.
"""
from __future__ import annotations

import random
from typing import List

from .kgdata import KG, gen_lp_mc, build_prompt, to_mrtext, _MiniGraph
from .encoders import ENCODINGS, enc_pointer2


def gold_cot(kg: KG, p: dict) -> str:
    """Short supervised reasoning: cite the query edge pattern, then answer.

    We cannot cite the test edge (it is held out), so the CoT names the
    relation and head and defers to the option letters -- supervision teaches
    format + lookup, mirroring what an in-context reasoner would do."""
    rel = kg.rel(p["rel"])
    head = kg.n(p["head"])
    gold_letter = p["gold"]
    gold_name = kg.n(p["gold_ent"])
    return (f"The target of relation '{rel}' starting from '{head}' is matched "
            f"by option {gold_letter} ('{gold_name}').\nANSWER: {gold_letter}")


def gen_train(kg: KG, n: int, seed: int = 77, avoid_heads: set = None,
              n_cand: int = 10, max_edges: int = 160) -> List[dict]:
    rng = random.Random(seed)
    avoid = avoid_heads or set()
    probes = []
    tests = kg.test[:]
    rng.shuffle(tests)
    for h, r, t in tests:
        if h in avoid or t not in kg.id2ent:
            continue
        pool = [x for x in kg.tails_by_rel.get(r, []) if x != t and x in kg.id2ent]
        if len(pool) < n_cand - 1:
            continue
        if not (15 <= len(kg.out_.get(h, [])) + len(kg.in_.get(h, [])) <= 140):
            continue
        negs = rng.sample(pool, n_cand - 1)
        cands = [t] + negs
        rng.shuffle(cands)
        probes.append({"kind": "lp_mc", "head": h, "rel": r, "gold_ent": t,
                       "cands": cands, "gold": "ABCDEFGHIJ"[cands.index(t)],
                       "edges": kg.egocentric(h, max_edges)})
        if len(probes) >= n:
            break
    return probes


def prompt_train(kg: KG, p: dict, encoding: str) -> str:
    """Same builder as build_prompt but with the demo (E2 eval uses demo too)."""
    from .kgdata import DEMO_TRIPLES, DEMO_ALIASES, _MiniGraph
    triples, alias_names = to_mrtext(kg, p["edges"])
    g = _MiniGraph(triples, alias_names)
    inv_alias = {name: a for a, name in alias_names.items()}
    head_alias = inv_alias.get(kg.n(p["head"]))
    rel_txt = kg.rel(p["rel"])
    if encoding == "pointer2" and head_alias is not None:
        gt = enc_pointer2(g, query=(head_alias, rel_txt))
    else:
        gt = ENCODINGS[encoding](g)
    alias_tbl = "\nAliases:\n" + "\n".join(f"{a} = {n2}" for a, n2 in alias_names.items())
    dg = _MiniGraph(DEMO_TRIPLES, DEMO_ALIASES)
    if encoding == "pointer2":
        dt = enc_pointer2(dg, query=("e0", "works_with"))
    else:
        dt = ENCODINGS[encoding](dg)
    dtbl = "\nAliases:\n" + "\n".join(f"{a} = {n2}" for a, n2 in DEMO_ALIASES.items())
    demo = ("Example:\n" + dt + dtbl + "\n\nQuestion: Which entity is the correct "
            "target of the relation 'works_with' starting from entity 'alice'?\n"
            "Choose exactly one letter among A-C.\nOptions:\n  A. paris\n  B. bob\n  C. france\n"
            "The 'works_with' edges from e0 (alice) point to e2 = bob.\nANSWER: B\n\n"
            "Now solve:\n\n")
    cand_lines = "\n".join(f"  {l}. {kg.n(c)}" for l, c in zip("ABCDEFGHIJ", p["cands"]))
    q = (f"Which entity is the correct target of the relation '{rel_txt}' "
         f"starting from entity '{kg.n(p['head'])}'?\nChoose exactly one letter "
         f"among A-J.\nOptions:\n{cand_lines}\nEnd your reply with: 'ANSWER: <letter>'.")
    hdr = ("You are given a subgraph of a knowledge graph. Nodes are e0..ek; the "
           "alias list after the graph gives their real names. Relation names are "
           "readable.\n\n")
    return hdr + demo + gt + alias_tbl + "\n\nQuestion: " + q
