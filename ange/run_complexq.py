#!/usr/bin/env python
"""E-B: complex-query probes (limitation ④: beyond nbr/reach/iso).

Three new families on n=20 multi-relational graphs, frozen Qwen-7B,
encodings {edge_list, pointer2, pointer2pg}:
  join  : common 1-hop targets of (u1,r) and (u2,r)   (conjunctive query)
  neg   : is v NOT reachable from u in <=k steps along r (negated reach)
  count : number of r-targets of u                     (aggregation)
20 probes x 3 tasks x 3 encodings, temperature 0.
"""
from __future__ import annotations
import json, random, sys
sys.path.insert(0, '/mnt/ssd/ange/code')

from ange.graph import gen_graph
from ange.encoders import ENCODINGS, enc_pointer2, enc_pointer2pg
from ange.probes import HEADER, FOOT_BOOL, FOOT_LIST
from ange.engine import parse_answer, score

B7 = "/mnt/ssd/ange/models/Qwen2.5-7B-Instruct"
R = "/mnt/ssd/ange/results"


def render(g, enc):
    if enc == "pointer2":
        return enc_pointer2(g)
    if enc == "pointer2pg":
        return enc_pointer2pg(g)
    return ENCODINGS[enc](g)


def gen_probes(n_per=20):
    rng = random.Random(99)
    probes = []
    # join
    got = 0
    while got < n_per:
        g = gen_graph(900 + got, 20, 4, 1.4)
        u1, u2 = rng.sample(g.nodes(), 2)
        r = rng.choice(g.rels)
        common = set(g.out_neighbors(u1, r)) & set(g.out_neighbors(u2, r))
        if not (1 <= len(common) <= 4):
            continue
        q = (f"List ALL nodes that are targets of {r} edges from BOTH node "
             f"{u1} AND node {u2}. ")
        probes.append({"task": "join", "q": q, "gold": ",".join(sorted(common)),
                       "g": g, "parse": "list"})
        got += 1
    # neg
    got = 0
    while got < n_per:
        g = gen_graph(950 + got, 20, 4, 1.4)
        u = rng.choice(g.nodes()); r = rng.choice(g.rels)
        k = rng.choice([1, 2, 3])
        reach = g.reach(u, r, k)
        pool = sorted(x for x in g.nodes() if x != u and x not in reach)
        want_no = got % 2 == 0
        if want_no and not pool:
            continue
        if not want_no:
            pool = sorted(reach)
            if not pool:
                continue
        v = rng.choice(pool)
        q = (f"Starting from node {u} and following only {r} edges forward, "
             f"is node {v} NOT reachable in at most {k} steps? "
             f"Reply with exactly one word, YES or NO. ")
        probes.append({"task": "neg", "q": q,
                       "gold": "YES" if want_no else "NO",
                       "g": g, "parse": "bool"})
        got += 1
    # count
    got = 0
    while got < n_per:
        g = gen_graph(980 + got, 20, 4, 1.4)
        u = rng.choice(g.nodes()); r = rng.choice(g.rels)
        d = len(g.out_neighbors(u, r))
        if not (2 <= d <= 5):
            continue
        q = (f"How many nodes are targets of {r} edges starting from node "
             f"{u}? End your reply with 'ANSWER: <number>'. ")
        probes.append({"task": "count", "q": q, "gold": str(d),
                       "g": g, "parse": "list"})
        got += 1
    return probes


def main():
    from vllm import LLM, SamplingParams
    llm = LLM(model=B7, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.6)
    sp = SamplingParams(temperature=0.0, max_tokens=120)
    probes = gen_probes(20)
    print(f"[data] {len(probes)} complex probes", flush=True)
    out = open(f"{R}/complex_queries2.jsonl", "w")
    for enc in ("edge_list", "pointer2", "pointer2pg"):
        for p in probes:
            gt = render(p["g"], enc)
            foot = FOOT_BOOL if p["parse"] == "bool" else FOOT_LIST
            prompt = HEADER + gt + "\n\nQuestion: " + p["q"] + foot
            o = llm.chat([[{"role": "user", "content": prompt}]], sp)[0]
            txt = o.outputs[0].text
            pred = parse_answer(txt, p["parse"])
            if p["task"] == "count":
                sc = 1.0 if pred.strip() == p["gold"] else 0.0
            else:
                sc = score(pred, p["gold"], p["parse"])
            out.write(json.dumps({"enc": enc, "task": p["task"],
                                  "gold": p["gold"], "pred": pred,
                                  "score": sc, "raw": txt[:150]}) + "\n")
        out.flush()
        print(f"[{enc}] done", flush=True)
    out.close()
    from collections import defaultdict
    acc = defaultdict(list)
    for line in open(f"{R}/complex_queries2.jsonl"):
        r = json.loads(line)
        acc[(r["enc"], r["task"])].append(r["score"])
    print("\n== complex queries (frozen 7B) ==")
    for enc in ("edge_list", "pointer2", "pointer2pg"):
        row = []
        for t in ("join", "neg", "count"):
            v = acc.get((enc, t))
            row.append(f"{t}={sum(v)/len(v):.2f}" if v else f"{t}=--")
        print(f"  {enc:12s} " + " ".join(row))
    print("DONE")


if __name__ == "__main__":
    main()
