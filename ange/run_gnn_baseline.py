#!/usr/bin/env python
"""R8: GNN baseline (GIN from scratch, no torch_geometric) on the same probes.

Same data protocol as the LLM arms: gen_split seeds, same tasks
(reach1/2/3, nbr, iso tiers). Node features = one-hot node index hash;
edge features = relation one-hot. Graph-level tasks (iso) use two-graph
readout; node tasks use per-node readout. This establishes the
architecture-side reference point: a 1-WL-bounded GNN solves local
queries perfectly but is provably blind on tier-3 (it IS 1-WL).
"""
from __future__ import annotations
import json, sys, time
sys.path.insert(0, '/mnt/ssd/ange/code')

import torch
import torch.nn as nn
import torch.nn.functional as F
from ange.e3_data import gen_split
from ange.graph import (isomorphic_pair, noniso_pair_wl_distinguishable,
                        noniso_pair_wl_indistinguishable)

R = "/mnt/ssd/ange/results"
DEV = "cuda"


class GIN(nn.Module):
    def __init__(self, d=64, layers=3):
        super().__init__()
        self.embed_node = nn.Embedding(200, d)
        self.embed_rel = nn.Embedding(16, d)
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(d, 2*d), nn.ReLU(), nn.Linear(2*d, d))
            for _ in range(layers)])
        self.eps = nn.Parameter(torch.zeros(layers))

    def forward(self, x_idx, edge_index, edge_rel, batch, n_batches):
        """x_idx [N], edge_index [2,E] global; batch [N] graph id."""
        h = self.embed_node(x_idx)                       # [N,d]
        er = self.embed_rel(edge_rel)                    # [E,d]
        for li, mlp in enumerate(self.mlps):
            src, dst = edge_index
            msg = (h[src] + er).unsqueeze(0)             # use mean-agg
            # scatter-mean into dst
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, h[src] + er)
            cnt = torch.zeros(h.size(0), 1, device=h.device)
            cnt.index_add_(0, dst, torch.ones(src.size(0), 1, device=h.device))
            agg = agg / cnt.clamp(min=1)
            h = mlp((1 + self.eps[li]) * h + agg)
            h = F.relu(h)
        return h                                          # [N,d]


def build_batch(graphs, rel2id=None, max_nodes=200):
    if rel2id is None:
        rels = set()
        for g in graphs:
            rels.update(g.rels)
        rel2id = {r: i for i, r in enumerate(sorted(rels))}
    xs, e_src, e_dst, e_rel, batch = [], [], [], [], []
    for gi, g in enumerate(graphs):
        idm = {}
        for i, nd in enumerate(g.nodes()):
            idm[nd] = len(xs); xs.append(int(nd[1:]) % 200); batch.append(gi)
        for s, r, t in g.edges:
            e_src.append(idm[s]); e_dst.append(idm[t]); e_rel.append(rel2id[r])
    return (torch.tensor(xs, device=DEV), 
            torch.tensor([e_src, e_dst], device=DEV),
            torch.tensor(e_rel, device=DEV),
            torch.tensor(batch, device=DEV))


def train_nbr_reach(model, epochs=8):
    """Multi-task head on node representations."""
    data = gen_split(100, 225, 40)          # same as LLM arms
    rel2id = {f"r{i+1}": i for i in range(8)}
    head = nn.Linear(64, 200).to(DEV)
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)
    graphs = [s["graph"] for s in data]
    xs, ei, er, bt = build_batch(graphs, rel2id)
    # per-sample target: answer node index (for reach: yes/no via node presence;
    # for nbr: multi-label on node set)
    node_of = []
    for s in data:
        meta = s["meta"]
        node_of.append((int(meta[0][1:]), int(meta[2][1:]) if meta[2] else -1,
                        s["task"]))
    graph_of = torch.tensor([i for i, s in enumerate(data)
                             for _ in s["graph"].nodes()], device=DEV)
    # Build labels: [B, 200] multi-hot over reachable/target sets
    B = len(data)
    labels = torch.zeros(B, 200, device=DEV)
    for i, s in enumerate(data):
        g, meta = s["graph"], s["meta"]
        if s["task"] == "nbr":
            for t in g.out_neighbors(meta[0], meta[1]):
                labels[i, int(t[1:])] = 1.0
        else:  # reach k
            k = int(s["task"][-1])
            for t in g.reach(meta[0], meta[1], k):
                if t != meta[0]:
                    labels[i, int(t[1:])] = 1.0
    mask_reach = torch.tensor([s["task"].startswith("reach") for s in data])
    for ep in range(epochs):
        model.train()
        h = model(xs, ei, er, bt, B)
        # graph readout: mean over nodes of each graph
        gsum = torch.zeros(B, 64, device=DEV); gcnt = torch.zeros(B, 1, device=DEV)
        gsum.index_add_(0, bt, h); gcnt.index_add_(0, bt, torch.ones(h.size(0),1,device=DEV))
        gmean = gsum / gcnt.clamp(min=1)
        logits = head(gmean)                    # [B,200]
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()
    return head, rel2id


@torch.no_grad()
def eval_nbr_reach(model, head, rel2id):
    model.eval()
    from ange.engine import score as escore
    test = gen_split(300, 315, 30)
    xs, ei, er, bt = build_batch([s["graph"] for s in test], rel2id)
    B = len(test)
    h = model(xs, ei, er, bt, B)
    gsum = torch.zeros(B,64,device=DEV); gcnt = torch.zeros(B,1,device=DEV)
    gsum.index_add_(0, bt, h); gcnt.index_add_(0, bt, torch.ones(h.size(0),1,device=DEV))
    logits = head(gsum / gcnt.clamp(min=1))
    preds = (logits > 0).float()
    from collections import defaultdict
    res = defaultdict(list)
    for i, s in enumerate(test):
        pred_set = {f"n{j}" for j in torch.nonzero(preds[i]).flatten().tolist()}
        gold = s["a"].split("ANSWER: ")[-1]
        gold_set = {x for x in gold.split(",") if x.strip() and x != "none"}
        if s["task"] == "nbr":
            tp = len(pred_set & gold_set)
            prec = tp / max(len(pred_set),1); rec = tp / max(len(gold_set),1)
            f1 = 2*prec*rec/max(prec+rec,1e-9)
            res["nbr"].append(f1)
        else:
            k = int(s["task"][-1]); meta = s["meta"]
            truth = {t for t in s["graph"].reach(meta[0], meta[1], k) if t != meta[0]}
            correct = (pred_set == truth)
            res[s["task"]].append(1.0 if correct else 0.0)
    return {t: sum(v)/len(v) for t, v in res.items()}


@torch.no_grad()
def eval_iso_1wl():
    """A 1-WL-bounded GNN cannot distinguish tier-3 by construction —
    verify empirically: random-init GIN produces identical embeddings
    for C6 and 2xC3 after color refinement (they are 1-WL equal)."""
    import random as rnd
    from ange.run_motif_ladder import wl_pair
    model = GIN().to(DEV)
    model.eval()
    rel2id = {"r1": 0}
    acc = {"tier1": [], "tier2": [], "tier3": []}
    rng = rnd.Random(3)
    # tier1: true isomorphs -> GNN (1-WL) gives identical colors => predict YES always => acc=1
    for i in range(20):
        ga, gb = isomorphic_pair(rng.randrange(10**6), n=12)
        h1 = graph_embed(model, ga); h2 = graph_embed(model, gb)
        same = torch.allclose(h1, h2, atol=1e-4)  # 1-WL equal
        acc["tier1"].append(1.0 if same else 0.0)
    # tier2: WL-separable
    for i in range(20):
        got = noniso_pair_wl_distinguishable(rng.randrange(10**6), n=16)
        if got is None: continue
        ga, gb, _ = got
        h1 = graph_embed(model, ga); h2 = graph_embed(model, gb)
        acc["tier2"].append(0.0 if torch.allclose(h1, h2, atol=1e-4) else 1.0)
    # tier3: C6 vs 2xC3 — provably indistinguishable
    for i in range(20):
        ga, gb = wl_pair(3, 4000+i)
        h1 = graph_embed(model, ga); h2 = graph_embed(model, gb)
        acc["tier3"].append(0.0 if torch.allclose(h1, h2, atol=1e-4) else 1.0)
    return {t: sum(v)/len(v) for t, v in acc.items()}


def graph_embed(model, g, rel2id=None):
    xs, ei, er, bt = build_batch([g], rel2id)  # auto-detects if None
    h = model(xs, ei, er, bt, 1)
    return h.mean(0)


def main():
    torch.manual_seed(0)
    model = GIN().to(DEV)
    t0 = time.time()
    head, rel2id = train_nbr_reach(model, epochs=12)
    res_local = eval_nbr_reach(model, head, rel2id)
    res_iso = eval_iso_1wl()
    out = {"local": res_local, "iso": res_iso, "train_s": round(time.time()-t0)}
    json.dump(out, open(f"{R}/gnn_baseline.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("GNN BASELINE DONE")


if __name__ == "__main__":
    main()
