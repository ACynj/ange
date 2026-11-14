#!/usr/bin/env python
"""E-A2: motif ladder — generalize the 2-PG boundary lift beyond triangles.

Finding from run_wl_family.py: TRI-only records solve C6 vs 2xC3 (k=3) but
not C8 vs 2xC4 (k=4) or C10 vs 2xC5 (k=5) — those families are triangle-free.
This script adds SQR (4-cycle) and PEN (5-cycle) partner records, trains ONE
7B LoRA arm on mixed iso data covering all three families, and re-tests.
Expected: the ladder lifts each family whose motif record is present,
demonstrating the grammar is extensible by record type.
"""
from __future__ import annotations
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from ange.graph import MRGraph, gen_graph, isomorphic_pair, noniso_pair_wl_distinguishable
from ange.encoders import _node_line, _rel_line
from ange.encoders import enc_pointer2pg as _orig_2pg
from ange.e3_data import HEADER, FOOT_BOOL, ISO_Q
from ange.engine import parse_answer, score

B7 = "/mnt/ssd/ange/models/Qwen2.5-7B-Instruct"
R = "/mnt/ssd/ange/results"
RUN = "/mnt/ssd/ange/runs/fu_2pgx_7b"


# ---------------- motif records ----------------

def cycle_partners(g: MRGraph, k: int):
    """For each node, partner sets closing a k-cycle through it (undirected,
    any relation, simple cycles of exact length k)."""
    from collections import defaultdict
    und = defaultdict(set)
    for s, r, t in g.edges:
        und[s].add(t); und[t].add(s)
    partners = defaultdict(set)
    nodes = sorted(und)

    def dfs(path, start):
        if len(path) == k:
            if start in und[path[-1]] and len(set(path)) == k:
                partners[start].update(x for x in path if x != start)
            return
        for nxt in und[path[-1]]:
            if nxt == start and len(path) < k:
                continue
            if nxt in path:
                continue
            if nxt < start:            # canonical: min node = anchor
                continue
            dfs(path + [nxt], start)

    for v in nodes:
        dfs([v], v)
    return partners


def enc_2pgx(g: MRGraph, gb: MRGraph | None = None, query=None) -> str:
    """2-PG + motif ladder records (TRI / SQR / PEN)."""
    lines = [_node_line(g), _rel_line(g), "Edges (u -rel-> v):"]
    lines += [f"{s} -{r}-> {t}" for s, r, t in g.edges]
    tri = cycle_partners(g, 3)
    lines.append("Triangles (TRI v -> partners):")
    lines += [f"TRI {v} -> {{{', '.join(sorted(tri[v]))}}}" for v in sorted(tri)]
    sqr = cycle_partners(g, 4)
    lines.append("Squares (SQR v -> partners):")
    lines += [f"SQR {v} -> {{{', '.join(sorted(sqr[v]))}}}" for v in sorted(sqr)]
    pen = cycle_partners(g, 5)
    lines.append("Pentagons (PEN v -> partners):")
    lines += [f"PEN {v} -> {{{', '.join(sorted(pen[v]))}}}" for v in sorted(pen)]
    return "\n".join(lines)


def prompt_iso(ga, gb):
    ta = enc_2pgx(ga); tb = enc_2pgx(gb)
    return HEADER + f"GRAPH A:\n{ta}\n\nGRAPH B:\n{tb}\n\nQuestion: " + ISO_Q + FOOT_BOOL


# ---------------- data ----------------

def cycle_edges(k, offset=0):
    nodes = [f"n{offset+i}" for i in range(k)]
    return [(nodes[i], "r1", nodes[(i+1) % k]) for i in range(k)]


def mr(edges, n):
    return MRGraph(n, ["r1"], list(edges))


def wl_pair(k, seed):
    ga = mr(cycle_edges(2*k), 2*k)
    gb = mr(cycle_edges(k, 100) + cycle_edges(k, 100+k), 2*k)
    rng = random.Random(seed)
    rng.shuffle(ga.edges); rng.shuffle(gb.edges)
    return ga, gb


def gen_train(n_samples=1200, seed=7):
    """Mixed iso supervision: true isomorphs, WL-separable pairs, and the
    three ladder families (k=3,4,5), gold NO with a motif-count rationale."""
    rng = random.Random(seed)
    out = []
    for i in range(n_samples):
        fam = i % 4
        if fam == 0:
            ga, gb = isomorphic_pair(rng.randrange(10**6), n=rng.choice([10, 12]))
            gold = "YES"
            cot = ("Renaming nodes of B maps every edge of A one-to-one; TRI/SQR/PEN "
                   "partner multisets agree.\nANSWER: YES")
        elif fam == 1:
            got = noniso_pair_wl_distinguishable(rng.randrange(10**6), n=16)
            if got is None:
                continue
            ga, gb, _ = got
            gold = "NO"
            cot = ("A directed 3-cycle motif in A has no counterpart in B.\nANSWER: NO")
        else:
            k = 2 + fam  # fam 2->k=3, fam 3->k=4 ... rotate k in {3,4,5}
            k = {2: 3, 3: 4, 0: 5, 1: 5}[fam] if False else (3 if fam == 2 else (4 if fam == 3 else 5))
            ga, gb = wl_pair(k, rng.randrange(10**6))
            gold = "NO"
            tag = {3: "TRI", 4: "SQR", 5: "PEN"}[k]
            cot = (f"Graph B is two disjoint {k}-cycles, so its {tag} partner records "
                   f"are non-empty; graph A is one {2*k}-cycle with no {k}-cycles, "
                   f"so its {tag} records are empty. Different motif multisets.\nANSWER: NO")
        out.append({"ga": ga, "gb": gb, "gold": gold, "cot": cot})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "eval", "both"], default="both")
    args = ap.parse_args()

    if args.phase in ("train", "both"):
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  Trainer, TrainingArguments)
        from peft import LoraConfig, get_peft_model
        tok = AutoTokenizer.from_pretrained(B7)
        model = AutoModelForCausalLM.from_pretrained(
            B7, dtype=torch.bfloat16, device_map="cuda")
        model.gradient_checkpointing_enable(); model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        data = gen_train(1200)
        feats = []
        for s in data:
            pr = prompt_iso(s["ga"], s["gb"])
            full = tok.apply_chat_template(
                [{"role": "user", "content": pr},
                 {"role": "assistant", "content": s["cot"]}],
                tokenize=False, add_generation_prompt=False)
            prefix = tok.apply_chat_template(
                [{"role": "user", "content": pr}],
                tokenize=False, add_generation_prompt=True)
            ids = tok(full, add_special_tokens=False)["input_ids"][:3072]
            pre = tok(prefix, add_special_tokens=False)["input_ids"]
            labels = [-100]*len(pre) + ids[len(pre):]
            feats.append({"input_ids": ids[:3072], "labels": labels[:3072]})
        from ange.train_lora import Collator
        targs = TrainingArguments(
            output_dir=RUN, num_train_epochs=2.0, per_device_train_batch_size=2,
            learning_rate=1e-4, warmup_steps=30, lr_scheduler_type="cosine",
            logging_steps=20, save_strategy="no", bf16=True, report_to=[],
            remove_unused_columns=False, gradient_checkpointing=True)
        Trainer(model=model, args=targs, train_dataset=feats,
                data_collator=Collator(tok)).train()
        model.save_pretrained(os.path.join(RUN, "adapter"))
        tok.save_pretrained(os.path.join(RUN, "adapter"))
        print("saved", RUN, flush=True)

    if args.phase in ("eval", "both"):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        llm = LLM(model=B7, dtype="bfloat16", max_model_len=8192,
                  gpu_memory_utilization=0.42, enable_lora=True, max_lora_rank=16)
        sp = SamplingParams(temperature=0.0, max_tokens=400)
        adp = os.path.join(RUN, "adapter")
        out = open(f"{R}/motif_ladder.jsonl", "w")
        for k in (3, 4, 5):
            probes = [wl_pair(k, 2000+s) for s in range(20)]
            for enc_tag, req in (("2pgx_lora", LoRARequest("x", 1, adp)),
                                 ("2pgx_frozen", None)):
                prompts = [prompt_iso(ga, gb) for ga, gb in probes]
                outs = llm.chat([[{"role": "user", "content": p}] for p in prompts],
                                sp, lora_request=req)
                hits = 0
                for (ga, gb), o in zip(probes, outs):
                    pred = parse_answer(o.outputs[0].text, "bool")
                    sc = score(pred, "NO", "bool")
                    hits += sc
                    out.write(json.dumps({"k": k, "enc": enc_tag, "gold": "NO",
                                          "pred": pred, "score": sc}) + "\n")
                out.flush()
                print(f"[k={k} {enc_tag}] acc = {hits/20:.2f}", flush=True)
        out.close()
        print("LADDER DONE")


if __name__ == "__main__":
    main()
