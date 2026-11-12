#!/usr/bin/env python
"""Follow-up trainers.

mode=2pg      : 7B/1.5B on pointer2pg + iso training (goal: break C6 vs 2C3)
mode=frontier : 7B on pointer2 with frontier-trace gold (goal: accuracy at
                1/7 the tokens of free CoT)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)
from peft import LoraConfig, get_peft_model

from ange.e3_data import gen_split, gen_split_with_iso, build_prompt, build_prompt_iso
from ange.encoders import enc_pointer2pg
from ange.graph import gen_graph


def frontier_gold(g, src, rel, k, gold):
    trace = frontier_trace_of(g, src, rel, k)
    steps = "\n".join(f"Step {i+1}: {{{', '.join(fs)}}}" for i, fs in enumerate(trace) if fs)
    verdict = "reachable" if gold == "YES" else "not reachable"
    return (steps + f"\nTarget is {'among' if gold=='YES' else 'not among'} the "
            f"visited nodes ({verdict} within {k} steps).\nANSWER: {gold}")


def frontier_trace_of(g, src, rel, k):
    frontier, seen, trace = {src}, {src}, []
    for _ in range(k):
        nxt = set()
        for u in frontier:
            for s, r, t in g.edges:
                if s == u and r == rel and t not in seen:
                    nxt.add(t)
        seen |= nxt
        frontier = nxt
        trace.append(sorted(nxt))
    return trace


def build_feats(tok, chats, max_len):
    feats = []
    for ch in chats:
        prefix_txt = tok.apply_chat_template([ch["messages"][0]], tokenize=False,
                                             add_generation_prompt=True)
        full_txt = tok.apply_chat_template(ch["messages"], tokenize=False)
        prefix = tok(prefix_txt, add_special_tokens=False)["input_ids"]
        full = tok(full_txt, add_special_tokens=False)["input_ids"][:max_len]
        labels = [-100] * min(len(prefix), len(full)) + full[len(prefix):]
        if all(x == -100 for x in labels[:max_len]):
            continue
        feats.append({"input_ids": full, "labels": labels[:max_len], "n": len(full)})
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["2pg", "frontier"])
    ap.add_argument("--base", default="/mnt/ssd/ange/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=3.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()

    if args.mode == "2pg":
        split = gen_split_with_iso(100, 225, 40)
        chats = []
        for s in split:
            if s["task"].startswith("iso"):
                up = build_prompt_iso(s["graph"], s["graph_b"], "pointer2pg")
            else:
                up = build_prompt(s["graph"], s["q"], s["task"], "pointer2pg", s["meta"])
            chats.append({"messages": [{"role": "user", "content": up},
                                       {"role": "assistant", "content": s["a"]}],
                          "task": s["task"]})
        max_len = 2048
    else:  # frontier
        split = gen_split(100, 225, 40)
        chats = []
        for s in split:
            up = build_prompt(s["graph"], s["q"], s["task"], "pointer2", s["meta"])
            if s["task"].startswith("reach"):
                k = int(s["task"][5]); src, rel, tgt = s["meta"]
                gold = "YES" if s["a"].endswith("YES") else "NO"
                ans = frontier_gold(s["graph"], src, rel, k, gold)
            else:
                ans = s["a"]
            chats.append({"messages": [{"role": "user", "content": up},
                                       {"role": "assistant", "content": ans}],
                          "task": s["task"]})
        max_len = 2048

    feats = build_feats(tok, chats, max_len)
    lens = sorted(f["n"] for f in feats)
    print(f"[data] {len(feats)} samples, p50/p95/max = {lens[len(lens)//2]}/"
          f"{lens[int(len(lens)*0.95)]}/{lens[-1]}")

    from ange.train_lora import Collator
    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs, learning_rate=1e-4,
        warmup_steps=30, lr_scheduler_type="cosine", logging_steps=20,
        save_strategy="no", bf16=True, report_to=[], remove_unused_columns=False,
        gradient_checkpointing=True)
    Trainer(model=model, args=targs, train_dataset=feats,
            data_collator=Collator(tok)).train()
    model.save_pretrained(os.path.join(args.out, "adapter"))
    tok.save_pretrained(os.path.join(args.out, "adapter"))
    json.dump({"mode": args.mode, "n": len(feats)}, open(os.path.join(args.out, "meta.json"), "w"))
    print("saved", os.path.join(args.out, "adapter"))


if __name__ == "__main__":
    main()
