#!/usr/bin/env python
"""P3 trainer: LoRA on NELL lp_mc for 7B (edge_list vs pointer2 arms)."""
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

from ange.kgdata import KG, gen_lp_mc
from ange.e3_real import gen_train, prompt_train, gold_cot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding", required=True, choices=["edge_list", "pointer2"])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--base", default="/mnt/ssd/ange/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="NELL-betae",
                    help="BetaE-suite KG name, e.g. NELL-betae / FB15k-237-betae")
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--maxlen", type=int, default=5376,
                    help="must cover the dataset's prompt lengths (FB needs ~8.7k)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    kg = KG(args.dataset)
    test_heads = {p["head"] for p in gen_lp_mc(kg, 100, seed=5)} | \
                 {p["head"] for p in gen_lp_mc(kg, 100, seed=6)}
    train = gen_train(kg, args.n_train, avoid_heads=test_heads)
    print(f"[data] {len(train)} train probes (heads disjoint from test)")

    feats = []
    for p in train:
        up = prompt_train(kg, p, args.encoding)
        ans = gold_cot(kg, p)
        prefix_txt = tok.apply_chat_template([{"role": "user", "content": up}],
                                             tokenize=False, add_generation_prompt=True)
        full_txt = tok.apply_chat_template(
            [{"role": "user", "content": up}, {"role": "assistant", "content": ans}],
            tokenize=False)
        prefix = tok(prefix_txt, add_special_tokens=False)["input_ids"]
        full = tok(full_txt, add_special_tokens=False)["input_ids"]
        MAXLEN = args.maxlen
        full = full[:MAXLEN]
        labels = [-100] * min(len(prefix), len(full)) + full[len(prefix):]
        # drop samples whose supervision got truncated away (no positive label)
        if all(l == -100 for l in labels):
            continue
        feats.append({"input_ids": full, "labels": labels[:MAXLEN], "n": len(full)})
    print(f"[data] kept {len(feats)}/{len(train)} (dropped {len(train)-len(feats)} "
          f"truncated-supervision samples)")
    lens = sorted(f["n"] for f in feats)
    print(f"[data] len p50/p95/max = {lens[len(lens)//2]}/{lens[int(len(lens)*0.95)]}/{lens[-1]}")

    from ange.train_lora import Collator
    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs, learning_rate=1e-4,
        warmup_steps=30, lr_scheduler_type="cosine", logging_steps=20,
        save_strategy="no", bf16=True, report_to=[], remove_unused_columns=False,
        gradient_checkpointing=True)
    trainer = Trainer(model=model, args=targs, train_dataset=feats,
                      data_collator=Collator(tok))
    trainer.train()
    model.save_pretrained(os.path.join(args.out, "adapter"))
    tok.save_pretrained(os.path.join(args.out, "adapter"))
    print("saved", os.path.join(args.out, "adapter"))


if __name__ == "__main__":
    main()
