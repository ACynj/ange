#!/usr/bin/env python
"""E3: LoRA SFT of Qwen2.5-1.5B on encoded graph probes (one arm per encoding).

Usage:
  python -m ange.train_lora --encoding pointer --gpu 0 \
      --out /mnt/ssd/ange/runs/e3_pointer
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

from ange.e3_data import gen_split, make_chat

BASE = "/mnt/ssd/ange/models/Qwen2.5-1.5B-Instruct"


def build_dataset(tok, chats, max_len=2048):
    features = []
    for ch in chats:
        msgs = ch["messages"]
        prefix_txt = tok.apply_chat_template(
            [msgs[0]], tokenize=False, add_generation_prompt=True)
        full_txt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False)
        prefix = tok(prefix_txt, add_special_tokens=False)["input_ids"]
        full = tok(full_txt, add_special_tokens=False)["input_ids"]
        input_ids = full[:max_len]
        labels = [-100] * min(len(prefix), len(input_ids)) + input_ids[len(prefix):]
        labels = labels[:max_len]
        # drop the trailing <|im_end|>-adjacent pad if tokenizer appended
        features.append({"input_ids": input_ids, "labels": labels,
                         "task": ch["task"], "n": len(input_ids)})
    lens = [f["n"] for f in features]
    print(f"[data] {len(features)} samples, len p50/p95/max = "
          f"{sorted(lens)[len(lens)//2]}/{sorted(lens)[int(len(lens)*0.95)]}/{max(lens)}")
    return features


class Collator:
    def __init__(self, tok):
        self.pad = tok.pad_token_id or tok.eos_token_id

    def __call__(self, feats):
        import torch as T
        mx = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            ids, lab = f["input_ids"], f["labels"]
            pad = mx - len(ids)
            input_ids.append(ids + [self.pad] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {"input_ids": T.tensor(input_ids), "labels": T.tensor(labels),
                "attention_mask": T.tensor(attn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding", required=True, choices=["pointer", "pointer2", "edge_list", "sog"])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_graphs", type=int, default=125)
    ap.add_argument("--per_graph", type=int, default=40)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--data_seed", type=int, default=100)
    ap.add_argument("--init_seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    import torch as _t
    _t.manual_seed(args.init_seed)

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

    split = gen_split(args.data_seed, args.data_seed + args.n_graphs, args.per_graph)
    chats = make_chat(split, args.encoding)
    feats = build_dataset(tok, chats)

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs, learning_rate=args.lr,
        warmup_steps=30, lr_scheduler_type="cosine", logging_steps=20,
        save_strategy="no", bf16=True, report_to=[], remove_unused_columns=False,
        gradient_checkpointing=True)
    trainer = Trainer(model=model, args=targs, train_dataset=feats,
                      data_collator=Collator(tok))
    trainer.train()
    model.save_pretrained(os.path.join(args.out, "adapter"))
    tok.save_pretrained(os.path.join(args.out, "adapter"))
    json.dump({"encoding": args.encoding, "data_seed": args.data_seed,
               "init_seed": args.init_seed, "n_train": len(feats),
               "epochs": args.epochs, "lr": args.lr, "bs": args.bs},
              open(os.path.join(args.out, "meta.json"), "w"))
    print("saved to", os.path.join(args.out, "adapter"))


if __name__ == "__main__":
    main()
