#!/usr/bin/env python
"""补救1: 统一读者臂 —— 混合训练消除专用臂的不迁移。

训练数据 = 局部任务(reach/nbr) + iso 三族(k=3,4,5 motif 家族) + 真/分离对,
编码 = 2-PGx (TRI+SQR+PEN 记录)。目标: 一个臂同时拿到
iso>=0.95 且 reach1>=0.75 且 nbr>=0.45, 无任何任务回归。
"""
from __future__ import annotations
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from ange.graph import gen_graph, isomorphic_pair, noniso_pair_wl_distinguishable
from ange.e3_data import gen_split, HEADER, FOOT_BOOL, FOOT_LIST, ISO_Q
from ange.run_motif_ladder import enc_2pgx, wl_pair
from ange.engine import parse_answer, score

B7 = "/mnt/ssd/ange/models/Qwen2.5-7B-Instruct"
RUN = "/mnt/ssd/ange/runs/fu_unified_7b"
R = "/mnt/ssd/ange/results"


def build_prompt_local(g, q, task):
    gt = enc_2pgx(g)
    foot = FOOT_BOOL if task.startswith("reach") else FOOT_LIST
    return HEADER + gt + "\n\nQuestion: " + q + foot


def gold_local(s):
    return s["a"]


def main_train():
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)
    from peft import LoraConfig, get_peft_model
    from ange.train_lora import Collator
    tok = AutoTokenizer.from_pretrained(B7)
    model = AutoModelForCausalLM.from_pretrained(B7, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    rng = random.Random(11)
    chats = []
    # 1) local tasks: 600 samples from e3-style split
    local = gen_split(400, 415, 40, n=20)
    rng.shuffle(local)
    for s in local[:600]:
        chats.append((build_prompt_local(s["graph"], s["q"], s["task"]),
                      gold_local(s)))
    # 2) iso families: 3 x 150
    for k in (3, 4, 5):
        for i in range(150):
            ga, gb = wl_pair(k, rng.randrange(10**6))
            tag = {3: "TRI", 4: "SQR", 5: "PEN"}[k]
            cot = (f"Graph B is two disjoint {k}-cycles, so its {tag} records are "
                   f"non-empty; graph A is one {2*k}-cycle with no {k}-cycles, so "
                   f"its {tag} records are empty.\nANSWER: NO")
            chats.append((iso_prompt(ga, gb), cot))
    # 3) true isomorphs + WL-separable: 150 + 150
    for i in range(150):
        ga, gb = isomorphic_pair(rng.randrange(10**6), n=12)
        chats.append((iso_prompt(ga, gb),
                      "Renaming nodes of B maps every edge of A one-to-one; motif "
                      "records agree.\nANSWER: YES"))
    for i in range(150):
        got = noniso_pair_wl_distinguishable(rng.randrange(10**6), n=16)
        if got is None: continue
        ga, gb, _ = got
        chats.append((iso_prompt(ga, gb),
                      "A has a directed 3-cycle motif with no counterpart in B.\nANSWER: NO"))
    rng.shuffle(chats)
    print(f"[data] {len(chats)} mixed samples", flush=True)
    feats = []
    for pr, gold in chats:
        full = tok.apply_chat_template(
            [{"role": "user", "content": pr}, {"role": "assistant", "content": gold}],
            tokenize=False, add_generation_prompt=False)
        prefix = tok.apply_chat_template(
            [{"role": "user", "content": pr}], tokenize=False,
            add_generation_prompt=True)
        ids = tok(full, add_special_tokens=False)["input_ids"][:3072]
        pre = tok(prefix, add_special_tokens=False)["input_ids"]
        labels = [-100]*len(pre) + ids[len(pre):]
        feats.append({"input_ids": ids[:3072], "labels": labels[:3072]})
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


def iso_prompt(ga, gb):
    ta, tb = enc_2pgx(ga), enc_2pgx(gb)
    return HEADER + f"GRAPH A:\n{ta}\n\nGRAPH B:\n{tb}\n\nQuestion: " + ISO_Q + FOOT_BOOL


def main_eval():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=B7, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.42, enable_lora=True, max_lora_rank=16)
    sp = SamplingParams(temperature=0.0, max_tokens=300)
    adp = os.path.join(RUN, "adapter")
    req = LoRARequest("u", 1, adp)
    out = open(f"{R}/unified_reader.jsonl", "w")
    acc = {}
    # local tasks on held-out seeds
    test = gen_split(300, 315, 30, n=20)
    from collections import defaultdict
    a = defaultdict(list)
    prompts = [build_prompt_local(s["graph"], s["q"], s["task"]) for s in test]
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp, lora_request=req)
    for s, o in zip(test, outs):
        parse = "bool" if s["task"].startswith("reach") else "list"
        gold = s["a"].split("ANSWER: ")[-1]
        pred = parse_answer(o.outputs[0].text, parse)
        sc = score(pred, gold, parse)
        a[s["task"]].append(sc)
        out.write(json.dumps({"task": s["task"], "score": sc}) + "\n")
    # iso families
    for k in (3, 4, 5):
        probes = [wl_pair(k, 2000+s) for s in range(20)]
        prompts = [iso_prompt(ga, gb) for ga, gb in probes]
        outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp, lora_request=req)
        hits = 0
        for (ga, gb), o in zip(probes, outs):
            pred = parse_answer(o.outputs[0].text, "bool")
            sc = score(pred, "NO", "bool"); hits += sc
            out.write(json.dumps({"task": f"iso_k{k}", "score": sc}) + "\n")
        a[f"iso_k{k}"].append(hits/20)
    out.close()
    print("\n== unified reader ==")
    for k in sorted(a):
        v = a[k]
        print(f"  {k}: {sum(v)/len(v):.3f}")
    print("UNIFIED DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "eval", "both"], default="both")
    args = ap.parse_args()
    if args.phase in ("train", "both"): main_train()
    if args.phase in ("eval", "both"): main_eval()
