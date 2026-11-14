#!/usr/bin/env python
"""补救2: 复杂查询墙 —— 前沿链式监督能否救 join。

假设: join 失败是"监督缺口"而非"表示墙" —— 教模型先寻址两个源、再取交。
数据: join 查询 + 分步 gold CoT (Step1: u1 的 r-目标; Step2: u2 的; Step3: 交)。
对照: 同一编码(pointer2) 仅答案监督 vs 分步监督 vs 冻结。
若分步监督把 join 从 ~0.25 提到 >0.7: 墙是可训的计算缺口(梯子又一级);
若不动: 是更深的墙, 为二次文法保留动机。
"""
from __future__ import annotations
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from ange.graph import gen_graph
from ange.e3_data import HEADER
from ange.probes import FOOT_LIST
from ange.encoders import enc_pointer2
from ange.engine import parse_answer, score

B7 = "/mnt/ssd/ange/models/Qwen2.5-7B-Instruct"
RUN = "/mnt/ssd/ange/runs/fu_join_7b"
R = "/mnt/ssd/ange/results"


def gen_join(n, seed, n_per):
    rng = random.Random(seed)
    out = []
    got = 0
    while got < n_per:
        g = gen_graph(500 + got, 20, 4, 1.4)
        u1, u2 = rng.sample(g.nodes(), 2)
        r = rng.choice(g.rels)
        s1 = set(g.out_neighbors(u1, r)); s2 = set(g.out_neighbors(u2, r))
        common = s1 & s2
        if not (1 <= len(common) <= 4):
            continue
        q = (f"List ALL nodes that are targets of {r} edges from BOTH node "
             f"{u1} AND node {u2}. ")
        out.append({"g": g, "q": q, "common": sorted(common),
                    "s1": sorted(s1), "s2": sorted(s2), "u1": u1, "u2": u2,
                    "r": r})
        got += 1
    return out


def prompt_of(p):
    gt = enc_pointer2(p["g"])
    return HEADER + gt + "\n\nQuestion: " + p["q"] + FOOT_LIST


def cot_steps(p):
    return (f"Step 1: {p['r']} edges from {p['u1']} point to "
            f"{{{', '.join(p['s1']) or 'none'}}}.\n"
            f"Step 2: {p['r']} edges from {p['u2']} point to "
            f"{{{', '.join(p['s2']) or 'none'}}}.\n"
            f"Step 3: their common targets: {{{', '.join(p['common'])}}}.\n"
            f"ANSWER: {', '.join(p['common'])}")


def cot_plain(p):
    return f"The common targets are {{{', '.join(p['common'])}}}.\nANSWER: {', '.join(p['common'])}"


def main_train(mode):
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
    data = gen_join(20, 77, 800)
    gold_fn = cot_steps if mode == "steps" else cot_plain
    feats = []
    for p in data:
        pr = prompt_of(p)
        full = tok.apply_chat_template(
            [{"role": "user", "content": pr},
             {"role": "assistant", "content": gold_fn(p)}],
            tokenize=False, add_generation_prompt=False)
        prefix = tok.apply_chat_template([{"role": "user", "content": pr}],
                                         tokenize=False, add_generation_prompt=True)
        ids = tok(full, add_special_tokens=False)["input_ids"][:2048]
        pre = tok(prefix, add_special_tokens=False)["input_ids"]
        labels = [-100]*len(pre) + ids[len(pre):]
        feats.append({"input_ids": ids[:2048], "labels": labels[:2048]})
    targs = TrainingArguments(
        output_dir=f"{RUN}_{mode}", num_train_epochs=2.0,
        per_device_train_batch_size=4, learning_rate=1e-4, warmup_steps=30,
        lr_scheduler_type="cosine", logging_steps=20, save_strategy="no",
        bf16=True, report_to=[], remove_unused_columns=False,
        gradient_checkpointing=True)
    Trainer(model=model, args=targs, train_dataset=feats,
            data_collator=Collator(tok)).train()
    model.save_pretrained(f"{RUN}_{mode}/adapter")
    tok.save_pretrained(f"{RUN}_{mode}/adapter")
    print("saved", RUN + "_" + mode, flush=True)


def main_eval():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=B7, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.42, enable_lora=True, max_lora_rank=16)
    sp = SamplingParams(temperature=0.0, max_tokens=300)
    test = gen_join(20, 555, 60)      # held-out seed
    out = open(f"{R}/join_fix.jsonl", "w")
    arms = [("frozen", None), ("plain_lora", f"{RUN}_plain/adapter"),
            ("steps_lora", f"{RUN}_steps/adapter")]
    for tag, adp in arms:
        req = LoRARequest(tag, 1, adp) if adp else None
        prompts = [prompt_of(p) for p in test]
        outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp,
                        lora_request=req)
        f1s = []
        for p, o in zip(test, outs):
            pred = parse_answer(o.outputs[0].text, "list")
            gold = ",".join(p["common"])
            sc = score(pred, gold, "list")
            f1s.append(sc)
            out.write(json.dumps({"arm": tag, "gold": gold, "pred": pred,
                                  "score": sc}) + "\n")
        out.flush()
        print(f"[{tag}] join F1 = {sum(f1s)/len(f1s):.3f}", flush=True)
    out.close()
    print("JOIN FIX DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train_plain", "train_steps", "eval", "all"],
                    default="all")
    a = ap.parse_args()
    if a.phase in ("train_plain", "all"): main_train("plain")
    if a.phase in ("train_steps", "all"): main_train("steps")
    if a.phase in ("eval", "all"): main_eval()
