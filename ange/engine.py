"""Inference engines: local vLLM batch generation and OpenAI-compatible APIs.

Local: one vLLM LLM instance per model, chat template applied, temperature 0.
API:   GLM (https://open.bigmodel.cn/api/paas/v4) and DeepSeek
       (https://api.deepseek.com), both OpenAI-compatible.
"""
from __future__ import annotations

import json
import os
import time
from typing import List

from .probes import Probe

API_KEYS_FILE = "/mnt/ssd/ange/api_keys.json"


class LocalEngine:
    def __init__(self, model_path: str, tp: int = 1, max_len: int = 16384):
        from vllm import LLM
        self.llm = LLM(model=model_path, tensor_parallel_size=tp,
                       dtype="bfloat16", max_model_len=max_len,
                       gpu_memory_utilization=0.90, enforce_eager=False)
        self.name = os.path.basename(model_path.rstrip("/"))

    def run(self, probes: List[Probe], max_tokens: int = 96) -> List[str]:
        from vllm import SamplingParams
        msgs = [[{"role": "user", "content": p.prompt}] for p in probes]
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outs = self.llm.chat(msgs, sp)
        return [o.outputs[0].text for o in outs]


class ApiEngine:
    """OpenAI-compatible client. Provider config in api_keys.json:

    {"glm": {"key": "...", "base_url": "https://open.bigmodel.cn/api/paas/v4",
             "models": ["glm-4-flash", "glm-4-plus"]},
     "deepseek": {"key": "...", "base_url": "https://api.deepseek.com",
               "models": ["deepseek-chat"]}}
    """

    def __init__(self, provider: str, model: str, keys_file: str = API_KEYS_FILE):
        from openai import OpenAI
        cfg = json.load(open(keys_file))[provider]
        self.client = OpenAI(api_key=cfg["key"], base_url=cfg["base_url"], timeout=240.0, max_retries=0)
        self.model = model
        self.name = model

    def run(self, probes: List[Probe], max_tokens: int = 96,
            workers: int = 12, retries: int = 5, _workers_override: int = 0) -> List[str]:
        from concurrent.futures import ThreadPoolExecutor
        lock_out: dict = {}

        def _one(i_p):
            i, p = i_p
            for attempt in range(retries):
                try:
                    r = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": p.prompt}],
                        temperature=0.0, max_tokens=max_tokens)
                    return i, (r.choices[0].message.content or "")
                except Exception as e:
                    if attempt == retries - 1:
                        return i, f"[ERROR {type(e).__name__}]"
                    time.sleep(min(2 ** attempt, 20))
            return i, "[ERROR]"

        with ThreadPoolExecutor(max_workers=(_workers_override or workers)) as ex:
            for i, txt in ex.map(_one, enumerate(probes)):
                lock_out[i] = txt
        return [lock_out[i] for i in range(len(probes))]


# ------------------------------------------------------------------ scoring

import re

_NODE = re.compile(r"\bn\d+\b", re.IGNORECASE)
_YES = re.compile(r"\byes\b|\bno\b", re.IGNORECASE)


def parse_answer(text: str, parse: str):
    # strip reasoning-model <think> blocks before parsing
    if "</think>" in text:
        text = text.split("</think>")[-1]
    if parse == "bool":
        m = _YES.search(text)
        return m.group(0).upper() if m else "NA"
    # list: prefer the final ANSWER line, else last comma list of nodes
    m = re.search(r"ANSWER\s*[:：]\s*(.+)", text, re.IGNORECASE)
    seg = m.group(1) if m else text
    if re.search(r"\bNONE\b", seg, re.IGNORECASE) and not _NODE.search(seg):
        return "NONE"
    nodes = _NODE.findall(seg)
    if not nodes:
        return "NONE" if re.search(r"\bNONE\b", seg, re.IGNORECASE) else "NA"
    uniq = list(dict.fromkeys(n.lower() for n in nodes))
    return ",".join(uniq)


def score(pred: str, gold: str, parse: str):
    if pred == "NA":
        return 0.0
    if parse == "bool":
        return 1.0 if pred == gold else 0.0
    if pred == "NONE" and gold == "NONE":
        return 1.0
    p = pred.split(",") if pred != "NONE" else []
    g = gold.split(",") if gold != "NONE" else []
    ps, gs = set(p), set(g)
    if not gs:
        return 1.0 if not ps else 0.0
    if not ps:
        return 0.0
    inter = len(ps & gs)
    prec, rec = inter / len(ps), inter / len(gs)
    return (2 * prec * rec / (prec + rec)) if prec + rec > 0 else 0.0
