# ANGE — Attention-Native Graph Encodings for LLMs

Reference implementation and full experimental record for:

> **Why LLMs Cannot Read Graphs — and What an Attention-Native Encoding Fixes.**
> Ling Zhu, Nengjie You, Ze Yang, Guangyu Liu. (IEEE TNNLS, submitted.)

Every serialization of a graph in current use — edge lists, adjacency/incidence
groupings, prose, JSON, traversals — is *positional*: the structural role of a
token rides on its column position. This repository contains the complete
artifact for the paper's three results:

1. **The three walls.** An information wall (lossy digests are provably
   statistics-bound, Thm. 1), an expressivity wall (isomorphism collapses at
   the 1-WL boundary for every lossless positional encoding, Thm. 2/4), and a
   reality wall (synthetic mastery does not transfer to real KGs).
2. **The pointer grammar `Γ_PG`.** Records whose structural roles are *content*
   tokens (`u -r-> v`), with losslessness, locality, and composability proofs,
   and a layer-exact alignment with relational message passing.
3. **What training repairs.** LoRA on 1.5B/7B models repairs *utilization*
   (nbr F1 0.21 → 0.63) but never *information* — the digest-trained arm
   hallucinates fluent, format-perfect wrong answers (119/210).

## Repository layout

```
ange/        Python package: graph generators, 9 encoders, 7 probe families,
             vLLM/API engine, E1–E3 runners, LoRA training, evaluation,
             significance tests, one-command report generator
results/     57 JSONL files — all ~3.5e4 raw inference records (one line per
             probe: task, encoding, gold, prediction, score) for every
             model × encoding × task cell in the paper
data/        NELL-995 and FB15k-237 splits used by the real-KG probes (E2)
```

LoRA adapters for all 18 training arms are attached to the
[releases page](../../releases) (`adapters.zip`, ~0.5 GB).

## Quickstart

```bash
pip install vllm transformers peft datasets openai

# E1: the encoding grid on a local model (vLLM)
python -m ange.run_e1 --model Qwen/Qwen2.5-7B-Instruct

# E3: LoRA training + evaluation (one arm)
python -m ange.train_lora --enc pointer --out runs/e3_pointer
python -m ange.eval_lora  --adapter runs/e3_pointer/adapter

# One-command report: every table and figure in the paper
python ange/aggregate.py && python ange/significance.py
```

API models (DeepSeek-V3, GPT-4o, GLM, R1) run through the same engine; put
your endpoints in `api_keys.json` (see `ange/engine.py`; the file is gitignored
and **not** part of this repository).

All probe generators are seeded (`--seed`), so every number in the paper can be
regenerated bit-exactly from the code plus the released records.

## Citation

```bibtex
@article{ange2026graphs,
  title   = {Why LLMs Cannot Read Graphs — and What an Attention-Native
             Encoding Fixes},
  author  = {Zhu, Ling and You, Nengjie and Yang, Ze and Liu, Guangyu},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
