# 2. Speculative decoding on Pascal

By **Adrian Ciprian Iancu** · [GitHub](https://github.com/ArtyomITA) · [Hugging Face](https://huggingface.co/adriandj3) · [LinkedIn](https://www.linkedin.com/in/adrian-ciprian-iancu-a2a477391)

Back to the [guide](../README.md).

## Why a draft is almost free on this card

At batch size one the card waits on kernel launches, not on arithmetic. Verifying 4 draft tokens costs about the same as verifying 1, so a short draft adds little time per step and every accepted token is a launch saved ([PASCAL-NOTES](https://github.com/ArtyomITA/minicpm5-kernel-pascal/blob/main/PASCAL-NOTES.md)). The draft has to be cheap itself: its own output head can become the bottleneck, which is what FR-Spec fixes.

## 2.1 DSpark draft for MiniCPM5-2B

**Setup.** Target MiniCPM5-2B Q8_0, the official DSpark draft, 5 draft tokens (`--spec-draft-n-max 5`), verification runs 6 columns. Replay of real agent turns (tool calls, answers, long JSON results), temperature 1.0.

**Result.** In a real agent replay the draft is worth about +20% over no draft, not the +50% a synthetic benchmark suggests. Acceptance on English agent turns is about 0.33 and 2.67 tokens come out per step.

```bash
llama-server -m MiniCPM5-2B-Q8_0.gguf \
  -md MiniCPM5-DSpark-fr32k-Q8_0.gguf -ngld 999 \
  --spec-type draft-dspark --spec-draft-n-max 5 --spec-draft-n-min 0 \
  --cache-type-k f16 --cache-type-v f16 \
  -c 49152 -b 2048 -ub 1024 -ngl 999 -fa on --jinja \
  --slot-save-path ./slots \
  --temp 1.0 --top-p 0.95 --min-p 0.0
```

On Linux I set `GGML_CUDA_DISABLE_GRAPHS=1` in front of it because graphs gave 0.0% there (three measurements within 0.5%). Full flag table: [RECIPE.md](https://github.com/ArtyomITA/minicpm5-kernel-pascal/blob/main/RECIPE.md).

## 2.2 FR-Spec: a smaller output head for the draft

**Problem.** The DSpark draft scores every draft position against the full 130,560-token output head, plus a 256 x 130,560 Markov layer. Most of those rows are never proposed.

**What I did.** Following the FR-Spec idea (frequency-ranked speculative sampling, Zhao et al. 2025), I kept the 32,768 most useful rows and mapped them back to the full vocabulary with a `d2t` table. The target still verifies every token, so the text is distributed exactly as before; only the acceptance rate could drop. Rows are copied byte for byte (Q8_0), no retraining and no requantization. The kept set comes from about 242k tokens of MiniCPM5 agent output (8,381 distinct ids), plus all special tokens, filled with the lowest ids; it covers 99.16% of the tokens of a held-out run.

**Result.** 24 real English agent turns, fixed seed, ABBA:

| Draft | decode tok/s | ms per step | tokens per step | acceptance | VRAM |
|---|---|---|---|---|---|
| official DSpark Q8_0 | 79.6 / 79.8 | 33.5 | 2.67 | 0.333 | 6285 MiB |
| FR-Spec 32k | **83.5 / 83.4** | **31.9** | 2.66 | 0.331 | 6407 MiB |

+4.7% in both pairs, acceptance -0.2 points, +122 MiB because the reduced head is a separate copy.

**Reproduce.** The draft is on Hugging Face: [MiniCPM5-2B-DSpark-FRSpec32k-GGUF](https://huggingface.co/adriandj3/MiniCPM5-2B-DSpark-FRSpec32k-GGUF) (394 MB). To build one from your own token statistics: [`tools/frspec.py`](https://github.com/ArtyomITA/minicpm5-kernel-pascal/blob/main/tools/frspec.py). It needs a llama.cpp build whose DFlash/DSpark graph supports the `d2t` tensor (tested with b10549, no source change needed).

## 2.3 MTP on Ternary Bonsai 2 9B

The model ships an MTP draft head (`Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf`). I run it with `--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-p-min 0`. With MTP the verification step runs 3 columns, and that is where the generic PQ2_0 path was slowest, so the exact-order 1-to-4-column mat-vec of [section 1.2](01-llm-inference-kernels.md) pays twice: without MTP the model decodes at 47.4 tok/s (llama-bench tg128, stock) and the patch adds +8.1%; with MTP the patch adds +17.3% on the server. Full numbers: PrismML-Eng/llama.cpp [#368](https://github.com/PrismML-Eng/llama.cpp/pull/368).

## 2.4 LFM2.5-2.6B with the official DSpark draft

My local assistant [Vergilius](https://github.com/ArtyomITA/vergilius) uses LFM2.5-2.6B (Liquid AI) with the official DSpark drafter on the GTX 1080: 64 tok/s decode, 78 to 100 tok/s with the draft active (figures from the project README).

## 2.5 What did not pay

| Idea | Result |
|---|---|
| n-gram speculation (`ngram-map-k`) on top of the DSpark draft | first pass +6.5%, with a fixed seed x1.003: noise |

## When a draft is worth it

- Batch size 1, a launch-bound model (small dense or MoE): yes, and keep the draft at 3 to 5 tokens.
- A draft with a large vocabulary head: shrink the head first (FR-Spec), the saving shows up directly in ms per step.
- Compare in ms per step at equal acceptance, not in tok/s of a single sampled run: at temperature 1.0 with a fixed seed, the same configuration reproduced the same sampled text in only 8 of 24 turns.
