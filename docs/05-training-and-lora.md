# 5. Training and LoRA on 8 GB

By **Adrian Ciprian Iancu** · [GitHub](https://github.com/ArtyomITA) · [Hugging Face](https://huggingface.co/adriandj3) · [LinkedIn](https://www.linkedin.com/in/adrian-ciprian-iancu-a2a477391)

Back to the [guide](../README.md).

Setup for everything in this chapter: GTX 1080 8 GB (Pascal, no bf16, no Triton, no flash attention), Windows 11, `torch==2.14` with CUDA 12.6 (the last PyTorch line with Pascal wheels), fp32 math.

## 5.1 Full-parameter SFT of a 2.5B model on 8 GB (BAdam)

**Problem.** Full fine-tuning of MiniCPM5-2B (2.5 B parameters, 42 layers) on 8 GB. [BAdam](https://github.com/Ledzy/BAdam) updates one block of layers at a time with AdamW, which fits; the question was how fast it can go.

**Setup.** `torch==2.14.1+cu126`, `transformers==5.18.0`, `badam==1.4`, weights in fp16, `switch_block_every=2`, random block order, batch 1, packed sequences of chat turns. Same seed and batches in every run so losses compare step by step; 12 steps per run, the first discarded, median seconds per step.

**The biggest lever: stop backward at the active block.** BAdam trains one block; the layers below it are frozen. Backward still walked through all of them in my baseline, because `gradient_checkpointing_enable()` in transformers 5 calls `enable_input_require_grads()` for every model with `input_ids`, which makes the embedding output require grad. One line after enabling checkpointing fixes it, and it needs non-reentrant checkpointing:

```python
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.disable_input_require_grads()   # backward stops at the active block
```

| Sequence 1024 | s / step | tokens / s | peak VRAM | loss vs baseline |
|---|---|---|---|---|
| baseline (reentrant checkpointing, standard loss) | 2.89 | 354 | 6.9 GB | |
| non-reentrant checkpointing | 2.89 | 354 | 6.9 GB | identical |
| **+ `disable_input_require_grads()`** | **1.93** | **531** | 6.9 GB | **identical** (12/12 steps) |
| + fused cross-entropy | 1.97 | 521 | 7.1 GB | max difference 0.0004 |
| + checkpoint only the MLP, keep attention activations | 1.82 | 563 | 7.0 GB | identical |

**Longer sequences: a fused cross-entropy.** With a 130,560-token vocabulary the logits are the memory problem. My fused loss computes loss and input gradient chunk by chunk in the forward pass (the head is frozen), so the full `(N, V)` logits never exist. The code is in [TRAINING-8GB.md](https://github.com/ArtyomITA/minicpm5-kernel-pascal/blob/main/TRAINING-8GB.md).

| With `disable_input_require_grads()` | s / step | tokens / s | peak VRAM |
|---|---|---|---|
| 2048, standard loss | 4.84 | 423 | 8.5 GB, spills into system RAM |
| 2048, fused loss | 4.48 | 457 | 6.4 GB |
| 4096, fused loss | 11.02 | 372 | 6.6 GB |
| 8192, fused loss | 34.60 | 237 | 7.2 GB |
| 8192, `F.linear_cross_entropy` + embedding on CPU + checkpoint offload + MLP in chunks of 512 | 31.48 | 260 | 6.3 GB |

**What did not help.** Adam 8-bit (3.10 s against 2.89 s per step), Adam-mini (does not run under BAdam 1.4), eager attention (out of memory at 4096), a legacy flash-attention build for Pascal (15x slower), resetting position ids per packed example (2.21 s against 1.93 s).

The same `enable_input_require_grads` default hides the savings of freezing layers in general: in my LoRA kit, freezing 8, 16 and 24 layers saved nothing until I disabled it, then -18%, -35% and -53%.

## 5.2 LoRA on a ternary model, trained on the 1080

**Problem.** Ternary Bonsai 2 9B is published only as GGUF (PQ2_0 weights with a Hadamard rotation) and MLX, so there are no fp16 weights to fine-tune. I rebuilt the model in PyTorch from the GGUF and trained a LoRA on top of the frozen ternary weights; the adapter is exported as a GGUF LoRA that the PrismML llama.cpp fork loads with `--lora`.

**Speed work.** Reference trainer: 20.2 s per example (median of 3 bit-identical runs), 5.84 GB. Each change was checked against the reference on the same 8 examples (per-example loss, the 400 LoRA gradients of the first step, the adapter after two steps):

| Change | s / example | Check |
|---|---|---|
| reference | 20.2 | 3 runs bit-identical to each other |
| Hadamard 4096 as two 64x64 factors (H4096 = H64 x H64) instead of a dense 4096x4096 matmul | 15.3 | within tolerance (loss 2e-6, gradients 5e-5) |
| own CUDA kernel that unpacks the 2-bit codes, one thread per byte, four fp32 weights at once | 12.4 | bit-identical |
| `disable_input_require_grads()` | 12.2 | bit-identical |
| **shared image prefix**, 2 targets of the same screenshot | 6.4 | within tolerance (loss 1.3e-6, gradients 1.9e-5) |
| shared image prefix, 4 targets | 3.3 | same |
| plus training only layers 8 to 31 | **2.7** | gradients of the trained layers bit-identical to the full run |

After these changes one example is about 8.2 s of fp32 GEMM (the 1080's real limit, about 6 TFLOPS), 3 s of small elementwise ops and the Gated DeltaNet recurrence, 0.4 s of unpacking and 0.3 s of attention. The only big lever left was to do less GEMM, which is what the shared prefix does: a screenshot is about 1,100 tokens and a target with its answer is 30 to 40.

**Result: TernaryPilot v1** ([model card](https://huggingface.co/adriandj3/TernaryPilot-v1-Computer-Browser-Use-LoRA)), rank 16, alpha 32, 5.5 hours, about 2,100 examples seen, 80 MB adapter. Greedy decoding, images at 1024 tokens, no evaluation screenshot used in training:

| | Base model | Earlier click-only LoRA | v1 |
|---|---:|---:|---:|
| ScreenSpot-v2, 200 case stratified subset | 80 / 200 | 153 / 200 | **156 / 200** |
| Real Windows apps in Italian, 37 targets | 2 / 37 | 13 / 37 | **15 / 37** |
| Real web pages, 210 targets | 28 / 210 | 99 / 210 | **101 / 210** |
| Agent actions, 145 held-out steps | 7 / 145 | 28 / 145 | **67 / 145** |
| Answers in the expected format | 131 / 145 | 93 / 145 | **145 / 145** |

With the adapter at scale 0 outside GUI requests, 12 out of 12 chat prompts give exactly the base model's greedy output. TernaryPilot v2 is the current version of this adapter, trained from scratch on 60,516 examples on a larger GPU; it is a drop-in with the same prompts ([card](https://huggingface.co/adriandj3/TernaryPilot-v2-Computer-Browser-Use-LoRA)).

**Reproduce.** `lora-training/` in [bonsai2-ternary-pascal](https://github.com/ArtyomITA/bonsai2-ternary-pascal/tree/master/lora-training):

```bash
export BONSAI_GGUF=Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf MMPROJ=Ternary-Bonsai-2-9B-mmproj-BF16.gguf
export QWEN_SRC=qwen35-9b GGUF_PY=llama.cpp/gguf-py CACHE_DIR=vision-cache
python make_data.py --agentnet agentnet_win_mac_18k.jsonl --agentnet-images agentnet_png --mind2web "mind2web/train-*.parquet" --out data/train.jsonl --images data/img
python vision_cache.py data/train.jsonl
HOURS=5.5 DATA=data/train.jsonl python train.py
```

Stop it with Ctrl+C or by creating `run/STOP`; it saves a checkpoint and resumes next time.

**Serving the adapter next to a chat model.** What I learned in one `llama-server`:

- Load it at scale 0 (`--lora-scaled file.gguf:0`) and turn it on per request with `"lora": [{"id": 0, "scale": 1}]`. With the adapter always on, the chat changes (KL 0.021, 6.4% of tokens with a different top-1) and decoding is about 30% slower (66.4 to 46.8 tok/s with MTP).
- Give GUI requests their own slot (`"id_slot": 1`). Changing the adapter on a slot clears its prompt cache: a click on the chat slot cost 17.7 s of prefill on an 8.6k-token context; on a dedicated slot the next chat turn took 110 ms.
- In llama.cpp b10549 a LoRA scale set by one request stays on that slot. Pass `"lora"` explicitly in every request, chat included.
- On Windows `--lora-scaled` splits the value on `:`, so `D:\x.gguf:0` breaks. Start the server from the adapter's folder and pass the file name only. I have a fix with a regression test ready for upstream.
- Send `"presence_penalty": 0` with clicks: with the server default of 1.0 I lost 4 clicks out of 200.

**Why a LoRA is needed here at all:** [docs/03](03-quantization-on-sm61.md#37-ternary-weights-cost-gui-skills-and-a-lora-brings-them-back).

## 5.3 Calibrated one-token decisions (a LoRA extracted from a merged model)

**Problem.** My assistant asks a small model many quick multiple-choice questions (which tool, which UI element, has the page loaded). Each answer is one token read from the probabilities of the option letters (`n_probs`, temperature 0), about 0.2 s on the 1080.

**What worked at zero cost.** Prefill `Answer:` after the closed thinking block, and apply batch calibration (Zhou et al., ICLR 2024): per category the prior is the mean of the letter probabilities over development inputs, estimated once on a balanced set, frozen, and subtracted in log space at decision time. 62 held-out English and Italian decisions, used once per final configuration (one question is 1.6 points):

| Model | Language | before | `Answer:` + calibration | full reasoning (7 to 9 s) |
|---|---|---|---|---|
| LFM2.5-2.6B | Italian | 66.1 | **79.0** | 90.3 |
| LFM2.5-2.6B | English | 71.0 | **85.5** | 91.9 |
| MiniCPM5-2B | Italian | 46.8 | 64.5 | 85.5 |
| MiniCPM5-2B | English | 51.6 | **77.4** | 87.1 |

**A LoRA for English.** pjev-2b is a MiniCPM5-2B fine-tuned for agent decisions. All 87 of its non-linear tensors are bit-identical to the base, so the fine-tune lives in the seven linear projections of each layer, and the singular values of the difference fall off at rank 64. I extracted it as a rank-64 LoRA by truncated SVD ([`tools/extract_lora.py`](https://github.com/ArtyomITA/minicpm5-kernel-pascal/blob/main/tools/extract_lora.py)), 201 MB in f16, published as [MiniCPM5-2B-pjev-LoRA](https://huggingface.co/adriandj3/MiniCPM5-2B-pjev-LoRA).

| Configuration, English held-out set (62) | Accuracy | Latency |
|---|---|---|
| MiniCPM5-2B, `Answer:` + calibration | 75.8 | |
| pjev-2b merged, its own prompt format | 91.9 (57/62) | median 0.16 s |
| **pjev as LoRA over MiniCPM5-2B Q8_0** | **93.5 (58/62)** | median 0.22 s |
| LFM2.5-2.6B with full reasoning | 91.9 | 7.8 s mean |

It costs about 200 MB of VRAM instead of a second 2.7 GB model. The LoRA and the merged model choose the same option on 60 of 62 questions. English only: 69.4 on the Italian set. Full notes, including what did not work (PriDe debiasing, permutation averaging, gamma tuned on development labels): [FAST-DECISIONS.md](https://github.com/ArtyomITA/minicpm5-kernel-pascal/blob/main/FAST-DECISIONS.md).

A trap when serving both: 20 one-token decisions during a 512-token chat answer took the chat from 76.5 to 50.1 tok/s, so give decisions their own slot.

## 5.4 Gradient checkpointing and optimizer notes

- `torch.optim.Muon` runs Newton-Schulz in bfloat16. Pascal has no native bf16, PyTorch emulates it, and on my card it was not slower (two 4096x4096 matmuls: 166 ms in bf16, 214 ms in fp32). I keep an fp32 version for full precision (cosine 0.9999 against the original on the same input).
- Choosing which layers to train from gradient windows did not replicate out of sample (negative rank correlation between halves). I decide only with real training runs now.
