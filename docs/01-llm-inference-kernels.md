# 1. LLM inference kernels in llama.cpp

By **Adrian Ciprian Iancu** · [GitHub](https://github.com/ArtyomITA) · [Hugging Face](https://huggingface.co/adriandj3) · [LinkedIn](https://www.linkedin.com/in/adrian-ciprian-iancu-a2a477391)

Back to the [guide](../README.md). Everything here is llama.cpp on a GTX 1080 (sm_61). Windows 11 with CUDA 12.4 unless a line says Linux.

## 1.1 CUDA graphs on Pascal: +40% on a MoE, +7% on a dense model

**Problem.** llama.cpp keeps CUDA graphs off below Volta. The gate comes from a 2024 change ([#6766](https://github.com/ggml-org/llama.cpp/pull/6766)) that measured 53.84 to 51.67 tok/s on a P40 with a dense model of about 700 graph nodes. On a small MoE model the graph has 1794 nodes and decoding on this card is launch-bound, so the picture is different.

**What I did.** One line in `ggml_cuda_graph_set_enabled`: the architecture gate goes from `GGML_CUDA_CC_VOLTA` to `GGML_CUDA_CC_DP4A` (610). That opens sm_61 and leaves sm_60 alone. `GGML_CUDA_DISABLE_GRAPHS=1` still turns graphs off.

**Result.** GTX 1080, driver 581.57, CUDA 12.4, build b10549 (`b2e5e9b`), Windows:

| Model | tg64 graphs off | tg64 graphs on | tg64 at depth 4096, off / on | smallest gain of the two depths | perplexity |
|---|---|---|---|---|---|
| granite-3.1-3b-a800m Q4_K_M | 81.2 | 124.5 | 76.5 / 97.4 | +27.3% | identical |
| Ling-3.0-tiny Q6_K (`bailingmoe3`, 7.9B total, 1.3B active, 128 experts, 8 used) | 52.7 | 79.8 | 51.7 / 76.4 | +48.0% | identical |
| granite-4.0-h-tiny Q4_K_M | 57.4 | 79.5 | 54.2 / 77.4 | +38.5% | identical |
| OLMoE-1B-7B Q4_K_M | 150.5 | 169.5 | 115.8 / 124.6 | +7.6% | identical |
| LFM2.5-8B-A1B Q4_K_M | 116.2 | 128.5 | 112.2 / 125.6 | +10.6% | identical |
| dense Qwen-based 9B Q5_K_M | 28.5 | 30.1 | 27.9 / 29.5 | +5.6% | identical |

In the server, on Ling-3.0-tiny Q6_K: short prompt 41.2 to 57.6 tok/s (+39.8%), long generation 47.7 to 72.2 (+51.4%), prefill 857.0 to 853.0 tok/s (-0.5%). On the dense 9B: 26.3 to 28.0 (+6.4%) and 25.3 to 27.2 (+7.7%) in two runs. All the gain is in generation; prefill stays inside the noise.

The gain follows the number of nodes in the graph. It shrinks where attention cost grows (OLMoE +12.7% at depth 0, +7.6% at depth 4096) and holds on architectures with cheap attention (granite-4.0-h, a Mamba hybrid, +42.8% at depth 4096).

**Correctness.** Temperature 0, top_k 1, fixed seed, 10 different prompts of 200 tokens each (math, prose, translation, Python, SQL, lists): 10/10 identical, dense model too. `test-backend-ops` on CUDA0: 13348/13348 passed.

**Other people's cards.** A commenter ran Ling-3.0-tiny Q8_0 on a P102-100 (10 GB, Ubuntu, CUDA 12.9): tg128 70.80 to 88.42 tok/s. On P100s (sm_60) another contributor, dfriehs, saw single-card regressions, which is why the gate stays at DP4A and leaves sm_60 off.

**Where it does not help.** MiniCPM5-2B with a DSpark draft on Linux: x1.00 with and without a draft (0.996, 0.997, 1.003). On Windows (WDDM) every launch costs more, which is where the MoE gain showed up. Measure it on your model.

**Reproduce.**

```bash
# ggml/src/ggml-cuda/ggml-cuda.cu, in ggml_cuda_graph_set_enabled:
#   cc < GGML_CUDA_CC_VOLTA   ->   cc < GGML_CUDA_CC_DP4A
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61-real -DGGML_CUDA_GRAPHS=ON
cmake --build build --config Release -j 2
GGML_CUDA_DISABLE_GRAPHS=1 ./build/bin/llama-bench -m model.gguf -r 3   # graphs off
./build/bin/llama-bench -m model.gguf -r 3                              # graphs on
```

**Links.** Pull request [ggml-org/llama.cpp#27721](https://github.com/ggml-org/llama.cpp/pull/27721), open since 25 August 2026. Notes: [PASCAL-NOTES](https://github.com/ArtyomITA/minicpm5-kernel-pascal/blob/main/PASCAL-NOTES.md).

## 1.2 Ternary Bonsai 2 9B: +51% decode with the same logits

**Problem.** [Ternary-Bonsai-2-9B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf) (PQ2_0 weights plus an MTP draft head) runs in the [PrismML fork of llama.cpp](https://github.com/PrismML-Eng/llama.cpp). On sm_61 the multi-column PQ2_0 kernel is only enabled on Ada, so 1 to 4 column work goes to the generic vec_dot.

**What I did.** Five changes to one file, `ggml/src/ggml-cuda/ggml-cuda.cu` (+987 / -22 lines), each kept only if it gained at least 3% over the previous best:

| # | Change | Gain when kept | Worth inside the final program |
|---|---|---|---|
| 1 | CUDA graphs on Pascal, only for small batches (none when a quantized mat-mul has more than 8 columns, so prefill keeps the stock path) | +3.9% | +17.1% |
| 2 | PQ2_0 mat-vec for 1 to 4 columns that reproduces the summation order of the generic path bit for bit: a warp per 4 rows, no shared memory, no barriers, activations in the PQ2 layout | +18.6% | +36.9% |
| 3 | Decode-cycle kernels without a sort: top-k for k <= 32 in two launches, GDN concat with one thread per element | +6.1% | +3.3% |
| 4 | Hadamard rotation (fwht, block 4096) and q8_1 quantization fused in one kernel in the PQ2 layout | +4.2% | +4.5% |
| 5 | Fusions with identical output: add + rms_norm, rms_norm + mul + swiglu, strided F32 copies, ssm alpha + beta, conv-state gather | +10.7% | +8.0% |

"Worth inside the final program" is final / final-without-the-piece, measured by ablation in the same session. The pieces overlap, so the column does not add up.

**Result.** Decode with PQ2_0 + MTP, GPU starting at 55 C or below: **55.9 to 84.4 tok/s (+51%)**. KL divergence against the stock logits at ubatch 512 / 3 / 1: 0 / 0 / 1e-6, same top token 100 / 100 / 99.902% (the stock build against its own saved logits also gives 99.902% at ubatch 1). Greedy text over 11k tokens and four concurrent slots: identical. VRAM -44 MiB. Kernel launches per decode cycle: 976 before change 5, 755 after. The mat-vec stays about 65% of the cycle at roughly 145 GB/s, far from the card's bandwidth, because the summation order fixed by the quality gate leaves it bound by instructions and occupancy.

At full heat the patched build runs at about 72 tok/s; all the numbers above start cool.

**Reproduce.**

```bash
git clone https://github.com/PrismML-Eng/llama.cpp && cd llama.cpp
git checkout 0082ac24a
git apply /path/to/patches/0001-sm61-pq2_0-mtp-decode.patch
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61-real -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-perplexity -j
```

**Merged upstream.** The mat-vec alone (change 2) is [PrismML-Eng/llama.cpp#368](https://github.com/PrismML-Eng/llama.cpp/pull/368): server decode 55.7 to 65.3 tok/s (+17.3%, all five pairs faster, +13.6% to +19.5%), llama-bench tg128 47.4 to 51.2 (+8.1%), pp512 629.7 to 635.7, `test-backend-ops` 15873/15873. Merged on 19 October 2026.

**What did not work.** A different summation order (KLD 7.3e-4, rejected by the gate); variants of the ordered mat-vec (4 chunks together, 16-byte loads through shared memory, 2 rows per warp), neutral or slower inside the model; an aligned copy of the weights (+901 MiB VRAM for +2.4%); CUDA graphs at every batch size (-4.9% prefill); a Q8_0 mat-vec of the MTP layer in the exact order (the stock path is already at 200 to 215 GB/s).

**Links.** [bonsai2-ternary-pascal](https://github.com/ArtyomITA/bonsai2-ternary-pascal) with the patch, the evaluator and `results/results.tsv` (every run).

## 1.3 MiniCPM5-2B kernels (kmcG), bit-identical

**Problem.** MiniCPM5-2B Q8_0 with a speculative draft verifies 6 columns per step, and the stock Q8 mat-vec for 1 to 8 columns leaves time on the table on sm_61.

**What I did.** A patch against llama.cpp b10549 (`b2e5e9b28`) with three kernel changes and two small ones, each with an environment valve:

| Part | What |
|---|---|
| kmcG Q8K (`mmvq.cu`) | Q8_0 mat-vec for 1 to 8 columns with the upstream thread mapping and sum order, constant pointers, `__ldg` on aligned words, funnel shifts; 4 rows per block at 1 column with 4096 rows or more |
| kmcG ROWWARP | one row per warp when K <= 256 (the DSpark Markov layer, 256 x 130560) |
| kmcG QKV | gate and up projections at 2 to 8 columns in one launch with one shared quantized input |
| B02 | allows CUDA graphs on cc 6.1 |
| B04 | Q8_0 to f16 conversion in float on Pascal |

**Result.** +5.5% decode (geometric mean of 4 prompt types, two ABBA pairs from +4.7% to +6.1%), 95/95 harness cases identical, 10/10 server outputs identical. Profile of one speculative step on Linux (675 steps): verification 25.8 ms of GPU work (quantized mat-vec 12, flash-attention tile 7, KV conversion 3, output head 1.4), draft 6.0, draft KV injection 2, host gaps about 3, gaps between kernels 2.

**Verified, not adopted.** Attention that reads the q8_0 KV cache directly: -46 us per layer at 8k and bit-identical (18/18 cases), but dispatched in prefill too, where it is 21% slower; an f16 cache removes the same conversion (section 1.4). A fused RMS norm + q8_1 quantization: +0.7%, below my 1.5% bar. Host-side work per step is 1.86 ms of 31.47 ms, so the ceiling there is +6%.

**Reproduce.** The harness runs the MiniCPM5 shapes on deterministic data against two builds and fails on any difference:

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout b2e5e9b28
patch -p1 < ../kernels/patches/pascal-sm61-kmcG-B02-B04.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build build --config Release -j 2 --target llama-server llama-bench
python3 harness/harness.py upstream-vs-patched upstream/build/bin patched/build/bin
```

Setting any valve variable (for example `GGML_CUDA_DISABLE_KMC_Q8K`, even to `0`) disables that part. With all valves set the build behaves like upstream. See [kernels/README](https://github.com/ArtyomITA/minicpm5-kernel-pascal/blob/main/kernels/README.md).

**Links.** [minicpm5-kernel-pascal](https://github.com/ArtyomITA/minicpm5-kernel-pascal), [RESULTS.md](https://github.com/ArtyomITA/minicpm5-kernel-pascal/blob/main/RESULTS.md).

## 1.4 KV cache in f16 instead of q8_0: +13.6%

**Problem.** With a q8_0 KV cache, every attention call first converts the whole cache to f16, about 3 ms per speculative step at 49k context.

**What I did.** `--cache-type-k f16 --cache-type-v f16`. It changes numbers, so it went through a quality gate: KL divergence on 80k English tokens, reference = Q8_0 weights + f16 KV (perplexity 3.5909).

| Candidate | PPL ratio | mean KLD | KLD 99th percentile | same top-1 token |
|---|---|---|---|---|
| reference with another ubatch (noise) | 1.00026 | 0.000000 | 0.00004 | 99.995% |
| KV q8_0 | 1.00067 | 0.00092 | 0.0062 | 98.67% |
| Q6_K weights + KV f16 | 1.00526 | 0.00643 | 0.043 | 96.36% |

Gate: mean KLD <= 0.001 and top-1 >= 99.5%. f16 KV is the reference, so it passes by definition; the q8_0 cache I used before was the less faithful one.

**Result.** English replay of 30 turns: q8_0 69.7 / 63.7 tok/s, f16 79.0 / 79.2 tok/s; ms per step 38.8 against 34.15 in both pairs (-12%, +13.6% at equal acceptance). VRAM 5341 to 6285 MiB at 49k context. If memory is short, `-c 32768` frees about 0.7 GB and does not change speed on short prompts.

## 1.5 Erase finished slots: the cheapest 2x I found

**Problem.** With a unified KV cache, llama.cpp keeps the sequences of finished conversations until the slot is reused or erased, and attention walks over them at every step.

**Result** (replay of 66 real agent turns, llama.cpp b10549, Linux):

| Arm | decode tok/s | prefill tok/s | ms per step |
|---|---|---|---|
| clean slots | 59.7 | 907 | 40.3 |
| three dead 10k-token sequences in the cache | 29.6 (x0.50) | 503 (x0.55) | 80.3 |
| after erasing the slots | 57.8 (x0.97) | 902 | 40.5 |

Fit over 61 requests: ms per step = 29.9 + 1.26 x (thousands of tokens in the cache, dead ones included).

**Reproduce.** Start the server with `--slot-save-path ./slots`, which enables the slot actions endpoint, and erase when a conversation ends:

```bash
curl -X POST "http://127.0.0.1:8080/slots/0?action=erase"
```

## 1.6 Build for the card, not for PTX

`-DCMAKE_CUDA_ARCHITECTURES=61-real` instead of the default PTX: +4.6% generation on Windows, the DLL drops from 538 MB to 69.5 MB and startup from 43.3 s to 20.4 s, output identical ([PASCAL-NOTES](https://github.com/ArtyomITA/minicpm5-kernel-pascal/blob/main/PASCAL-NOTES.md)). The binary then runs only on that architecture, so rebuild after every llama.cpp update.

Prefill micro-batch on an 8k prompt, f16 KV (`llama-bench -p 8192`, ABBA): ubatch 256 1151.9, 512 1198.8, 1024 1263.6, 2048 1269.0 tok/s. I use 1024; 2048 costs +534 MiB for +0.4%.

## 1.7 Things that did not help on this card

From [PASCAL-NOTES](https://github.com/ArtyomITA/minicpm5-kernel-pascal/blob/main/PASCAL-NOTES.md), all measured:

| Change | Result |
|---|---|
| Windows hardware-accelerated GPU scheduling | about 0 gain and +514 MiB of VRAM; turned back off |
| Fan fixed at 90% | same steady clock as auto after 4 minutes of prefill (1632 MHz, 88 C against 89 C); the throttling is thermal |
| n-gram speculation on top of the DSpark draft | +6.5% in a first pass without a fixed seed, x1.003 with seed 42 |
| CPU power profile `performance` against `balanced` | 83.3 against 83.4 tok/s |

Measurement traps I fell into: "server prefill is 3.5x slower than llama-bench" was an artifact (measured properly: 1.23x); a long run heats the card and drops the clock from about 1733 to 1632 MHz; at temperature 1.0 without a fixed seed two runs of the same build differed by up to 10%.
