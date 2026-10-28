# 9. Upstream contributions

By **Adrian Ciprian Iancu** · [GitHub](https://github.com/ArtyomITA) · [Hugging Face](https://huggingface.co/adriandj3) · [LinkedIn](https://www.linkedin.com/in/adrian-ciprian-iancu-a2a477391)

Back to the [guide](../README.md). Status checked on 28 October 2026.

| Pull request | Repository | What it does | Result | Status |
|---|---|---|---|---|
| [#27721](https://github.com/ggml-org/llama.cpp/pull/27721) | ggml-org/llama.cpp | enables CUDA graphs on Pascal (sm_61) by moving one architecture gate | **+40% on a MoE** (Ling-3.0-tiny 41.2 to 57.6 tok/s), **+7% on a dense 9B**, no regression, 13348/13348 `test-backend-ops` | **open**, since 25 August 2026 |
| [#368](https://github.com/PrismML-Eng/llama.cpp/pull/368) | PrismML-Eng/llama.cpp | PQ2_0 mat-vec for 1 to 4 columns on sm_61, same summation order as the generic path | **+17.3%** decode on Ternary-Bonsai-2-9B with MTP (55.7 to 65.3 tok/s), bit-identical, 15873/15873 `test-backend-ops` | **merged** on 19 October 2026 |
| [#2202](https://github.com/leejet/stable-diffusion.cpp/pull/2202) | leejet/stable-diffusion.cpp | skips the file-level LoRA `alpha` when the file also carries `baked_scale` | fixes broken MiniMax-H3 turbo-LoRA video; timing unchanged | **merged** on 24 October 2026 |

## #27721: CUDA graphs on Pascal

One changed line in `ggml_cuda_graph_set_enabled` (`GGML_CUDA_CC_VOLTA` to `GGML_CUDA_CC_DP4A`). The full table across six models, the correctness checks and the comments from people who tested on other Pascal cards are in [docs/01](01-llm-inference-kernels.md#11-cuda-graphs-on-pascal-40-on-a-moe-7-on-a-dense-model). A contributor who tested P100s (sm_60) recommended keeping graphs off there, and the DP4A gate does exactly that; his single-card results were regressions.

## #368: ternary mat-vec for Pascal in the PrismML fork

The safest piece of a larger set of changes for the same model and card, which together give +51% (55.9 to 84.4 tok/s) with the same logits. What each piece is worth in the full build: this mat-vec +36.9%, CUDA graphs for small batches +17.1%, identical-output fusions +8.0%, fwht + q8_1 quantization in one kernel +4.5%, top-k and GDN concat without a sort +3.3%. Details: [docs/01](01-llm-inference-kernels.md#12-ternary-bonsai-2-9b-51-decode-with-the-same-logits).

## #2202: LoRA alpha with baked_scale in stable-diffusion.cpp

Since an earlier change (#2159) the LoRA loader uses the file-level `alpha` metadata when a tensor has no alpha of its own. Some converted MiniMax-H3 LoRAs store both `alpha` and `baked_scale`: the converter already folded alpha / rank into `lora_B`, so stable-diffusion.cpp applied alpha / rank a second time. Every layer got an extra factor of 8 / rank, anywhere between 0.05 and 4, and the video came out broken. The fix skips the metadata alpha only when `baked_scale` is present; LoRAs without it behave as before. Details and the test setup: [docs/06](06-diffusion-image-and-motion.md#61-the-lora-alpha-fix-merged).

## Prepared, not yet opened

- **stable-diffusion.cpp, faster weight streaming with `--params-backend disk`** (two commits: no 200 ms idle wait at the end of `load_tensors`, and the next segment read on a worker thread while the GPU computes). Measured on upstream 95ff762, sampling -38%, whole run -23%. Numbers in [docs/06](06-diffusion-image-and-motion.md#64-video-and-image-generation-minimax-h3-on-8-gb).
- **llama.cpp, `--lora-scaled` on Windows paths**: the value is split on `:`, so `D:\x.gguf:0` breaks. Fix and regression test ready ([notes](https://github.com/ArtyomITA/bonsai2-ternary-pascal/tree/master/lora-training)).
- **transformers, `gradient_checkpointing_enable()` always calls `enable_input_require_grads()`**: with non-reentrant checkpointing it is not needed and it forces backward from layer 0 even when lower layers are frozen. Measurements in [TRAINING-8GB](https://github.com/ArtyomITA/minicpm5-kernel-pascal/blob/main/TRAINING-8GB.md) and [docs/05](05-training-and-lora.md).

## How I contribute

One change per pull request, a measured A/B in the description (median of alternated runs, noise floor stated), a correctness check next to every speed number, and a note on what I could not test: for example, I have no sm_60 card, so the DP4A gate keeps it out.
