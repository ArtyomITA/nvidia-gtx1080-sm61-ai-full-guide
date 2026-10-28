# 3. Quantization choices on sm_61

By **Adrian Ciprian Iancu** · [GitHub](https://github.com/ArtyomITA) · [Hugging Face](https://huggingface.co/adriandj3) · [LinkedIn](https://www.linkedin.com/in/adrian-ciprian-iancu-a2a477391)

Back to the [guide](../README.md).

## 3.1 Why FP16 is the wrong default here

Native fp16 arithmetic runs at 1/64 of fp32 on the GP104 chip. So on this card fp16 is a storage format only: weights and caches can be fp16, but the math has to go through fp32 or through the int8 dot product `dp4a`. My training runs in fp32 for that reason, and my kernels do their arithmetic in fp32 or int8 ([PASCAL-NOTES](https://github.com/ArtyomITA/minicpm5-kernel-pascal/blob/main/PASCAL-NOTES.md)). bf16 does not exist in hardware; PyTorch emulates it.

## 3.2 The rule I use: a quality gate before any speed number

Any change that alters numbers (weight quantization, KV type, kernel summation order) has to pass a KL-divergence gate against the most faithful configuration I can run, measured with `llama-perplexity --kl-divergence` on tens of thousands of tokens. Kernel-only changes must be bit-identical.

| Model | Gate | Where it came from |
|---|---|---|
| MiniCPM5-2B | mean KLD <= 0.001 and same top-1 token >= 99.5% | [RESULTS.md](https://github.com/ArtyomITA/minicpm5-kernel-pascal/blob/main/RESULTS.md) |
| Ternary Bonsai 2 9B | mean KLD <= 1e-4 and same top-1 token >= 99.5%, at ubatch 512, 3 and 1 so that every kernel path is covered | [bonsai2-ternary-pascal](https://github.com/ArtyomITA/bonsai2-ternary-pascal) |

## 3.3 Weights: Q8_0 for a 2B model, Q6_K does not pass

On MiniCPM5-2B, Q6_K weights with an f16 KV cache land at mean KLD 0.00643 and top-1 96.36% against the Q8_0 + f16 reference: seven times the threshold. I dropped them without a speed test. Q8_0 fits in 8 GB with room to spare (6407 MiB peak at 49k context with the draft).

## 3.4 Summation order is part of the format

For the ternary PQ2_0 format, any correct mat-vec with a different order of summation flips some q8_1 roundings. A first attempt that was exact in exact arithmetic landed at KLD 7.3e-4 at ubatch 3, against a gate of 1e-4. All five of my changes to the Bonsai 2 kernels were therefore written to reproduce the order of the generic path bit for bit, and that is why the final build matches the stock logits at the floor (0 / 0 / 1e-6). Details: [docs/01](01-llm-inference-kernels.md#12-ternary-bonsai-2-9b-51-decode-with-the-same-logits).

## 3.5 int8 `dp4a` is the card's fast path

Pascal has no tensor cores, but sm_61 has the packed int8 dot product. ggml's own gate for the int8 paths is `GGML_CUDA_CC_DP4A` (610), which is also the gate I use for CUDA graphs. I use `dp4a` in three places:

- the ternary and Q8 mat-vec kernels in [docs/01](01-llm-inference-kernels.md);
- the Q2_K matrix-matrix path of MiniMax-H3: in the profile of one step 1024 of 3120 SASS instructions in the main loop are useful `dp4a` calls (32.8%), about 11.6 TOPS expected and 11 measured, so that kernel is limited by instruction issue, not by memory;
- an attention kernel for the H3 diffusion transformer: QK^T in int8 with `dp4a`, softmax in fp32, PV in fp32 with V in f16. 2.0x on the attention kernel ([docs/06](06-diffusion-image-and-motion.md#645-8-bit-attention-for-the-diffusion-transformer-code-release-pending)).

## 3.6 KV cache: f16 beats q8_0 when it fits

The q8_0 cache needs a conversion of the whole cache to f16 before every attention call. f16 removes it, costs +944 MiB at 49k context and moved decode from 69.7 to 79.0 tok/s. KL numbers in [docs/01](01-llm-inference-kernels.md#14-kv-cache-in-f16-instead-of-q8_0-136).

## 3.7 Ternary weights cost GUI skills, and a LoRA brings them back

Same 38 clicking cases, same prompt:

| Model | Right clicks |
|---|---|
| Qwen3.5-9B Q4_K_M (same architecture, not ternary) | 36 / 38 |
| Holo 0.8B (a small dedicated GUI model) | 35 / 38 |
| Ternary-Bonsai-2-9B | 19 / 38 |

The full-precision model with the same architecture clicks almost perfectly, the ternary one does not, so the loss comes from the ternarization. A LoRA trained on the frozen ternary weights on the same GTX 1080 gets it back ([docs/05](05-training-and-lora.md)). Full write-up: [ternary-gui-grounding](https://github.com/ArtyomITA/bonsai2-ternary-pascal/blob/master/docs/ternary-gui-grounding.md).

## 3.8 Low-bit weights for a video diffusion transformer

For the MiniMax-H3 diffusion transformer the quantization of every tensor group matters more than the average bit count ([docs/06](06-diffusion-image-and-motion.md)):

- A uniform Q2 DiT produced a ghost-like face and a collapsing face on long clips. Q3_K fixed it.
- A "Q3_K_XL" mix that keeps attention at Q5_K but puts the MLP (most of the weights) and the condition projection at Q2_K looked clearly worse than a plain Q3_K by eye: it has the same weakness as the Q2 model.
- An encoder with only its `ffn_down` tensors at Q3 broke the speech.
- A Q8 video VAE: VAE decode in 24.0 s instead of 34.9 s, with a PSNR of 42.6 dB against the fp16 VAE.
