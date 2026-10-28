# 8. Experimental

By **Adrian Ciprian Iancu** · [GitHub](https://github.com/ArtyomITA) · [Hugging Face](https://huggingface.co/adriandj3) · [LinkedIn](https://www.linkedin.com/in/adrian-ciprian-iancu-a2a477391)

Back to the [guide](../README.md).

## 8.1 FlyToLLM: a whole fruit-fly connectome as the spiking core of a language model

**The idea.** Use the complete male Drosophila central nervous system (male-cns:v1.0, Janelia FlyEM and Google Research, CC-BY) uncut as the recurrent spiking core of a next-token language model: 166,700 neurons, 2,753,975 real connections, synapse signs taken from the neurotransmitter. No anatomical cuts, no pretrained backbone, one GTX 1080. Animation of the pipeline: [artyomita.github.io/flytollm](https://artyomita.github.io/flytollm/).

**How it runs on 8 GB.**

- Leaky integrate-and-fire neurons on the whole graph, binary spikes, surrogate gradients; sign fixed by the neurotransmitter, magnitude initialised from the synapse count and trained.
- Text enters as a 256-dimensional vector per BPE token (vocabulary 4096) injected as current into 17,937 neurons; a small external causal attention (4 heads, 128-token cache) re-enters through the same ports; the answer is read from 2,241 descending, motor and efferent neurons grouped in 77 populations.
- Training: 4 + 4 sub-passes per token, TBPTT of 8 tokens, 5.46 million parameters (almost all synapses), CUDA Graph in FP32. **0.65 s per update, 0.38 s with the fused kernel.**

**Speed work without changing the maths** (from the project report):

| Change | Result |
|---|---|
| fused CUDA kernel (gather, multiply, accumulate in one pass, skips zero spikes) | **x1.73**, equivalent to the reference at 100 and 2000 updates (loss 1e-6, gradient 1e-8) |
| chunking | x1.22 |
| cuSPARSE CSR | x0.60 (slower) |
| FP16 storage | x1.07 |
| INT32 indices, fused Adam | nothing |
| node reordering by (superclass, degree) | x2.4 in time but not equivalent, so not adopted |

A micro-benchmark showed that the atomic accumulation dominates, not the bandwidth of the weights.

**What it measured** (cross-entropy in nats on 16 held-out TinyStories, lower is better; every test pre-registered with an effect threshold of 0.10 nat and a noise floor of 0.01 to 0.02):

| Configuration, 8000 updates unless noted | CE |
|---|---|
| fly, real wiring, sensory ports, 26,247 updates | 3.634 |
| fly, standard configuration (relative threshold, random ports, block readout) | 3.652 |
| standard + Muon 1e-3 on the dense matrices (default candidate) | **3.541** |
| degree-preserving rewired fly | 3.814 |
| configuration model (topology destroyed) | 3.648 |
| GRU 2x256, same data | 3.429 |
| 2-block Transformer, same data | 3.563 |
| Kneser-Ney trigram, same data | 3.410 |

The controls matter more than the headline. With anatomical ports the null models beat the real wiring (rewired 3.814 and configuration model 3.648 against 3.959 at 8000 updates), the synaptic weights barely move during training (0.30% relative change at 8000 updates), and the core behaves as a fixed reservoir with trained interfaces. With the old anatomical ports the trained synapses did contribute (0.243 nat at 26,247 updates). The wiring as a reservoir is worth 1.68 nat: frozen real synapses reach 4.182 against 5.864 for isolated neurons, and training them adds nothing at 2000 updates (4.187). The tests, protocols and reports are all in the repository.

**Reproduce** (one control, 2000 updates; data not included, see the README for the connectome and TinyStories preparation):

```bash
python pretrain_control.py --threshold 10 --head separate --seed 17 --eval-every 500 --checkpoint-every 2000 --fast-mode fused --rewire-kind relthr --rewire-seed 41 --ports random_matched --readout chunks --core-variant lif --optimizer muon --muon-lr 1e-3 --updates 2000 --output results/default_2000.json
```

**Links.** [flytollm](https://github.com/ArtyomITA/flytollm). Python 3.12, PyTorch 2.14 with CUDA 12.6, cupy for the fused kernel.

## 8.2 HILDA: converting an autoregressive model into a discrete diffusion language model

**The idea.** A research stack for discrete diffusion LMs that stays on a consumer GPU: convert an existing AR model (Qwen3-0.6B / 1.7B-Base) with a Warmup-Stable-Decay schedule instead of training from scratch, using a LLaDA 2.1 style editable core (mask-to-token and token-to-token edits), then SFT, RL with verifiable rewards, and an inference-acceleration stack. Everything is meant to be modular and ablatable.

**Why it matters for this card.** It is built and tested on a GTX 1080 (CUDA 12.1, PyTorch 2.4) with a CPU fallback, and it avoids custom CUDA extensions on purpose: CUDA 13 drops offline compilation for Maxwell, Pascal and Volta, so a backend without custom kernels keeps working. No FlashAttention required.

**Status** (from the README roadmap): Stage 0 (WSD conversion, doc-level attention masking, data pipeline), Stage 1 (SFT) and Stage 2 (threshold-edit decoding, S_MODE / Q_MODE presets, FastAPI server, CLI) are done. Stage 3 (RL objectives) and Stage 4 (acceleration) are planned. There are no benchmark numbers yet, so none appear here.

**Links.** [hildanext](https://github.com/ArtyomITA/hildanext).
