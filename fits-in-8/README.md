# Fits in 8?

By **Adrian Ciprian Iancu** · [GitHub](https://github.com/ArtyomITA) · [Hugging Face](https://huggingface.co/adriandj3) · [LinkedIn](https://www.linkedin.com/in/adrian-ciprian-iancu-a2a477391) · [YouTube](https://www.youtube.com/@artyomadriandj3) · TikTok and Instagram [@artyomita](https://www.instagram.com/artyomita)

Fits in 8? is my short video series on new open models. Each episode takes a few models released in the last weeks and checks one thing: does the file fit on an 8 GB graphics card, like the GTX 1080 this guide is about.

## The rule

- I take the 4-bit GGUF (Q4_K_M when it exists, otherwise the closest 4 or 5-bit file), preferring the one published by the model's own authors.
- The size is the real file size listed on Hugging Face, with all parts added up. For vision and audio models I add the projector file (mmproj) separately.
- FITS: up to 6.5 GB. TIGHT: 6.5 to 8 GB. NO: above 8 GB.
- This is the file size, not total VRAM. The context and the KV cache add more on top, and long chats need more memory. The 6.5 GB margin is my own rule of thumb, not a measurement.

## Episode 1: LFM3 by Liquid AI

| Model | On Hugging Face since | File | GB | On 8 GB |
|---|---|---|---:|---|
| [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) | 23 Oct 2026 | [LFM3-1.7B-Q4_K_M.gguf](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF) | 1.10 | FITS |
| [LiquidAI/LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B) | 19 Oct 2026 | [LFM3-VL-2B-Q4_K_M.gguf](https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF) + mmproj F16 | 1.10 + 0.38 | FITS |
| [LiquidAI/LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B) | 21 Oct 2026 | [LFM3-Omni-2B-Q4_K_M.gguf](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF) + mmproj F16 | 1.10 + 0.95 | FITS |

I built on the previous generation: [NVIDIA Kimodo running with a small LFM2.5 encoder instead of the 16 GB Llama-3-8B one](https://huggingface.co/adriandj3/LFM2.5-to-Llama8B-LLM2Vec-Kimodo).

## Episode 2: NVIDIA

| Model | On Hugging Face since | File | GB | On 8 GB |
|---|---|---|---:|---|
| [nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB) | 5 Oct 2026 | Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf (official) | 6.18 | FITS, just |
| [nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16) | 16 Oct 2026 | [NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF) (official) | 3.67 | FITS |
| [nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4) | 12 Oct 2026 | NVFP4 safetensors, all parts | 75.0 | NO |

The Bolt model has "4GB" in its name, but the official Q4_K_M file is 6.18 GB.

## Where the numbers come from

Sizes and dates (repository creation) are read from the public Hugging Face API (model tree and model info) on 28 October 2026. If a file changes after that date, the table can be out of date: open the link and check the size on the Files tab.

Everything I measured on the GTX 1080 itself is in the [main guide](../README.md).
