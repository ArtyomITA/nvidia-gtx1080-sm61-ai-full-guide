# 10. In progress / not yet published

By **Adrian Ciprian Iancu** · [GitHub](https://github.com/ArtyomITA) · [Hugging Face](https://huggingface.co/adriandj3) · [LinkedIn](https://www.linkedin.com/in/adrian-ciprian-iancu-a2a477391)

Back to the [guide](../README.md). One line each; numbers appear here only after the code or the data is public.

- **MiniMax-H3 code release.** The int8 attention kernel, the sparse-attention port with the conditioning prefix, the FLUX.3 graph cuts, the motion-context chain for long clips and the weight cache are written and tested; the code release is pending. The results are in [docs/06](06-diffusion-image-and-motion.md).
- **stable-diffusion.cpp streaming pull request.** Ready locally on upstream 95ff762, not opened yet ([docs/09](09-upstream-contributions.md)).
- **YuE2 music with vocals.** A CUDA build of `yue2.cpp` for sm_61 and a ComfyUI package with presets and five modes. Works on the 1080; I am working on the quality of the output before I publish.
- **Meeting capture.** A local capture tool on faster-whisper `large-v3-turbo` int8 that writes notes and tasks. Built for my own use; not published.
- **Avatar video.** Tests of two more talking-avatar video models on the card (a face-region regeneration model and a long-form avatar model) with Pascal patches (fp16 attention, memory-mapped weights). Not published.
- **Ling-3.0-tiny.** Decode analysis of the MoE on this card beyond what is in [docs/01](01-llm-inference-kernels.md): kernel counts per token and the routing fusion. Notes not yet published.
- **TernaryPilot v2 benchmarks.** Running on a larger GPU; the model card is updated when they finish ([card](https://huggingface.co/adriandj3/TernaryPilot-v2-Computer-Browser-Use-LoRA)).
