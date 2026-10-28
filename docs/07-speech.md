# 7. Speech

By **Adrian Ciprian Iancu** · [GitHub](https://github.com/ArtyomITA) · [Hugging Face](https://huggingface.co/adriandj3) · [LinkedIn](https://www.linkedin.com/in/adrian-ciprian-iancu-a2a477391)

Back to the [guide](../README.md).

## 7.1 Local speech-to-text with faster-whisper: Whisper Pad

**Problem.** Sometimes I do not need an assistant, a cloud service or a browser tab. I need a small desktop window that turns my voice into text fast enough to paste it somewhere else.

**What I did.** [Whisper Pad](https://github.com/ArtyomITA/stt-whisper-pad), a small Windows desktop app: record, stop, copy. It runs [faster-whisper](https://github.com/SYSTRAN/faster-whisper) locally with the `turbo` model, beam size 1 and a VAD filter, on the default microphone. Nothing leaves the machine; the only network access is the first model download.

How it picks the compute type: the engine tries `cuda` with `int8_float16`, then `cuda` with `int8`, then `cpu` with `int8`, and records which attempt failed and why. The int8 modes matter on this card because fp16 arithmetic runs at 1/64 of fp32 ([docs/03](03-quantization-on-sm61.md)).

| Setting | Default |
|---|---|
| model | `turbo` (`small` and `medium` selectable) |
| language | `it` (`en` selectable) |
| VAD filter | on |
| beam size | 1 |
| hotkey | global `Ctrl+M`, auto-transcribe on stop, optional auto-copy |

**Reproduce.**

```powershell
git clone https://github.com/ArtyomITA/stt-whisper-pad.git
cd stt-whisper-pad
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_gui.ps1
```

I have not published timing numbers for it yet, so there is no speed figure in this guide for speech.

**Links.** [stt-whisper-pad](https://github.com/ArtyomITA/stt-whisper-pad).
