"""Generate the banner and the charts of the guide as dark SVG files (standard library only).

    python tools/make_charts.py            # writes assets/banner.svg and assets/charts/*.svg

Every number below is taken from a source linked in the guide (README table or docs/*.md).
Each SVG draws its own dark background, so it reads the same on GitHub's light and dark themes.
"""
import math
import os
from xml.sax.saxutils import escape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GREEN, GREEN_DIM, GREY, TEXT, MUTED, BG1, BG2, RED = "#76b900", "#4d7a00", "#4b535d", "#e6edf3", "#8b949e", "#0b0e11", "#151a20", "#e5534b"
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'SFMono-Regular', Consolas, 'Liberation Mono', monospace"


def svg_open(w, h, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{escape(title)}">'
            f'<title>{escape(title)}</title>'
            f'<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{BG1}"/><stop offset="1" stop-color="{BG2}"/></linearGradient>'
            f'<linearGradient id="gr" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{GREEN_DIM}"/><stop offset="1" stop-color="{GREEN}"/></linearGradient>'
            f'<filter id="glow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>'
            f'<rect width="{w}" height="{h}" rx="14" fill="url(#bg)"/>')


def text(x, y, s, size=14, fill=TEXT, weight="400", anchor="start", family=FONT, opacity=1):
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" opacity="{opacity}">{escape(s)}</text>')


def write(path, body):
    full = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    assert chr(0x2014) not in body and chr(0x2013) not in body, "dash characters are not allowed in the SVG text"
    with open(full, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    print("wrote", path)


# ---------------------------------------------------------------- banner
def banner():
    w, h = 1280, 320
    o = [svg_open(w, h, "NVIDIA GTX 1080, sm_61, AI Full Guide")]
    # a grid of cells that lights up from left to right, like work spread over SMs
    cols, rows, cell, gap, x0, y0 = 13, 6, 28, 8, 780, 44
    for r in range(rows):
        for c in range(cols):
            t = c / (cols - 1)
            noise = (math.sin(r * 12.9898 + c * 78.233) * 43758.5453) % 1  # deterministic pseudo-noise
            lit = t * 0.9 + noise * 0.25 - 0.1
            op = max(0.07, min(1.0, lit))
            fill = GREEN if lit > 0.35 else "#2b323a"
            o.append(f'<rect x="{x0 + c * (cell + gap)}" y="{y0 + r * (cell + gap)}" width="{cell}" height="{cell}" rx="5" fill="{fill}" opacity="{op:.2f}"/>')
    # speedup curve over the grid
    pts = []
    for i in range(0, 61):
        t = i / 60
        px = x0 + t * (cols * (cell + gap) - gap)
        py = y0 + rows * (cell + gap) - 6 - (1 - math.exp(-3.2 * t)) * (rows * (cell + gap) - 30)
        pts.append(f"{px:.1f},{py:.1f}")
    o.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{TEXT}" stroke-width="2.5" opacity="0.85" filter="url(#glow)"/>')
    o.append(text(x0, y0 + rows * (cell + gap) + 22, "before", 13, MUTED, family=MONO))
    o.append(text(x0 + cols * (cell + gap) - gap, y0 + rows * (cell + gap) + 22, "after", 13, GREEN, anchor="end", family=MONO))
    # left text
    o.append(text(48, 58, "PASCAL  /  COMPUTE CAPABILITY 6.1  /  8 GB", 14, GREEN, "700", family=MONO))
    o.append(text(48, 118, "NVIDIA GTX 1080 · sm_61", 50, TEXT, "800"))
    o.append(text(48, 174, "AI Full Guide", 50, GREEN, "800"))
    o.append(text(48, 212, "Everything I made run fast on a 2016 GPU with 8 GB", 22, "#c9d1d9"))
    o.append(text(48, 244, "By Adrian Ciprian Iancu", 15, MUTED))
    # stat chips (numbers from the README table)
    chips = [("2.5x", "MiniCPM5-2B decode"), ("+51%", "Bonsai 2 9B decode"), ("-74%", "MiniMax-H3 clip time")]
    cx = 48
    for big, small in chips:
        cw = 28 + 9.2 * len(small)
        o.append(f'<rect x="{cx}" y="266" width="{cw:.0f}" height="36" rx="18" fill="#10151a" stroke="{GREEN}" stroke-opacity="0.6"/>')
        o.append(text(cx + 16, 290, big, 17, GREEN, "800"))
        o.append(text(cx + 16 + 11 * len(big) + 6, 290, small, 13, "#c9d1d9"))
        cx += cw + 12
    o.append("</svg>")
    write("assets/banner.svg", "".join(o))


# ---------------------------------------------------------------- speedup leaderboard
# (name, detail, before, after, unit, lower_is_better, shown_before, shown_after, factor_label)
ROWS = [
    ("Ternary Bonsai LoRA training", "per example, unpack kernel + shared prefix", 20.2, 2.7, "s", True, "20.2 s", "2.7 s", None),
    ("MiniMax-H3 test clip", "320x576, 39 frames, streaming from disk", 674.5, 175.5, "s", True, "674.5 s", "175.5 s", None),
    ("MiniCPM5-2B decode", "agent replay, four changes, llama.cpp", 33.4, 83.5, "tok/s", False, "33.4", "83.5 tok/s", None),
    ("MiniMax-H3 int8 attention kernel", "N = 15,529, dp4a against the f32 kernel", 1875, 944, "ms", True, "1875 ms", "944 ms", None),
    ("FlyToLLM training update", "166,700-neuron spiking core, fused CUDA kernel", 0.65, 0.38, "s", True, "0.65 s", "0.38 s", None),
    ("MiniMax-H3 sparse attention", "15 s clip, per turbo LoRA step, 85% sparsity", 400.5, 242.5, "s", True, "397 to 404 s", "240 to 245 s", "1.65x"),
    ("Ternary Bonsai 2 9B decode", "PQ2_0 + MTP, five CUDA changes, same logits", 55.9, 84.4, "tok/s", False, "55.9", "84.4 tok/s", None),
    ("MiniCPM5-2B full SFT (BAdam)", "backward stops at the active block, identical losses", 354, 531, "tok/s", False, "354", "531 tok/s", None),
    ("CUDA graphs on a MoE", "Ling-3.0-tiny Q6_K, llama.cpp #27721", 41.2, 57.6, "tok/s", False, "41.2", "57.6 tok/s", None),
    ("LFM2.5-2.6B + DSpark draft", "local assistant, decode", 64, 78, "tok/s", False, "64", "78-100", "≤1.6x"),
    ("PQ2_0 mat-vec, merged upstream", "PrismML llama.cpp #368, server decode", 55.7, 65.3, "tok/s", False, "55.7", "65.3 tok/s", None),
    ("CUDA graphs on a dense 9B", "llama.cpp #27721, server decode", 26.3, 28.0, "tok/s", False, "26.3", "28.0 tok/s", None),
]


def factor(r):
    return r[2] / r[3] if r[5] else r[3] / r[2]


def leaderboard():
    rows = sorted(ROWS, key=lambda r: -factor(r))
    w, top, rh = 1000, 104, 60
    h = top + rh * len(rows) + 46
    o = [svg_open(w, h, "Speedup leaderboard on the GTX 1080")]
    o.append(text(32, 46, "Speedup leaderboard", 28, TEXT, "800"))
    o.append(text(32, 72, "GTX 1080 (sm_61), measured before and after. Bars are scaled inside each row.", 14, MUTED))
    bx, bw = 372, 430
    for i, r in enumerate(rows):
        y = top + i * rh
        if i % 2 == 0:
            o.append(f'<rect x="12" y="{y - 6}" width="{w - 24}" height="{rh - 2}" rx="8" fill="#ffffff" opacity="0.035"/>')
        name, detail, b, a, unit, lower, sb, sa, flabel = r
        o.append(text(32, y + 20, name, 16, TEXT, "700"))
        o.append(text(32, y + 40, detail, 12, MUTED))
        mx = 100 if "-100" in sa else max(b, a)  # the draft row draws up to the top of its range
        lb, la = bw * b / mx, bw * a / mx
        o.append(f'<rect x="{bx}" y="{y + 4}" width="{lb:.1f}" height="14" rx="4" fill="{GREY}"/>')
        o.append(text(bx + lb + 8, y + 16, sb, 12, MUTED, family=MONO))
        o.append(f'<rect x="{bx}" y="{y + 24}" width="{la:.1f}" height="14" rx="4" fill="url(#gr)"/>')
        if "-100" in sa:  # the draft case: solid to 78, lighter up to 100
            ext = bw * 100 / mx - la
            o.append(f'<rect x="{bx + la:.1f}" y="{y + 24}" width="{ext:.1f}" height="14" rx="4" fill="{GREEN}" opacity="0.35"/>')
            o.append(text(bx + la + ext + 8, y + 36, sa, 12, GREEN, "700", family=MONO))
        else:
            o.append(text(bx + la + 8, y + 36, sa, 12, GREEN, "700", family=MONO))
        o.append(text(w - 32, y + 33, flabel or f"{factor(r):.2f}x", 22, GREEN, "800", anchor="end"))
    o.append(text(32, h - 18, "Sources: README table and docs/01, 02, 05, 06 and 08. Lower is better for seconds and milliseconds, higher for tok/s.", 12, MUTED))
    o.append("</svg>")
    write("assets/charts/speedup-leaderboard.svg", "".join(o))
    # self-check: the factors printed in the README must hold
    assert abs(factor(ROWS[1]) - 3.84) < 0.01 and abs(factor(ROWS[6]) - 1.51) < 0.01 and abs(factor(ROWS[2]) - 2.50) < 0.01


# ---------------------------------------------------------------- where the 8 GB go
# (label, GB, kind) kind: llm, train, video, over
VRAM = [
    ("MiniCPM5-2B + draft, KV q8_0", 5341 / 1024, "llm", "5341 MiB"),
    ("MiniCPM5-2B + draft, KV f16", 6285 / 1024, "llm", "6285 MiB"),
    ("MiniCPM5-2B + FR-Spec draft, KV f16, 49k ctx", 6407 / 1024, "llm", "6407 MiB"),
    ("LoRA trainer, Ternary Bonsai 9B (reference)", 5.84, "train", "5.84 GB"),
    ("BAdam SFT, seq 1024", 6.9, "train", "6.9 GB"),
    ("BAdam SFT, seq 2048, standard loss", 8.5, "over", "8.5 GB, spills into RAM"),
    ("BAdam SFT, seq 2048, fused loss", 6.4, "train", "6.4 GB"),
    ("BAdam SFT, seq 4096, fused loss", 6.6, "train", "6.6 GB"),
    ("BAdam SFT, seq 8192, fused loss", 7.2, "train", "7.2 GB"),
    ("MiniMax-H3, 15 s 384x672 clip, dense", 7072 / 1024, "video", "7072 MiB"),
]
KIND = {"llm": GREEN, "train": "#5aa0d6", "video": "#c792ea", "over": RED}


def vram_chart():
    w, top, rh = 1000, 118, 40
    h = top + rh * len(VRAM) + 74
    o = [svg_open(w, h, "Peak VRAM of my runs on the 8 GB card")]
    o.append(text(32, 46, "Where the 8 GB go", 28, TEXT, "800"))
    o.append(text(32, 72, "Peak VRAM of runs that fit (or do not) on the card. MiB converted at 1024 per GB; training peaks as reported.", 14, MUTED))
    x0, scale = 400, 64  # 9 GB spans 576 px
    legend = [("LLM serving", "llm"), ("full-parameter training", "train"), ("video diffusion", "video"), ("over the limit", "over")]
    lx = 32
    for lab, k in legend:
        o.append(f'<rect x="{lx}" y="86" width="12" height="12" rx="3" fill="{KIND[k]}"/>')
        o.append(text(lx + 18, 97, lab, 12, MUTED))
        lx += 24 + 7.2 * len(lab)
    limit_x = x0 + 8 * scale
    o.append(f'<line x1="{limit_x}" y1="{top - 12}" x2="{limit_x}" y2="{top + rh * len(VRAM) - 8}" stroke="{RED}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    o.append(text(limit_x, top - 18, "8 GB card", 12, RED, "700", anchor="middle"))
    for i, (lab, gb, kind, shown) in enumerate(VRAM):
        y = top + i * rh
        o.append(text(32, y + 20, lab, 13, TEXT))
        bw = gb * scale
        o.append(f'<rect x="{x0}" y="{y + 4}" width="{bw:.1f}" height="22" rx="5" fill="{KIND[kind]}" opacity="0.9"/>')
        o.append(text(x0 + bw - 8, y + 20, shown, 12, "#ffffff" if kind == "over" else BG1, "700", anchor="end", family=MONO))
    for gbt in range(0, 10):
        xx = x0 + gbt * scale
        o.append(text(xx, top + rh * len(VRAM) + 12, str(gbt), 11, MUTED, anchor="middle", family=MONO))
    o.append(text(32, h - 14, "Sources: RECIPE.md (serving), TRAINING-8GB.md (training), docs/06 section 6.4.7 (video). 1 MiB = 1/1024 GB.", 12, MUTED))
    o.append("</svg>")
    write("assets/charts/peak-vram.svg", "".join(o))


# ---------------------------------------------------------------- MiniMax-H3 speed chain
CHAIN = [
    ("baseline: Q2 DiT, 8 steps", 674.5),
    ("one read stream (-t 1)", 334.2),
    ("turbo LoRA 4 steps + two sd.cpp fixes", 265.7),
    ("next segment read while the GPU computes", 248.9),
    ("VAE in tile groups with prefetch", 221.2),
    ("embedding rows from file, LoRA scales, FA on", 209.2),
    ("Q3 recipe: Q8 VAE, 4B encoder, int8 attention", 175.5),
]


def chain_chart():
    w, top, rh = 1000, 108, 50
    h = top + rh * len(CHAIN) + 52
    o = [svg_open(w, h, "MiniMax-H3 test clip time on the GTX 1080")]
    o.append(text(32, 46, "MiniMax-H3 on 8 GB: 674.5 s to 175.5 s", 28, TEXT, "800"))
    o.append(text(32, 72, "320x576, 39 frames, seed 42, weights streamed from disk. Seconds, lower is better.", 14, MUTED))
    x0, bw = 360, 500
    base = CHAIN[0][1]
    for i, (lab, s) in enumerate(CHAIN):
        y = top + i * rh
        o.append(text(32, y + 24, lab, 14, TEXT))
        wd = bw * s / base
        fill = GREY if i == 0 else "url(#gr)"
        o.append(f'<rect x="{x0}" y="{y + 6}" width="{wd:.1f}" height="26" rx="6" fill="{fill}"/>')
        pct = "" if i == 0 else f"   -{(1 - s / base) * 100:.0f}%"
        o.append(text(x0 + wd + 10, y + 25, f"{s} s{pct}", 14, GREEN if i else MUTED, "700", family=MONO))
    o.append(text(32, h - 16, "Source: my run logs, see docs/06 section 6.4.2. The last row was measured twice: 175.705 s and 175.26 s.", 12, MUTED))
    o.append("</svg>")
    write("assets/charts/h3-speed-chain.svg", "".join(o))
    assert abs((1 - CHAIN[-1][1] / CHAIN[0][1]) * 100 - 74.0) < 0.1


if __name__ == "__main__":
    banner()
    leaderboard()
    vram_chart()
    chain_chart()
