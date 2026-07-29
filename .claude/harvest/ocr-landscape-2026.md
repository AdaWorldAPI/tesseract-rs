# The OCR landscape as of 2026-07 — two ladders, not one

Operator-supplied reference stream, 2026-07-29. Recording it because the
references arrive as one list but describe **two ladders that do not substitute
for each other**, and conflating them would misdirect this repo's roadmap.

## Ladder A — classical front-end (improves what tesseract-rs already is)

`Otsu → Niblack → Sauvola → Wolf-Jolion → Singh et al.`

Detail in `binarization-roadmap.md`. Local adaptive thresholding; each rung is a
different closing formula over the *same* windowed statistics `binarize.rs`
already computes byte-parity green. Cheap to add, CPU-only, no model.

- Wolf-Jolion — <https://github.com/chriswolfvision/local_adaptive_binarization>
- Singh et al., *A New Local Adaptive Thresholding Technique in Binarization* —
  arXiv 1201.5227

## Ladder B — end-to-end VLM (REPLACES the classical pipeline entirely)

| Work | Shape | Notes |
|---|---|---|
| **GOT-OCR2.0** — arXiv 2409.01704 | ~580M unified end-to-end | The "OCR-2.0" framing: one model, image → text, no segmentation/binarization stage at all. |
| **olmOCR** (AllenAI) — <https://github.com/allenai/olmocr> | Qwen2-VL-7B fine-tune + "document anchoring" (PDF-extracted text/layout fed into the prompt alongside the image) | A batch PDF→text *toolkit*, not just a model. **Not read in depth — listed from the repo, not the paper.** |
| **DeepSeek-OCR** — arXiv 2510.18234 | DeepEncoder 380M (SAM-base windowed attn → 16× conv compressor → CLIP-large global attn) + DeepSeek-3B-MoE (570M active) | **96%+ decoding precision at 9-10× text compression**; ~60% at 20×. Beats GOT-OCR2.0 with 100 vision tokens; 200k+ pages/day on one A100-40G. |
| *Unlimited OCR Works…* — arXiv 2606.23050 | one-shot long-horizon parsing | **Title only — not read.** Flagged, not characterized. |
| Sinha & Rekha B S — arXiv 2506.11156 | OCR engine + LLM post-processing → structured key-value + confidence | An integration/pipeline paper, materially lighter than the three above. Its shape (OCR → LLM → structure) is *already* what this repo's `doc.v1` → OGAR seam does. |

## The connection worth naming

**DeepSeek-OCR's thesis is the same claim this workspace already makes, in a
different medium.** "Contexts optical compression" — represent text as vision
tokens at 9-10× compression — is a claim about the *input representation*.
lance-graph/bgz-tensor's thesis (attention as table lookup: weight matrix 64 MB
→ Base17 136 KB → 256 archetypes 8.5 KB → distance table 128 KB) is the same
claim about the *computation*.

They **compose** rather than compete: compressing the token stream and
compressing the attention that consumes it are orthogonal axes.

## What this means for tesseract-rs specifically — the honest read

tesseract-rs's actual value proposition is narrow and real: **byte-parity,
pure-Rust, zero C at runtime, CPU-only, ~4 MB of model.** That is a deployment
story (Railway container, embedded, air-gapped, Power Platform connector), not
an accuracy story. Ladder B wins on accuracy and loses on every axis that
proposition is built from — 380M-7B params, GPU-shaped, no byte-parity notion
at all.

So:

- **Ladder A belongs inside this repo.** It strengthens exactly what the repo
  already claims, reuses proven machinery, and stays CPU-only. This is the
  roadmap.
- **Ladder B does NOT belong inside the transcode.** Putting a VLM in
  `tesseract-rs` would falsify the crate's whole premise and its name.
- **Ladder B has an obvious correct seam anyway:** `doc.v1` was explicitly
  designed as "the OPTIONAL seed a consumer feeds (via OGAR) to
  `lance-graph-arm-discovery` / DeepNSM" (`CLAUDE.md`, the operator-set
  boundary). A VLM arm slots at *that* seam — a sibling producer of `doc.v1`,
  or a consumer that refines it — never inside the recognizer.
- **The workspace is not badly positioned to host one**, if wanted later:
  `ndarray` already carries the SIMD/GEMM + int8 quantization, and
  `tesseract-recognizer` already proves a byte-exact CPU forward pass of a real
  network. A small-VLM CPU arm would be a NEW crate on those foundations, not a
  modification of the transcode.

## Status

Ladder A: Sauvola shipped byte-parity; Wolf + Singh queued (task #32).
Ladder B: recorded as landscape, **no work scheduled**. Two of the five
references are unread (2606.23050, the olmOCR paper) and are labelled as such
above rather than summarized from the title — do not cite them from this doc.
