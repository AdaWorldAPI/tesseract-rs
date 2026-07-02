# tesseract-rs

A **pure-Rust transcode** of [Tesseract](https://github.com/tesseract-ocr/tesseract) OCR — *not* a binding to the C++ library.

The original `antimatter15` FFI wrapper (`tesseract-sys` / `tesseract-plumbing`) was removed on 2026-06-18 per the operator directive: **transcode Tesseract into Rust, do not wrap libtesseract.** The OCR is rebuilt leaf-by-leaf in pure Rust, each leaf byte-parity-proven against the C++ original (source: `AdaWorldAPI/Tesseract`) before it lands. Transcoded primitives ride the OGAR Core (`lance-graph-contract`) per the Core-First transcode doctrine.

## Layout

- **`crates/tesseract-core`** — the pure-Rust OCR core. It consumes the `UNICHARSET` / `UNICHAR` primitives from the OGAR Core (`lance-graph-contract`), where each is byte-parity-proven against a libtesseract oracle. Landed so far: the id↔unichar bijection (112/112) and the `unichar` UTF-8 codec (268/268), plus the per-id `properties`, `script`, and `other_case` accessors (112/112 each), and `ids_to_text`.

Start with **`CLAUDE.md`** (the session contract: Core-First doctrine, the proven self-validating-oracle method, iron rules, and the next leaf). See `.claude/plans/` for the transcode plan (`tesseract-rs-ast-dll-codegen-v1`, `tesseract-rs-receive-contract-v1`).
