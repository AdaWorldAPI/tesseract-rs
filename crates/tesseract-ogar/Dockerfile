# tesseract-ogar — the OGAR + lance-graph reasoning-enabled image.
#
# The SECOND, heavier deployment image, alongside tesseract-ocr-web's lean
# one. tesseract-ocr-web deliberately TRIMS tesseract-ogar (and therefore the
# OGAR sibling) out of its build — "no C OCR libraries, no OGAR" is its whole
# selling point. This image is the opposite trim: it KEEPS tesseract-ogar (the
# OGAR-declared capability executor) and its `reasoning`/`sentences` modules
# (deepnsm SPO extraction + NarsTruth beliefs over recognized text — see
# `tesseract-rs/CLAUDE.md`'s "Reasoning layer" section), and runs the
# `ocr_demo` example: recognize a page, then assemble it into sentences and
# extract what SPO triples + beliefs deepnsm can find. Still zero C OCR
# libraries at runtime (the recognizer itself is unchanged pure-Rust) — this
# image is heavier because it carries the OGAR + lance-graph + deepnsm source
# tree through the build, not because it links any new native library.
#
# ── The sibling-repo wrinkle (three siblings here, not two) ─────────────────
# This crate's transitive path deps escape the repo:
#   tesseract-core       → ../../../lance-graph/crates/lance-graph-contract
#   tesseract-recognizer → ../../../ndarray
#   tesseract-ogar       → ../../../OGAR/crates/ogar-vocab
#                        → ../../../lance-graph/crates/deepnsm  (reasoning)
# i.e. cargo expects `lance-graph/`, `ndarray/`, AND `OGAR/` as SIBLINGS of the
# repo (exactly how CI checks them out — see .github/workflows/rust.yml). The
# builder stage clones all three itself with a plain `git clone`; pin with
# LANCE_GRAPH_REF / NDARRAY_REF / OGAR_REF for a reproducible build (default =
# each repo's default branch = CI).
#
# Local build (from the tesseract-rs repo root, using your own git credentials):
#   docker build -f crates/tesseract-ogar/Dockerfile -t ogar-demo .
# Local run (no PORT — this is a one-shot CLI, not a server):
#   docker run --rm ogar-demo

# ── builder ─────────────────────────────────────────────────────────────────
FROM rust:1.97.1-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
# The build context is the tesseract-rs repo root.
COPY . /src/tesseract-rs

# Fetch the three sibling repos next to the checkout (unless already supplied).
# Railway's builder already carries the GitHub credential from the account's
# first-login OAuth, so a plain `git clone` of the private siblings just works
# — no token to pass, no secret mount. Locally, your own git credentials do
# the same. *_REF pins a branch/tag for reproducibility; unset = each repo's
# default branch (what CI uses).
ARG LANCE_GRAPH_REF=""
ARG NDARRAY_REF=""
ARG OGAR_REF=""
RUN set -eu; \
    clone() { \
      name="$1"; ref="$2"; \
      if [ -f "/src/$name/Cargo.toml" ]; then echo "using pre-supplied /src/$name"; return; fi; \
      git clone --depth 1 ${ref:+--branch "$ref"} \
        "https://github.com/AdaWorldAPI/$name.git" "/src/$name"; \
      echo "cloned $name @ ${ref:-default branch}"; \
    }; \
    clone lance-graph "$LANCE_GRAPH_REF"; \
    clone ndarray "$NDARRAY_REF"; \
    clone OGAR "$OGAR_REF"

WORKDIR /src/tesseract-rs
# Trim ONLY tesseract-ocr-python (the pyo3/maturin wheel crate — needs system
# Python headers this image has no use for). tesseract-ogar stays IN the
# workspace this time — it is the whole point of this image, unlike
# tesseract-ocr-web's Dockerfile which trims it out. Cargo loads every
# workspace member's manifest (and its path deps) even when building a single
# `-p` package, so a stale trim would fail loudly rather than silently — the
# guard below makes that a hard build failure instead of a confusing later
# "failed to load pyo3" error.
# Line-delete, not a `, "X"` substring substitution: the root Cargo.toml's
# members list is ONE ENTRY PER LINE with a TRAILING comma
# (`"crates/tesseract-ocr-python",`), so a leading `, "X"` pattern (written
# when this list was single-line) never matches and the guard below fired
# FATAL on every build — the identical root cause
# `tesseract-paperless-web/Dockerfile` already fixed for its own sed line
# (codex P1 on PR #88), measured here on a real Railway build log after this
# file was never updated to match.
RUN sed -i '/"crates\/tesseract-ocr-python"/d' Cargo.toml; \
    if grep -q '"crates/tesseract-ocr-python"' Cargo.toml; then \
      echo "FATAL: failed to trim tesseract-ocr-python from workspace members (Cargo.toml reformatted?)" >&2; \
      exit 1; \
    fi

# Strip the release binary via cargo (no binutils needed in the slim runtime).
ENV CARGO_PROFILE_RELEASE_STRIP=true
# No BuildKit cache mount — Railway's builder rejects the mount syntax, and a
# plain build works on every builder (just no cross-deploy caching).
RUN cargo build --release -p tesseract-ogar --example ocr_demo \
    && cp target/release/examples/ocr_demo /usr/local/bin/ocr_demo

# ── runtime ─────────────────────────────────────────────────────────────────
# debian-slim only for glibc + libgcc; zero OCR/TLS system libraries.
FROM debian:bookworm-slim AS runtime

WORKDIR /app
COPY --from=builder /usr/local/bin/ocr_demo /app/ocr_demo
# The eng model (~4 MB): eng.lstm + unicharset + recoder + the three DAWGs.
COPY --from=builder /src/tesseract-rs/corpus/model /app/model
# deepnsm's 4,096-word COCA vocabulary CSVs — what the reasoning layer's
# SentenceReasoner::from_vocab_dir needs at RUNTIME (a few hundred KB of
# text, not the crate's own target/ build residue).
COPY --from=builder /src/lance-graph/crates/deepnsm/word_frequency /app/word_frequency
# A bundled demo page (same fixture ocr_demo.rs falls back to locally), copied
# to a path that exists in THIS image rather than relying on the compile-time
# CARGO_MANIFEST_DIR-relative default (which points at the builder's /src
# layout and would not resolve here).
COPY --from=builder /src/tesseract-rs/corpus/pages/page_01.pgm /app/page_01.pgm

# Drop root — the app only READS /app and writes nothing to disk.
RUN useradd --system --uid 10001 --no-create-home appuser
USER appuser

ENV MODEL_DIR=/app/model
ENV DEEPNSM_VOCAB_DIR=/app/word_frequency
ENV DEMO_IMAGE=/app/page_01.pgm

# One-shot CLI, not a server — no PORT. Pass a mounted image path as an extra
# `docker run` arg to recognize something other than the bundled page:
#   docker run --rm -v /host/page.pgm:/in.pgm ogar-demo /in.pgm
CMD ["/app/ocr_demo"]
