# tesseract-paperless-web — Railway-ready image for the document archive.
#
# The SECOND upload-serving image, alongside tesseract-ocr-web's lean OCR
# demo. Same upload story (multipart file / URL fetch, pure-Rust image
# decode, no C image libs at runtime) plus the paperless functions this repo
# added in `tesseract-paperless`: the S-2 dedup gate (hash first, recognize
# only on a miss), `doc.v1` -> `DocIr` via OGAR's `ogar-from-docv1`,
# persistence via `lancedb` — the embedded columnar database `lance-graph`
# itself is built on, consumed straight from crates.io per that repo's own
# upstream-authoritative carve-out (`E-LANCE-IS-UPSTREAM-AUTHORITATIVE-1`),
# NOT the forbidden `lance-graph` engine/planner "brain" crates — AND real
# ranked full-text search (BM25 + highlighted snippets) via a persistent
# Tantivy index (`tesseract-paperless`'s `search` feature; the AdaWorldAPI
# `tantivy` fork, an ordinary on-disk index, NOT the `token` feature's
# resident-lane research seam — see `tesseract-paperless/src/search.rs`'s
# module doc). This is the paperless-ngx-parity move: a document row +
# search index pair, the same shape paperless-ngx itself runs (its DB +
# Whoosh index). Still zero C OCR libraries at runtime — this image is
# heavier than tesseract-ocr-web's only because it carries the OGAR +
# lance-graph source tree through the build (to resolve `tesseract-ogar`,
# `ogar-doc-ir`, `ogar-from-docv1`, `lance-graph-contract`), not because it
# links any new native OCR library. `ingest.rs` also runs `tesseract-ogar`'s
# sentence-assembly + deepnsm SPO/`NarsTruth` reasoning layer over every
# recognized document (a SECOND `RecognizePageWords` pass alongside the
# `RecognizeDocument` call, using the SAME decoded grey buffer), storing the
# result as `spo_json` and rendering it on the document detail page —
# gracefully absent (not a build/boot failure) when the deepnsm vocabulary
# below isn't present at runtime.
#
# ── The sibling-repo wrinkle (three siblings, same as tesseract-ogar's) ─────
# This crate's transitive path deps escape the repo:
#   tesseract-core        → ../../../lance-graph/crates/lance-graph-contract
#   tesseract-recognizer  → ../../../ndarray
#   tesseract-ogar        → ../../../OGAR/crates/ogar-vocab (+ reasoning deps)
#   tesseract-paperless   → ../../../lance-graph/crates/lance-graph-contract
#                          → ../../../OGAR/crates/{ogar-doc-ir,ogar-vocab}
#   tesseract-paperless-web → ../../../OGAR/crates/{ogar-doc-ir,ogar-from-docv1}
# i.e. cargo expects `lance-graph/`, `ndarray/`, AND `OGAR/` as SIBLINGS of the
# repo (exactly how CI checks them out — see .github/workflows/rust.yml). The
# builder stage clones all three itself with a plain `git clone`; pin with
# LANCE_GRAPH_REF / NDARRAY_REF / OGAR_REF for a reproducible build (default =
# each repo's default branch = CI).
#
# NOT a fourth sibling: `tesseract-paperless`'s `search`/`token` features pull
# `tantivy` via a pinned `git` dependency (not the sibling `path` form above),
# specifically so this Dockerfile needs no clone-and-trim step for it — see
# that Cargo.toml entry's own comment. Nothing else in this workspace
# path-deps tantivy, so it has none of the "tesseract-ogar itself needs the
# real directory present" ripple the three siblings above have.
#
# ── The protoc wrinkle (new versus the other two Dockerfiles) ───────────────
# `lance-encoding` (a `lancedb`/`lance` transitive dep, pulled in by the
# `store` feature) has a build script that compiles `.proto` files. Debian's
# `protobuf-compiler` package alone is NOT enough — it ships the `protoc`
# binary but not the well-known includes (`google/protobuf/empty.proto` etc.),
# and the build fails with `File not found` without `libprotobuf-dev` too.
# Measured this session: both packages are required together.
#
# Local build (from the tesseract-rs repo root, using your own git credentials):
#   docker build -f crates/tesseract-paperless-web/Dockerfile -t paperless-web .
# Local run:
#   docker run --rm -e PORT=8080 -p 8080:8080 -v paperless-data:/app/data paperless-web

# ── builder ─────────────────────────────────────────────────────────────────
FROM rust:1.97.1-bookworm AS builder

# `git` for the sibling clones; `protobuf-compiler` + `libprotobuf-dev` for
# `lance-encoding`'s build script (see the wrinkle above — both are required,
# neither alone is sufficient).
RUN apt-get update && apt-get install -y --no-install-recommends \
      git protobuf-compiler libprotobuf-dev \
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
# Python headers this image has no use for). tesseract-ogar and
# tesseract-paperless-web both stay IN the workspace — the whole point of this
# image. Cargo loads every workspace member's manifest (and its path deps)
# even when building a single `-p` package, so a stale trim would fail loudly
# rather than silently — the guard below makes that a hard build failure
# instead of a confusing later "failed to load pyo3" error.
# Line-delete, not a `, "X"` substring substitution: the members list is
# ONE ENTRY PER LINE with a TRAILING comma (`"crates/tesseract-ocr-python",`),
# so a leading `, "X"` pattern (borrowed from tesseract-ocr-web's/
# tesseract-ogar's Dockerfiles, written when this list was single-line) never
# matches here and the guard below fired FATAL on every build (codex P1 on
# PR #88) — caught before merge, not in a Railway deploy log.
RUN sed -i '/"crates\/tesseract-ocr-python"/d' Cargo.toml; \
    if grep -q '"crates/tesseract-ocr-python"' Cargo.toml; then \
      echo "FATAL: failed to trim tesseract-ocr-python from workspace members (Cargo.toml reformatted?)" >&2; \
      exit 1; \
    fi

# Strip the release binary via cargo (no binutils needed in the slim runtime).
ENV CARGO_PROFILE_RELEASE_STRIP=true
# No BuildKit cache mount — Railway's builder rejects the mount syntax, and a
# plain build works on every builder (just no cross-deploy caching).
RUN cargo build --release -p tesseract-paperless-web \
    && cp target/release/tesseract-paperless-web /usr/local/bin/tesseract-paperless-web

# ── runtime ─────────────────────────────────────────────────────────────────
# debian-slim only for glibc + libgcc; zero OCR/TLS system libraries.
FROM debian:bookworm-slim AS runtime

WORKDIR /app
COPY --from=builder /usr/local/bin/tesseract-paperless-web /app/tesseract-paperless-web
# The eng model (~4 MB): eng.lstm + unicharset + recoder + the three DAWGs.
COPY --from=builder /src/tesseract-rs/corpus/model /app/model
# deepnsm's 4,096-word COCA vocabulary CSVs — the reasoning layer's SPO/
# `NarsTruth` extraction ingest.rs runs alongside recognition (same file
# tesseract-ogar's own Dockerfile already copies for its ocr_demo). Absence
# here is NOT a build failure -- `AppState::load` degrades to
# `reasoner: None` and archiving proceeds with `spo_json: None` -- this COPY
# exists so a Railway deploy gets the real reasoning layer, not the
# degraded fallback, by default.
COPY --from=builder /src/lance-graph/crates/deepnsm/word_frequency /app/word_frequency

# Both the lancedb archive AND the Tantivy search index need a writable,
# ideally PERSISTENT directory — on Railway, mount a volume at /app/data
# (Railway's volume UI lets you pick the mount path; point it here) or both
# are lost on every redeploy. Both live under the SAME mount so one volume
# covers both stores. Created and owned by the non-root runtime user up
# front so `lancedb::connect` / `SearchIndex::open_or_create` can create
# their files on first boot without a permissions surprise.
RUN mkdir -p /app/data \
    && useradd --system --uid 10001 --no-create-home appuser \
    && chown -R appuser:appuser /app/data
USER appuser

ENV MODEL_DIR=/app/model
ENV ARCHIVE_URI=/app/data/archive.lance
ENV SEARCH_INDEX_DIR=/app/data/search_index
ENV DEEPNSM_VOCAB_DIR=/app/word_frequency
# NOTE: the listen port is intentionally NOT set here. The binary binds
# 0.0.0.0:$PORT at runtime; Railway injects PORT itself. For a local run pass
# `-e PORT=8080`. Do not add `ENV PORT=...` — that would shadow Railway's value.

CMD ["/app/tesseract-paperless-web"]
