# tesseract-ocr-web — Railway-ready image for the pure-Rust OCR demo.
#
# The whole selling point: the runtime image contains NO C OCR libraries — no
# libtesseract, no leptonica, no OpenCV. Just the Rust binary (glibc-linked) and
# ~4 MB of eng model data. Image decode (PNG/JPEG/WebP/TIFF/GIF/BMP/PNM) and TLS (rustls +
# bundled webpki roots) are pure Rust, so the runtime needs no ca-certificates
# and no system TLS.
#
# ── The sibling-repo wrinkle ────────────────────────────────────────────────
# This crate's transitive path deps escape the repo:
#   tesseract-core       → ../../../lance-graph/crates/lance-graph-contract
#   tesseract-recognizer → ../../../ndarray
# i.e. cargo expects `lance-graph/` and `ndarray/` as SIBLINGS of the repo
# (exactly how CI checks them out). Railway clones a single repo, so the builder
# stage clones those two siblings itself with a plain `git clone` — Railway's
# builder already carries the account's GitHub OAuth credential, so the private
# repos need no token. Pin the siblings with LANCE_GRAPH_REF / NDARRAY_REF for a
# reproducible build (default = each repo's default branch = CI).
#
# Local build (from the tesseract-rs repo root, using your own git credentials):
#   docker build -f crates/tesseract-ocr-web/Dockerfile -t ocr-web .
# Local run:  docker run --rm -e PORT=8080 -p 8080:8080 ocr-web

# ── builder ─────────────────────────────────────────────────────────────────
FROM rust:1.95-bookworm AS builder

# ring (the rustls crypto backend selected here) builds some C/asm — the gcc in
# rust:bookworm covers it; no cmake needed (that would be aws-lc-rs).
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
# The build context is the tesseract-rs repo root.
COPY . /src/tesseract-rs

# Fetch the two sibling repos next to the checkout (unless already supplied).
# Railway's builder already carries the GitHub credential from the account's
# first-login OAuth, so a plain `git clone` of the private siblings just works —
# no token to pass, no secret mount. Locally, your own git credentials do the
# same. LANCE_GRAPH_REF / NDARRAY_REF pin a branch/tag for reproducibility;
# unset = each repo's default branch (what CI uses).
ARG LANCE_GRAPH_REF=""
ARG NDARRAY_REF=""
RUN set -eu; \
    clone() { \
      name="$1"; ref="$2"; \
      if [ -f "/src/$name/Cargo.toml" ]; then echo "using pre-supplied /src/$name"; return; fi; \
      git clone --depth 1 ${ref:+--branch "$ref"} \
        "https://github.com/AdaWorldAPI/$name.git" "/src/$name"; \
      echo "cloned $name @ ${ref:-default branch}"; \
    }; \
    clone lance-graph "$LANCE_GRAPH_REF"; \
    clone ndarray "$NDARRAY_REF"

WORKDIR /src/tesseract-rs
# Trim the one workspace member that path-deps the OGAR sibling (the web binary
# needs neither tesseract-ogar nor OGAR), so cargo can resolve without it. The
# guard makes a silent no-op (e.g. if the members line is ever reformatted
# one-entry-per-line) a hard build failure instead of a confusing later
# "failed to load OGAR/Cargo.toml".
RUN sed -i 's#, "crates/tesseract-ogar"##' Cargo.toml; \
    if grep -q '"crates/tesseract-ogar"' Cargo.toml; then \
      echo "FATAL: failed to trim tesseract-ogar from workspace members (Cargo.toml reformatted?)" >&2; \
      exit 1; \
    fi

# Strip the release binary via cargo (no binutils needed in the slim runtime).
ENV CARGO_PROFILE_RELEASE_STRIP=true
# No BuildKit cache mount — Railway's builder rejects the mount syntax, and a
# plain build works on every builder (just no cross-deploy caching).
RUN cargo build --release -p tesseract-ocr-web \
    && cp target/release/tesseract-ocr-web /usr/local/bin/tesseract-ocr-web

# ── runtime ─────────────────────────────────────────────────────────────────
# debian-slim only for glibc + libgcc; zero OCR/TLS system libraries.
FROM debian:bookworm-slim AS runtime

WORKDIR /app
COPY --from=builder /usr/local/bin/tesseract-ocr-web /app/tesseract-ocr-web
# The eng model (~4 MB): eng.lstm + unicharset + recoder + the three DAWGs.
COPY --from=builder /src/tesseract-rs/corpus/model /app/model

# Drop root — the app only READS /app (binary + model) and writes nothing to
# disk (the download is an in-memory data: URI), so a non-root user suffices.
RUN useradd --system --uid 10001 --no-create-home appuser
USER appuser

# MODEL_DIR is safe to pin (it is not the port); the binary reads it at startup.
ENV MODEL_DIR=/app/model
# NOTE: the listen port is intentionally NOT set here. The binary binds
# 0.0.0.0:$PORT at runtime; Railway injects PORT itself. For a local run pass
# `-e PORT=8080`. Do not add `ENV PORT=...` — that would shadow Railway's value.

CMD ["/app/tesseract-ocr-web"]
