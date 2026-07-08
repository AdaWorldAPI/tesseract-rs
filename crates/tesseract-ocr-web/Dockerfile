# syntax=docker/dockerfile:1
#
# tesseract-ocr-web — Railway-ready image for the pure-Rust OCR demo.
#
# The whole selling point: the runtime image contains NO C OCR libraries — no
# libtesseract, no leptonica, no OpenCV. Just the Rust binary (glibc-linked) and
# ~4 MB of eng model data. Image decode (PNG/JPEG/PNM) and TLS (rustls +
# bundled webpki roots) are pure Rust, so the runtime needs no ca-certificates
# and no system TLS.
#
# ── The sibling-repo wrinkle ────────────────────────────────────────────────
# This crate's transitive path deps escape the repo:
#   tesseract-core       → ../../../lance-graph/crates/lance-graph-contract
#   tesseract-recognizer → ../../../ndarray
# i.e. cargo expects `lance-graph/` and `ndarray/` as SIBLINGS of the repo
# (exactly how CI checks them out). Railway clones a single repo, so the builder
# stage fetches those two siblings itself. Provide a GitHub token as either a
# BuildKit secret (`--secret id=gh_token,src=...`, preferred) or a `GH_TOKEN`
# build arg (Railway build variable). Pin the siblings with LANCE_GRAPH_REF /
# NDARRAY_REF if you need a reproducible build (default = each repo's default
# branch, which is what CI uses).
#
# Local build (from the tesseract-rs repo root):
#   DOCKER_BUILDKIT=1 docker build -f crates/tesseract-ocr-web/Dockerfile \
#     --secret id=gh_token,src=<(printf %s "$GH_TOKEN") -t ocr-web .
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
# Railway makes the service's variables available at build time, so a token with
# access to the private repos (the one Railway's GitHub login already granted)
# drops straight in as GH_TOKEN or GITHUB_TOKEN — no per-repo setup. Locally,
# prefer the BuildKit secret.
ARG GH_TOKEN=""
ARG GITHUB_TOKEN=""
ARG LANCE_GRAPH_REF=""
ARG NDARRAY_REF=""
RUN --mount=type=secret,id=gh_token,required=false set -eu; \
    TOKEN="$(cat /run/secrets/gh_token 2>/dev/null || printf '%s' "${GH_TOKEN:-${GITHUB_TOKEN:-}}")"; \
    clone() { \
      name="$1"; ref="$2"; \
      if [ -f "/src/$name/Cargo.toml" ]; then echo "using pre-supplied /src/$name"; return; fi; \
      [ -n "$TOKEN" ] || { echo "FATAL: /src/$name absent and no GH_TOKEN/GITHUB_TOKEN (build secret or arg)" >&2; exit 1; }; \
      git clone --depth 1 ${ref:+--branch "$ref"} \
        "https://x-access-token:${TOKEN}@github.com/AdaWorldAPI/$name.git" "/src/$name"; \
      echo "cloned $name @ ${ref:-default branch}"; \
    }; \
    clone lance-graph "$LANCE_GRAPH_REF"; \
    clone ndarray "$NDARRAY_REF"

WORKDIR /src/tesseract-rs
# Trim the one workspace member that path-deps the OGAR sibling (the web binary
# needs neither tesseract-ogar nor OGAR), so cargo can resolve without it.
RUN sed -i 's#, "crates/tesseract-ogar"##' Cargo.toml

# Strip the release binary via cargo (no binutils needed in the slim runtime).
ENV CARGO_PROFILE_RELEASE_STRIP=true
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/tesseract-rs/target \
    cargo build --release -p tesseract-ocr-web \
    && cp target/release/tesseract-ocr-web /usr/local/bin/tesseract-ocr-web

# ── runtime ─────────────────────────────────────────────────────────────────
# debian-slim only for glibc + libgcc; zero OCR/TLS system libraries.
FROM debian:bookworm-slim AS runtime

WORKDIR /app
COPY --from=builder /usr/local/bin/tesseract-ocr-web /app/tesseract-ocr-web
# The eng model (~4 MB): eng.lstm + unicharset + recoder + the three DAWGs.
COPY --from=builder /src/tesseract-rs/corpus/model /app/model

# MODEL_DIR is safe to pin (it is not the port); the binary reads it at startup.
ENV MODEL_DIR=/app/model
# NOTE: the listen port is intentionally NOT set here. The binary binds
# 0.0.0.0:$PORT at runtime; Railway injects PORT itself. For a local run pass
# `-e PORT=8080`. Do not add `ENV PORT=...` — that would shadow Railway's value.

CMD ["/app/tesseract-ocr-web"]
