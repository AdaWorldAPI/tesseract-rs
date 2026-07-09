# tesseract-ocr-web

A single-binary web demo for the pure-Rust Tesseract transcode. Upload an image
(or paste an image URL), the server runs the recognizer and returns the text
plus stats, with a one-click `.txt` download.

**The selling point:** OCR with **zero C libraries at runtime** — no
libtesseract, no leptonica, no OpenCV. Image decode (PNG/JPEG/PNM) and TLS
(rustls + bundled webpki roots) are pure Rust too, so the Docker image is just
the glibc-linked binary and ~4 MB of `eng` model data.

## Stack

| Concern | Choice | Why |
|---|---|---|
| HTTP | `axum` 0.7 (+ multipart) | Tower ecosystem, typed extractors |
| Templates | `askama` 0.12 | Compile-time-checked HTML, no runtime engine |
| Runtime | `tokio` | Async, single binary |
| Image decode | `image` 0.25 (png/jpeg/pnm) | Pure Rust, no C deps |
| URL fetch | `reqwest` 0.12 (rustls) | rustls only — never openssl |
| Body cap | axum `DefaultBodyLimit` + `tower-http` | 12 MB upload ceiling (both raised) |
| OCR | `tesseract-ocr` + `tesseract-core` | The pure-Rust recognizer |

## Run locally

The model lives in the repo at `corpus/model`. From the repo root:

```sh
# Binds 0.0.0.0:$PORT; PORT defaults to 8080 for local dev.
cargo run -p tesseract-ocr-web
# → open http://localhost:8080

# Custom port / model dir:
PORT=3000 MODEL_DIR=/path/to/model cargo run -p tesseract-ocr-web
```

- **`file`** upload wins over a **`url`** when both are given.
- The result page shows image size, character/line counts, and recognition
  time, plus a **Download .txt** link (a `data:` URI, no temp files).

## Ports — `$PORT`, not hardcoded

The binary reads `PORT` from the environment and binds `0.0.0.0:$PORT`. `8080`
is **only** the local-dev fallback. On Railway, **Railway injects `PORT`
itself** — so it is deliberately not set in the Dockerfile or `railway.toml`.
Don't add `ENV PORT=...`; it would shadow Railway's value.

## Security — the URL arm is SSRF-guarded

Fetching a user-supplied URL is an SSRF vector, so `fetch_image_url`:

1. allows **http/https only**;
2. resolves the host **once**, **rejects any non-global IP** — loopback, private
   (10/8, 172.16/12, 192.168/16), CGNAT (100.64/10, incl. Alibaba metadata
   `100.100.100.200`), link-local incl. `169.254.169.254` (cloud metadata),
   multicast, reserved (240/4), TEST-NET / benchmarking ranges, `0.0.0.0/8`,
   ULA `fc00::/7`, v6 link-local `fe80::/10` / multicast, and IPv4 embedded in
   IPv6 (mapped / compatible / 6to4 / Teredo);
3. **pins the request to those vetted addresses** (`resolve_to_addrs`) so
   reqwest cannot re-resolve to a different IP at connect time (DNS rebinding),
   and **disables env proxies** (`no_proxy`, else a proxy would connect on our
   behalf);
4. **disables redirects** (a 3xx could bounce past the guard);
5. caps the download at **10 MB / 10 s** (content-length pre-check + a hard
   streaming cap so a lying/omitted length can't OOM the process), and returns
   **generic** errors (detail is logged) so it is not a host/port oracle.

## Resource limits

Recognition is heavy CPU work, so the server protects itself:

- **Decode bounds** — the decoder is capped (max dimension + 256 MiB alloc) and
  the decoded pixel count is rejected above **40 MP** *before* the grey buffer
  is allocated, so a small "decompression bomb" upload can't force a ~GB raster.
- **Off-runtime + bounded** — OCR runs via `spawn_blocking` (never on the async
  worker threads) behind a semaphore sized to the machine's parallelism, so a
  burst of uploads can't stall the healthcheck or oversubscribe CPU/RAM.

## Tests

```sh
cargo test -p tesseract-ocr-web
```

Covers the base64 download encoder (RFC 4648 vectors), the SSRF blocklist on
literal IPs, non-http scheme rejection, a real corpus-page OCR (`page_01.pgm`
→ contains "clock"), and a `GET /` 200 via `tower`'s `oneshot`. Tests that need
the model skip gracefully if `corpus/model` is absent.

## Deploy on Railway

Railway clones a single repo, but this crate's path deps escape it:

```
tesseract-core       → ../../../lance-graph/crates/lance-graph-contract
tesseract-recognizer → ../../../ndarray
```

so the **builder stage clones those two siblings itself** — everything ends up in
one binary. They're private, but **Railway's builder already carries the
account's GitHub credential** (the implicit OAuth from first login), so the plain
`git clone` just works:

- **Railway:** connect the repo and deploy — Railway auto-detects `railway.toml`
  and builds `crates/tesseract-ocr-web/Dockerfile`. **No token variable to set.**
- Optionally pin the siblings with `LANCE_GRAPH_REF` / `NDARRAY_REF` build args
  (default = each repo's default branch, matching CI).

### Local Docker build

The Dockerfile clones the private siblings with a plain `git clone`, so a local
build uses **your own git credentials** (a credential helper, or an `insteadOf` /
`~/.netrc` rewrite for `github.com`). From the repo root:

```sh
docker build -f crates/tesseract-ocr-web/Dockerfile -t ocr-web .

docker run --rm -e PORT=8080 -p 8080:8080 ocr-web
# → http://localhost:8080
```

The final runtime image contains only the binary + model — no source, no
credentials.

## Routes

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Upload/URL form |
| `POST` | `/ocr` | Multipart `file` or `url` → result page |
