# ogar-vocab: mint the 0x08 OCR PDF-to-text plan concepts (0x0805..0x0809)

## Plateau PR — commit carried as patch (OGAR push denied for this session's token)

Branch `claude/ocr-pdf-plan-mints`, base `2e346ea` (merge of #171). One commit:
five container-KIND mints for the tesseract-rs PDF→text integration plan
(`pdf-to-text-ocr-v1.md` Phase 0 D0.3) — the plan is the emit-seam declaration
(mint-on-emit guard honored):

| concept | id | consumer phase |
|---|---|---|
| `textline` | 0x0805 | P1B word/box + P3E line formation |
| `blob` | 0x0806 | P3B/3D connected components |
| `page_layout` | 0x0807 | P3F/4A layout result |
| `page_image` | 0x0808 | P2 decode/threshold input |
| `ocr_renderer` | 0x0809 | P4B-D output kinds (fmt = custom-low) |

All lockstep regions updated (CODEBOOK, consts+ALL, builder Classes,
domain pins OCR 4→9, count fuse 79→84). Gates: 108/108 tests, fmt+clippy clean.

**Pairing (two-sided drift fuse):** merge together with the lance-graph mirror
commit (contract `ogar_codebook` +5 rows, `lance-graph-ogar` COUNT_FUSE 79→84)
— prepared in lance-graph as the paired change; do not merge one side alone.

## How to land
```sh
cd OGAR && git checkout -b claude/ocr-pdf-plan-mints main
git am 0001-*.patch      # or: git fetch <path>/ocr-pdf-plan-mints.bundle
git push -u origin claude/ocr-pdf-plan-mints
```

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_016b33swuXE23hKtqxsHu9p1
