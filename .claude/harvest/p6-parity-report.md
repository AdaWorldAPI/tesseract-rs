# tesseract-rs golden parity report (D6.2)

Corpus root: /home/user/tesseract-rs/crates/tesseract-ocr/../../corpus

## Tier A — transcoded chain, byte gates (regression goldens; parity proven vs libtesseract API oracles at landing time)

| fixture | golden bytes | note |
|---|---|---|
| lines/img_100.txt | 2 | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |
| lines/img_16.txt | 2 | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |
| lines/img_24.txt | 3 | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |
| lines/img_40.txt | 3 | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |
| lines/img_64.txt | 5 | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |
| lines/img_8.txt | 1 | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |
| lines/line36.dict.tsv | 221 | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |
| lines/line36.dict.txt | 3 | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |
| lines/line36.txt | 3 | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |
| pages/page_01.txt | 222 | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |
| pages/page_02.txt | 230 | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |
| pages/page_03.txt | 243 | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |
| pages/page_04.txt | 220 | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |
| pages/page_05.txt | 236 | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |
| pages/page_06.txt | 225 | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |
| pages/page_07.txt | 177 | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |
| pages/page_08.txt | 263 | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |
| pages/page_09.txt | 187 | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |
| pages/page_10.txt | 227 | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |
| pdfs/doc_01.txt | 142 | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |
| pdfs/doc_02.txt | 226 | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |
| pdfs/doc_03.txt | 107 | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |
| pdfs/doc_04.txt | 210 | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |
| pdfs/doc_05.txt | 16 | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |

## Tier B — vs C++ tesseract CLI (report only; CLI wraps untranscoded framing: DPI estimation, invert-retry, PSM-13 row normalization, PSM-6 ColumnFinder/tospace)

### Lines (--psm 13)

| fixture | ours | CLI | byte-equal? | CER(ours vs CLI) |
|---|---|---|---|---|
| img_100 | o | La | no | 1.0000 |
| img_16 | p | p | yes | 0.0000 |
| img_24 | y, | i, | no | 0.5000 |
| img_40 | ie | Le | no | 0.5000 |
| img_64 | Viee | Vie | no | 0.3333 |
| img_8 |  |  | yes | 0.0000 |
| line36 | y, | i, | no | 0.5000 |

_Footnote: "ours" and "CLI" above are trimmed of trailing newline(s) ONLY before display and comparison -- no other normalization is applied to the lines table._

### Pages (--psm 6)

| page | CER(ours vs gt) | WER(ours vs gt) | CER(cli vs gt) | WER(cli vs gt) | CER(ours vs cli) |
|---|---|---|---|---|---|
| page_01 | 0.0178 | 0.0952 | 0.0044 | 0.0476 | 0.0223 |
| page_02 | 0.0172 | 0.0909 | 0.0000 | 0.0000 | 0.0172 |
| page_03 | 0.0163 | 0.0889 | 0.0000 | 0.0000 | 0.0163 |
| page_04 | 0.0223 | 0.1220 | 0.0000 | 0.0000 | 0.0223 |
| page_05 | 0.0042 | 0.0238 | 0.0000 | 0.0000 | 0.0042 |
| page_06 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| page_07 | 0.0383 | 0.1892 | 0.0000 | 0.0000 | 0.0383 |
| page_08 | 0.0076 | 0.0444 | 0.0114 | 0.0667 | 0.0115 |
| page_09 | 0.0211 | 0.1143 | 0.0000 | 0.0000 | 0.0211 |
| page_10 | 0.0088 | 0.0526 | 0.0000 | 0.0000 | 0.0088 |

_Footnote: CER/WER in the pages table normalizes each of ours/cli/gt identically before comparing: trim trailing whitespace from every line, drop empty lines (the CLI's --psm 6 output emits blank separator lines between blocks), then rejoin with "\n"._

Summary: mean CER(ours vs gt) = 0.0153, mean CER(cli vs gt) = 0.0016 (n=10)
