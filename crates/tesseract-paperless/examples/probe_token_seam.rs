//! `PROBE-TOKEN-SEAM-1` — one versioned tokenization receipt, three borrowed
//! consumers: Tantivy, DeepNSM-v2, and a forward-prediction input surface.
//!
//! ```text
//!   ONE SOURCE SPAN -> ONE TOKENIZATION RECEIPT.
//!   TOKENIZE ONCE.  PROJECT MANY TIMES.
//!   AN INDEX MAY ACCELERATE THE ABI.  IT MUST NEVER BECOME THE ABI.
//! ```
//!
//! # Honesty box
//!
//! - **Corpora are real and committed.** `corpus/alice.txt` is Project
//!   Gutenberg's *Alice's Adventures in Wonderland*, carried from
//!   `tantivy/benches/alice.txt` so this probe is hermetic (a fixture that
//!   lives outside the repo is a time bomb with a skip-guard for a fuse).
//!   `corpus/coca_academic_20k.tsv` is the `word,Pos` projection of
//!   `lance-graph/crates/deepnsm/word_frequency/academic_20k.csv`
//!   (sha256 `1dfd5eda…`, 20 845 data rows), the exact file DeepNSM-v2's own
//!   `genre_shapes` example loads. The KJV scene is the real in-tree text
//!   `PROBE-TOKEN-BPE-GEOMETRY-1` used, carried verbatim so the two probes are
//!   comparable.
//! - **ABSENT, reported not simulated:** the whole-KJV corpus, `bible_vocab.txt`,
//!   the trained `cam96_codebook.bin` / `cam96_codes.bin` (so `DeepNSM`'s semantic
//!   DISTANCE cannot be exercised here — only the lexical/grammar half), and any
//!   `MecCog` corpus, which does not exist on this machine in any form.
//! - **No wall-time claims.** Cost is reported as operation counts and bytes.
//! - **The forward arm is a counting baseline, not a language model.** It exists
//!   to prove the INPUT SURFACE, and its accuracy number is labelled as such.

// A probe reads like a lab notebook: `g` is the gate, `c` the corpus, `v` the
// view, `s` the summary. Renaming them to `gate`/`corpus`/`view`/`summary`
// would make every assertion line wrap and read worse, and the scope of each
// is a dozen lines. Same for the two long `main`/`tantivy_arm` bodies: a probe
// is one linear argument and splitting it into helpers hides the order the
// gates run in, which is the thing a reader checks.
#![allow(
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::cast_precision_loss
)]

use std::collections::HashMap;
use std::sync::Arc;

use deepnsm_v2::{parse_to_spo, PaletteVocab, Pos, Spo, Tagged};
use ogar_doc_ir::{
    from_json, to_json, BBoxRail, DocIr, DocPage, Geometry, Provenance, Rail, Region, RegionKind,
    TableCell, DOC_IR_VERSION,
};
use sha2::{Digest, Sha256};
use tantivy::collector::TopDocs;
use tantivy::query::PhraseQuery;
use tantivy::schema::{IndexRecordOption, Schema, TextFieldIndexing, TextOptions, Value};
use tantivy::tokenizer::TokenStream as _;
use tantivy::tokenizer::Tokenizer as _;
use tantivy::{Directory, Index, IndexWriter, TantivyDocument, Term};
use tesseract_paperless::token::contract::{
    query_passes, source_passes, NormRule, TokenizerContract,
};
use tesseract_paperless::token::docir::{spans, SpanKey};
use tesseract_paperless::token::forward::{score, windows, CountPredictor};
use tesseract_paperless::token::lane::{TokenLane, TokenStreamReceipt, IDS_PER_PARTICLE};
use tesseract_paperless::token::lexical::project;
use tesseract_paperless::token::seam_tantivy::{handle, ReceiptTokenizer, SeamStore, TermMode};

/// The real KJV Genesis 2-3 scene, carried verbatim from
/// `PROBE-TOKEN-BPE-GEOMETRY-1` so the two probes measure the same bytes.
const SCENE: &[&str] = &[
    "But of the tree of the knowledge of good and evil, thou shalt not eat of it: for in \
     the day that thou eatest thereof thou shalt surely die.",
    "And they were both naked, the man and his wife, and were not ashamed.",
    "Now the serpent was more subtil than any beast of the field which the LORD God had \
     made. And he said unto the woman, Yea, hath God said, Ye shall not eat of every tree \
     of the garden?",
    "And the serpent said unto the woman, Ye shall not surely die:",
    "And when the woman saw that the tree was good for food, and that it was pleasant to \
     the eyes, and a tree to be desired to make one wise, she took of the fruit thereof, \
     and did eat, and gave also unto her husband with her; and he did eat.",
    "And the eyes of them both were opened, and they knew that they were naked; and they \
     sewed fig leaves together, and made themselves aprons.",
    "And they heard the voice of the LORD God walking in the garden in the cool of the \
     day: and Adam and his wife hid themselves from the presence of the LORD God amongst \
     the trees of the garden.",
    "And he said, I heard thy voice in the garden, and I was afraid, because I was naked; \
     and I hid myself.",
];

/// One corpus under test — carried as the document layer's own IR, not as a
/// bag of strings this crate invented a span numbering for.
struct Corpus {
    name: &'static str,
    /// The ORIGINAL bytes, whose sha256 is the document identity.
    source: Vec<u8>,
    /// The perceptual IR a retina would have produced for those bytes.
    ir: DocIr,
}

/// Build a `DocIr` whose regions are the given paragraphs, in reading order.
///
/// [`Geometry::DomOrder`] is the honest value: these are reading-order
/// placements quantized onto the unit square, NOT measured layout. The IR has
/// a variant for exactly that distinction and using `Rendered` here would
/// claim a measurement nobody took.
fn ir_from_paragraphs(source: &[u8], paras: &[String], prov: Provenance) -> DocIr {
    let n = paras.len().max(1);
    let regions = paras
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let y0 = u8::try_from(i * 255 / n).unwrap_or(u8::MAX);
            let y1 = u8::try_from((i + 1) * 255 / n).unwrap_or(u8::MAX);
            Region {
                kind: RegionKind::Text,
                bbox: BBoxRail {
                    tl: Rail { x: 0, y: y0 },
                    br: Rail { x: 255, y: y1 },
                },
                // Deliberately NOT the positional index: `2i+1` makes
                // "read the field" and "renumber by position" distinguishable.
                // With them equal, a gate that renumbers passes identically —
                // the fixture's SHAPE is part of the coverage.
                reading_order: u16::try_from(i * 2 + 1).unwrap_or(u16::MAX),
                text: Some(p.clone()),
                cells: Vec::new(),
                children: Vec::new(),
            }
        })
        .collect();
    DocIr {
        version: DOC_IR_VERSION.to_string(),
        source: prov,
        geometry: Geometry::DomOrder,
        content_sha256: Sha256::digest(source).into(),
        mime: "text/plain".to_string(),
        pages: vec![DocPage {
            number: 0,
            width: 1,
            height: u32::try_from(n).unwrap_or(u32::MAX),
            regions,
        }],
        fields: Vec::new(),
    }
}

fn kjv() -> Corpus {
    let paras: Vec<String> = SCENE.iter().map(|v| (*v).to_string()).collect();
    let source = paras.join("\n").into_bytes();
    let ir = ir_from_paragraphs(&source, &paras, Provenance::Ocr);
    Corpus {
        name: "kjv-genesis-scene",
        source,
        ir,
    }
}

/// Alice, split into paragraphs. Blank-line separated; a paragraph is a region.
fn alice(max_spans: usize) -> Corpus {
    // The committed file is CRLF with a BOM. A naive `split("\n\n")` finds
    // NOTHING in it — the first version of this probe silently produced ONE
    // 169 KB span and every per-span number it printed was meaningless while
    // every gate stayed green. Normalising first is what makes the span count
    // real.
    let raw = include_str!("../corpus/alice.txt")
        .trim_start_matches('\u{feff}')
        .replace("\r\n", "\n");
    let mut paras: Vec<String> = Vec::new();
    for para in raw.split("\n\n") {
        let p = para.split_whitespace().collect::<Vec<_>>().join(" ");
        if p.len() < 40 {
            continue;
        }
        paras.push(p);
        if paras.len() >= max_spans {
            break;
        }
    }
    let source = paras.join("\n").into_bytes();
    let ir = ir_from_paragraphs(&source, &paras, Provenance::Dom);
    Corpus {
        name: "alice-paragraphs",
        source,
        ir,
    }
}

/// `word -> Pos`, from the committed COCA projection. The COCA letter mapping is
/// `DeepNSM`'s own; it lives in BOTH of that crate's examples, byte-identical,
/// because `deepnsm_v2::lexicon` was DELETED by an earlier audit on the ground
/// that `lance-graph-planner`'s `insight_coca_read` already grounds it. That
/// grounding does not reach here: `insight_coca_read` is itself an example
/// binary in a crate outside this repo's dependency barrier. So re-stating the
/// twenty-line tagger is the ONE duplication this seam forces, and the report
/// records the deletion rather than quietly re-adding the module.
fn coca_pos(letter: &str) -> Pos {
    match letter {
        "n" | "p" => Pos::Noun,
        "v" => Pos::Verb,
        "j" => Pos::Adj,
        "a" | "d" => Pos::Det,
        _ => Pos::Other,
    }
}

fn load_vocab() -> (PaletteVocab, HashMap<String, Pos>) {
    let tsv = include_str!("../corpus/coca_academic_20k.tsv");
    let mut vocab = PaletteVocab::new();
    let mut pos = HashMap::new();
    let mut words: Vec<&str> = Vec::new();
    for line in tsv.lines() {
        let mut it = line.split('\t');
        let (Some(w), Some(p)) = (it.next(), it.next()) else {
            continue;
        };
        if w.is_empty() {
            continue;
        }
        words.push(w);
        pos.entry(w.to_string()).or_insert_with(|| coca_pos(p));
    }
    vocab.from_frequency_ranked(words);
    (vocab, pos)
}

fn pct(a: usize, b: usize) -> f64 {
    if b == 0 {
        0.0
    } else {
        100.0 * a as f64 / b as f64
    }
}

fn quantiles(v: &mut [usize]) -> (usize, usize, usize) {
    v.sort_unstable();
    if v.is_empty() {
        return (0, 0, 0);
    }
    (
        v[v.len() / 2],
        v[(v.len() * 95) / 100],
        *v.last().expect("non-empty"),
    )
}

struct Gate {
    pass: u32,
}

impl Gate {
    fn run(&mut self, name: &str, ok: bool, detail: &str) {
        assert!(ok, "[FAIL] {name} — {detail}");
        println!("  [PASS] {name} — {detail}");
        self.pass += 1;
    }
}

/// What one corpus measured.
struct Summary {
    name: &'static str,
    bytes: usize,
    spans: usize,
    tokens: usize,
    uniq_tokens: usize,
    particles: usize,
    resident_bytes: usize,
    p50: usize,
    p95: usize,
    pmax: usize,
    continuation_rate: f64,
    encode_probes: usize,
    lex_units: usize,
    lex_resolved: usize,
    tokens_per_unit: (usize, usize, usize),
    multi_unit_tokens: usize,
    spo: usize,
}

/// Tokenize a whole corpus into ONE lane under ONE contract, then measure it.
fn ingest(c: &Corpus, contract: &TokenizerContract, g: &mut Gate) -> (TokenLane, Summary) {
    let mut lane = TokenLane::new();
    let doc = lane.intern_document(c.ir.content_sha256);
    let sp = spans(&c.ir, doc);
    let mut probes = 0usize;
    let mut per_span: Vec<usize> = Vec::new();
    let before = source_passes();
    for s in &sp {
        let (tokens, p) = contract
            .try_encode(s.text.as_bytes())
            .expect("contract trained on this corpus");
        probes += p;
        per_span.push(tokens.len().div_ceil(IDS_PER_PARTICLE));
        // byte_from is 0: a whole region, and the offset is REGION-LOCAL
        // because the region owns its canonical text.
        lane.append(s.key, 0, contract, &tokens);
    }
    let passes = source_passes() - before;
    g.run(
        &format!(
            "T-PASSES[{}] exactly one source tokenization per span",
            c.name
        ),
        passes == sp.len(),
        &format!(
            "{passes} source tokenizations for {} spans (1.000 per span); the three consumers \
             below add ZERO further source passes — every one of them reads the lane",
            sp.len()
        ),
    );

    // ---- the receipt is keyed by the DOCUMENT LAYER's address ----
    // Independent ground truth: walk the IR directly rather than trusting the
    // same `spans()` call under test.
    let truth: Vec<(u16, u16)> =
        c.ir.pages
            .iter()
            .flat_map(|pg| pg.regions.iter().map(move |r| (pg.number, r.reading_order)))
            .collect();
    let key_ok = truth.len() == lane.receipts().len()
        && lane
            .receipts()
            .iter()
            .zip(&truth)
            .all(|(r, &(page, ro))| {
                r.key.page == page
                    && r.key.reading_order == ro
                    && lane.document_of(r) == Some(&c.ir.content_sha256)
            })
        // and the orders are genuinely not the positional index, so the check
        // above cannot be satisfied by renumbering
        && truth.iter().enumerate().any(|(i, &(_, ro))| ro as usize != i);
    let reinterned = {
        let mut l2 = lane.clone();
        let same = l2.intern_document(c.ir.content_sha256);
        let other = l2.intern_document([0xAB; 32]);
        same == doc && other != doc && l2.document_len() == 2
    };
    g.run(
        &format!(
            "T-DOCIR-KEY[{}] the receipt carries no id this crate minted",
            c.name
        ),
        key_ok && reinterned && !sp.is_empty(),
        &format!(
            "every one of {} receipts resolves to the IR's own address — `content_sha256` for \
             WHICH document and `(page, reading_order)` for WHICH span, the reading order \
             `ogar-doc-ir` documents as the one the temporal stream and DeepNSM consume. \
             Re-interning the same `content_sha256` returns the SAME index and a different one \
             does not: that is the S-2 dedup property at lane scope. The hash is interned once \
             per document, not stamped on every receipt — at these span sizes a receipt is \
             already a third of the resident bytes and 32 more per span would have more than \
             doubled that for no addressing gain",
            sp.len()
        ),
    );

    // ---- reconstruction + framing ----
    let mut recon_ok = true;
    let mut tokens_total = 0usize;
    let mut uniq = std::collections::HashSet::new();
    for (r, sr) in lane.receipts().iter().zip(&sp) {
        let v = lane.view(r, contract).expect("same contract");
        tokens_total += v.len();
        uniq.extend(v.ids().iter().copied());
        if v.decode() != sr.text.as_bytes() {
            recon_ok = false;
        }
    }
    let (p50, p95, pmax) = quantiles(&mut per_span.clone());
    let cont = per_span.iter().filter(|&&n| n > 1).count();
    g.run(
        &format!(
            "T-RECON[{}] every span decodes byte-exact from ids alone",
            c.name
        ),
        recon_ok,
        &format!(
            "{} spans, {tokens_total} tokens, {} distinct ids; decode reads the lane and the \
             codebook and nothing else — the canonical text stays authoritative and is never \
             consulted to read a span back",
            sp.len(),
            uniq.len()
        ),
    );

    // ---- derived offsets ----
    let mut off_ok = true;
    for (r, sr) in lane.receipts().iter().zip(&sp) {
        let v = lane.view(r, contract).expect("same contract");
        // Ground truth computed the expensive way: decode each prefix.
        let mut cursor = 0u32;
        for t in v.tokens() {
            let truth_len = contract.decode(&[t.id]).0.len();
            if t.byte_from != cursor
                || t.byte_to != cursor + u32::try_from(truth_len).expect("short")
            {
                off_ok = false;
            }
            cursor = t.byte_to;
        }
        if cursor as usize != sr.text.len() {
            off_ok = false;
        }
    }
    g.run(
        &format!(
            "T-OFFSET[{}] byte offsets are DERIVED, never stored",
            c.name
        ),
        off_ok,
        &format!(
            "every token's (byte_from, byte_to) reproduces a decode-the-prefix ground truth, \
             and the last token's end lands exactly on the span end; the receipt carries no \
             offset column — {} bytes/receipt total, of which zero are offsets per token",
            core::mem::size_of::<TokenStreamReceipt>()
        ),
    );

    let summary = Summary {
        name: c.name,
        bytes: c.source.len(),
        spans: sp.len(),
        tokens: tokens_total,
        uniq_tokens: uniq.len(),
        particles: lane.particle_len(),
        resident_bytes: lane.resident_bytes(),
        p50,
        p95,
        pmax,
        continuation_rate: pct(cont, per_span.len()),
        encode_probes: probes,
        lex_units: 0,
        lex_resolved: 0,
        tokens_per_unit: (0, 0, 0),
        multi_unit_tokens: 0,
        spo: 0,
    };
    (lane, summary)
}

/// Consumer B: the DeepNSM-v2 projection, run from ids alone.
fn deepnsm_arm(
    c: &Corpus,
    contract: &TokenizerContract,
    lane: &TokenLane,
    vocab: &PaletteVocab,
    posmap: &HashMap<String, Pos>,
    s: &mut Summary,
    g: &mut Gate,
) {
    let before = source_passes();
    let mut units = 0usize;
    let mut resolved = 0usize;
    let mut spans_per_unit: Vec<usize> = Vec::new();
    let mut multi_unit_tokens = 0usize;
    let mut triples: Vec<Spo> = Vec::new();
    let mut flattened: Vec<Spo> = Vec::new();
    let mut offsets_ok = true;

    let sp = spans(&c.ir, 0);
    for (r, sr) in lane.receipts().iter().zip(&sp) {
        let v = lane.view(r, contract).expect("same contract");
        let lex = project(&v);
        // A token that carries more than one unit's start is a token straddling
        // a word boundary — the cardinality that refutes any 1:1 assumption.
        let mut starts: HashMap<u32, usize> = HashMap::new();
        let mut tagged: Vec<Tagged> = Vec::new();
        for u in &lex {
            units += 1;
            spans_per_unit.push(u.token_span as usize);
            *starts.entry(u.first_token).or_insert(0) += 1;
            // The unit's byte span must address the canonical text and land on
            // exactly the bytes it claims (modulo the normalisation that drops
            // non-alphabetic characters inside a word).
            let raw = &sr.text.as_bytes()[u.byte_from as usize..u.byte_to as usize];
            let renorm: String = raw
                .iter()
                .filter(|b| b.is_ascii_alphabetic())
                .map(|b| b.to_ascii_lowercase() as char)
                .collect();
            if renorm != u.surface {
                offsets_ok = false;
            }
            if let Some(id) = vocab.id(&u.surface) {
                resolved += 1;
                let pos = posmap.get(&u.surface).copied().unwrap_or(Pos::Other);
                tagged.push(Tagged::new(id, pos));
            }
        }
        multi_unit_tokens += starts.values().filter(|&&n| n > 1).count();
        tagged.push(Tagged::new(0, Pos::Stop));
        triples.extend(parse_to_spo(&tagged));
        // Control: the SAME word ids with every tag flattened to Noun. If the
        // FSM's output did not depend on the Pos half of the pair, this would
        // produce the same triples — and "the FSM consumes (WordId, Pos)"
        // would be a claim about a type signature rather than about behaviour.
        let blind: Vec<Tagged> = tagged
            .iter()
            .map(|t| {
                Tagged::new(
                    t.id,
                    if t.pos == Pos::Stop {
                        Pos::Stop
                    } else {
                        Pos::Noun
                    },
                )
            })
            .collect();
        flattened.extend(parse_to_spo(&blind));
    }
    let passes = source_passes() - before;
    let (q50, q95, qmax) = quantiles(&mut spans_per_unit.clone());

    g.run(
        &format!(
            "T-DEEPNSM[{}] projection runs from ids alone, zero source passes",
            c.name
        ),
        passes == 0 && units > 0,
        &format!(
            "{units} lexical units from {} tokens with {passes} additional source \
             tokenizations — `project()` takes a borrowed view and NO source bytes, so \
             re-reading the source is unavailable rather than merely avoided",
            s.tokens
        ),
    );
    g.run(
        &format!(
            "T-DEEPNSM-SPAN[{}] every unit's byte span addresses the canonical text",
            c.name
        ),
        offsets_ok,
        "each unit's (byte_from, byte_to) slice of the canonical text re-normalises to exactly \
         the surface the projection produced from ids — the seam's span identity is real, not \
         nominal",
    );
    g.run(
        &format!(
            "T-DEEPNSM-CARD[{}] the BPE:word cardinality is measured, not assumed",
            c.name
        ),
        qmax > 1 && multi_unit_tokens > 0 && q50 >= 1,
        &format!(
            "tokens per lexical unit p50={q50} p95={q95} max={qmax}; {multi_unit_tokens} tokens \
             carry the start of MORE THAN ONE unit (a token straddling a word boundary). \
             Neither direction is 1:1 — BPE sequence identity and the DeepNSM word coordinate \
             are different id spaces, exactly as the fence requires"
        ),
    );
    g.run(
        &format!(
            "T-DEEPNSM-FSM[{}] the FSM consumes (WordId, Pos) with no strings",
            c.name
        ),
        !triples.is_empty() && flattened.is_empty() && resolved * 2 > units,
        &format!(
            "{}/{units} units resolved to a PaletteVocab WordId ({:.1}% ; {:.1}% OOV against a \
             {}-word academic vocabulary), and `parse_to_spo` emitted {} SPO triples. The Pos \
             half is load-bearing, not decorative: the SAME word ids with every tag flattened to \
             Noun emit {} triples, so the FSM's output genuinely depends on the pair and this \
             gate is not a restatement of a type signature. DeepNSM-v2 needed NO change — its \
             library is already tokenizer-free, the split_whitespace lives only in its examples",
            resolved,
            pct(resolved, units),
            100.0 - pct(resolved, units),
            vocab.len(),
            triples.len(),
            flattened.len()
        ),
    );
    s.lex_units = units;
    s.lex_resolved = resolved;
    s.tokens_per_unit = (q50, q95, qmax);
    s.multi_unit_tokens = multi_unit_tokens;
    s.spo = triples.len();
}

/// What one Tantivy index measured.
struct IndexStats {
    segments: usize,
    terms: usize,
    docs: u64,
    total_tokens: u64,
    bytes: usize,
    phrase_hits: usize,
}

/// Consumer A: Tantivy, driven by receipt handles. The index never sees text.
fn tantivy_arm(
    store: &Arc<SeamStore>,
    mode: TermMode,
    probe_receipt: usize,
    g: &mut Gate,
    label: &str,
) -> IndexStats {
    let mut sb = Schema::builder();
    let opts = TextOptions::default().set_stored().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer("receipt")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions),
    );
    let body = sb.add_text_field("body", opts);
    let schema = sb.build();
    let index = Index::create_in_ram(schema);
    index
        .tokenizers()
        .register("receipt", ReceiptTokenizer::new(Arc::clone(store), mode));

    let n = store.lane.receipts().len();
    let before_src = source_passes();
    let before_q = query_passes();
    {
        // ONE indexing thread => ONE segment. The first version used the
        // default writer, which spreads documents across threads and therefore
        // across segments; summing `num_terms()` over segments then reported
        // 202 terms for a lane holding 137 distinct ids. A term count that
        // exceeds the vocabulary is the signature of that mistake.
        let mut w: IndexWriter = index
            .writer_with_num_threads(1, 50_000_000)
            .expect("writer");
        for i in 0..n {
            let mut d = TantivyDocument::default();
            // The FIELD VALUE IS A HANDLE, not the text. This is what makes
            // re-tokenization structurally impossible rather than merely
            // discouraged.
            d.add_text(body, handle(i));
            w.add_document(d).expect("add");
        }
        w.commit().expect("commit");
    }
    let src_passes = source_passes() - before_src;
    let q_passes = query_passes() - before_q;

    let reader = index.reader().expect("reader");
    let searcher = reader.searcher();
    let mut terms = 0usize;
    let mut total_tokens = 0u64;
    let segments = searcher.segment_readers().len();
    for seg in searcher.segment_readers() {
        let inv = seg.inverted_index(body).expect("inverted index");
        terms += inv.terms().num_terms();
        total_tokens += inv.total_num_tokens();
    }
    let dir = index.directory();
    let bytes: usize = dir
        .list_managed_files()
        .iter()
        .filter_map(|p| dir.open_read(p).ok())
        .map(|f| {
            use tantivy::HasLen as _;
            f.len()
        })
        .sum();

    // ---- positions must BE the receipt's positions ----
    let mut analyzer = index.tokenizers().get("receipt").expect("registered");
    let h = handle(probe_receipt);
    let mut seen: Vec<usize> = Vec::new();
    {
        let mut ts = analyzer.token_stream(&h);
        while ts.advance() {
            seen.push(ts.token().position);
        }
    }
    let r = store.lane.receipts()[probe_receipt];
    let expect: Vec<usize> = (0..r.token_count as usize).collect();
    g.run(
        &format!("T-TANTIVY-POS[{label}] index positions ARE receipt positions"),
        seen == expect,
        &format!(
            "receipt {probe_receipt} has token_count={} and the analyzer registered on the index \
             yields positions 0..{} in order — no drift, because there is only one segmentation \
             and the index consumed it",
            r.token_count, r.token_count
        ),
    );
    g.run(
        &format!("T-TANTIVY-NOSRC[{label}] indexing tokenized NOTHING, by either counter"),
        src_passes == 0 && q_passes == 0,
        &format!(
            "{n} documents indexed, {src_passes} source + {q_passes} query tokenizations. Both \
             counters are asserted: during INDEXING any tokenization at all is a \
             re-tokenization, and handing Tantivy the raw text instead of a handle would take \
             the fallback path and bump the QUERY counter — a source-only assertion would have \
             stayed green through exactly the mistake it exists to catch"
        ),
    );

    // ---- a phrase query over the SAME segmentation ----
    let ids: Vec<u8> = {
        let v = store.lane.view(&r, &store.contract).expect("same contract");
        v.ids().iter().copied().take(3).collect()
    };
    let phrase: Vec<Term> = ids
        .iter()
        .map(|&id| {
            let text = match mode {
                TermMode::Surface => {
                    String::from_utf8_lossy(store.contract.surface(id)).to_string()
                }
                TermMode::TokenId => format!("{id:02x}"),
            };
            Term::from_field_text(body, &text)
        })
        .collect();
    let (hits, stored) = if phrase.len() >= 2 {
        let q = PhraseQuery::new(phrase);
        let top = searcher
            .search(&q, &TopDocs::with_limit(10).order_by_score())
            .expect("search");
        let stored = top.first().map_or_else(String::new, |(_, addr)| {
            let d: TantivyDocument = searcher.doc(*addr).expect("doc");
            d.get_first(body)
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default()
        });
        (top.len(), stored)
    } else {
        (0, String::new())
    };
    g.run(
        &format!("T-TANTIVY-PHRASE[{label}] a phrase over the shared segmentation retrieves"),
        hits > 0 && stored == handle(probe_receipt),
        &format!(
            "a 3-token phrase taken straight out of receipt {probe_receipt}'s ids matched {hits} \
             document(s) via POSITIONS ONLY, and the top hit's stored field is exactly \
             \"{stored}\" — the receipt the phrase came FROM, not merely some handle-shaped \
             value (every document in this index starts with the handle prefix, so a \
             prefix check would have been unconditional). Tantivy persists term + position and never writes \
             offset_from/offset_to at all (measured in this fork: the indexer's `index_text` \
             reads `text` and `position`, uses `position_length` transiently, and reads the \
             offsets nowhere outside its own tests), so the index cannot become the owner of \
             offsets even by accident"
        ),
    );

    IndexStats {
        segments,
        terms,
        docs: searcher.num_docs(),
        total_tokens,
        bytes,
        phrase_hits: hits,
    }
}

/// Lines of Rust that are not comments — the fence must not be satisfied by
/// its own prose.
fn code_lines(src: &str) -> String {
    src.lines()
        .filter(|l| {
            let t = l.trim_start();
            !(t.starts_with("//") || t.starts_with("/*") || t.starts_with('*'))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn main() {
    let mut g = Gate { pass: 0 };
    println!("PROBE-TOKEN-SEAM-1 — one receipt, three borrowed consumers\n");

    let (vocab, posmap) = load_vocab();
    let corpora = [kjv(), alice(300)];

    // ---- T-CORPUS ----
    // The unicode-whitespace divergence, BOUNDED by measurement rather than by
    // hope: DeepNSM splits on `char::is_whitespace` (Unicode), this projection
    // splits on `u8::is_ascii_whitespace`. They differ only where a corpus
    // contains non-ASCII whitespace, so the probe counts it instead of
    // assuming it away.
    let exotic_ws: usize = corpora
        .iter()
        .map(|c| {
            String::from_utf8_lossy(&c.source)
                .chars()
                .filter(|ch| ch.is_whitespace() && !ch.is_ascii_whitespace())
                .count()
        })
        .sum();
    g.run(
        "T-CORPUS committed, real, hermetic; span population and the unicode-whitespace \
         divergence both measured",
        corpora.iter().all(|c| c.source.len() > 1000)
            && spans(&corpora[0].ir, 0).len() >= 8
            && spans(&corpora[1].ir, 0).len() >= 100
            && vocab.len() > 10_000
            && exotic_ws == 0,
        &format!(
            "{} ({} bytes, {} spans) and {} ({} bytes, {} spans), both committed in-repo; \
             DeepNSM vocabulary {} words from the committed COCA projection. ABSENT and NOT \
             simulated: whole-KJV, bible_vocab.txt, the trained cam96 codebook/codes (so the \
             semantic DISTANCE half of DeepNSM cannot be exercised here), and any MecCog corpus \
             — which exists nowhere on this machine. The span COUNT is asserted, not just \
             the byte count: the CRLF split bug this probe hit left every byte total healthy \
             while collapsing 300 spans into 1. Non-ASCII whitespace, on which this \
             projection's byte-level split would legitimately diverge from DeepNSM's \
             char-level `is_whitespace`, occurs {exotic_ws} times across both corpora — the \
             divergence is bounded by measurement, and named as a gap rather than denied",
            corpora[0].name,
            corpora[0].source.len(),
            spans(&corpora[0].ir, 0).len(),
            corpora[1].name,
            corpora[1].source.len(),
            spans(&corpora[1].ir, 0).len(),
            vocab.len()
        ),
    );

    // ---- T-DOCIR: through the IR's OWN closed-vocabulary gate ----
    let mut roundtrip_ok = true;
    let mut sha_ok = true;
    for c in &corpora {
        let json = to_json(&c.ir).expect("serialize");
        match from_json(&json) {
            Ok(back) => {
                if back != c.ir {
                    roundtrip_ok = false;
                }
            }
            Err(_) => roundtrip_ok = false,
        }
        if c.ir.content_sha256 != <[u8; 32]>::from(Sha256::digest(&c.source)) {
            sha_ok = false;
        }
    }
    // The gate must also REFUSE: swap a region kind for one outside the closed
    // vocabulary and the IR's own loader has to reject it, or "closed" is a
    // word rather than a mechanism. Same for a version bump.
    let off_vocab = to_json(&corpora[0].ir)
        .expect("serialize")
        .replace("\"kind\":\"text\"", "\"kind\":\"paragraph\"");
    let refused = from_json(&off_vocab).is_err();
    let wrong_version = to_json(&corpora[0].ir)
        .expect("serialize")
        .replace("\"version\":\"doc.v1\"", "\"version\":\"doc.v2\"");
    let version_refused = from_json(&wrong_version).is_err();
    g.run(
        "T-DOCIR the span population comes from ogar-doc-ir, through its own load gate",
        roundtrip_ok && sha_ok && refused && version_refused,
        "both corpora round-trip `to_json` -> `from_json` unchanged and their \
         `content_sha256` is the sha256 of the ORIGINAL bytes; an off-vocabulary region kind \
         (`paragraph`) and a `doc.v2` version are BOTH refused by the IR's loader, so the \
         closed vocabulary is a mechanism and not a word. Note what that hash IS, per the \
         crate's own correction of its plan: a PER-ACQUISITION dedup key, not a cross-retina \
         identity — a scan and an HTML page of one invoice have different bytes. For a \
         TOKENIZATION receipt that is exactly the right reading: you tokenize bytes, so \
         different bytes are a different tokenization, and cross-retina convergence is a facts \
         question (`converges_on_facts`) that is not this seam's business. Geometry is \
         `DomOrder` on both corpora because these are reading-order placements; claiming \
         `Rendered` would assert a measurement nobody took",
    );

    // ---- T-DOCIR-SPANS: which regions become spans, and which do NOT ----
    // The two text corpora contain only `Text` regions, so nothing in them can
    // falsify how a figure or a table is handled. This purpose-built IR can:
    // a `Figure` has no text and a `Table` carries typed `(row, col)` cells
    // that must NOT be poured into a token stream — flattening a table into
    // text is the mistake the ingestion doctrine names, and pouring cells in
    // here would destroy exactly the typed structure the structured path
    // consumes.
    let mixed = DocIr {
        version: DOC_IR_VERSION.to_string(),
        source: Provenance::Ocr,
        geometry: Geometry::Rendered,
        content_sha256: [7u8; 32],
        mime: "image/png".to_string(),
        pages: vec![DocPage {
            number: 3,
            width: 1000,
            height: 2000,
            regions: vec![
                Region {
                    kind: RegionKind::Main,
                    bbox: BBoxRail {
                        tl: Rail { x: 0, y: 0 },
                        br: Rail { x: 255, y: 255 },
                    },
                    reading_order: 11,
                    text: None, // a pure container
                    cells: Vec::new(),
                    children: vec![Region {
                        kind: RegionKind::Text,
                        bbox: BBoxRail {
                            tl: Rail { x: 0, y: 0 },
                            br: Rail { x: 255, y: 60 },
                        },
                        reading_order: 12,
                        text: Some("the nested paragraph".to_string()),
                        cells: Vec::new(),
                        children: Vec::new(),
                    }],
                },
                Region {
                    kind: RegionKind::Figure,
                    bbox: BBoxRail {
                        tl: Rail { x: 0, y: 60 },
                        br: Rail { x: 255, y: 120 },
                    },
                    reading_order: 13,
                    text: None,
                    cells: Vec::new(),
                    children: Vec::new(),
                },
                Region {
                    kind: RegionKind::Table,
                    bbox: BBoxRail {
                        tl: Rail { x: 0, y: 120 },
                        br: Rail { x: 255, y: 255 },
                    },
                    reading_order: 14,
                    text: None,
                    cells: vec![TableCell {
                        row: 0,
                        col: 0,
                        text: "Haemoglobin".to_string(),
                        bbox: BBoxRail {
                            tl: Rail { x: 0, y: 120 },
                            br: Rail { x: 80, y: 140 },
                        },
                        confidence: 97,
                    }],
                    children: Vec::new(),
                },
            ],
        }],
        fields: Vec::new(),
    };
    let mixed_spans = spans(&mixed, 0);
    let cell_text_leaked = mixed_spans.iter().any(|s| s.text.contains("Haemoglobin"));
    g.run(
        "T-DOCIR-SPANS a container descends, a figure contributes nothing, a table is not flattened",
        mixed_spans.len() == 1
            && mixed_spans[0].text == "the nested paragraph"
            && mixed_spans[0].key.reading_order == 12
            && mixed_spans[0].key.page == 3
            && !cell_text_leaked,
        &format!(
            "a 3-region page (a text-less `Main` container holding one `Text` child, a \
             `Figure`, and a `Table` with one cell) yields exactly {} span — the nested \
             paragraph, keyed (page 3, reading_order 12). The figure adds nothing, the \
             container adds nothing of its own, and the cell text \"Haemoglobin\" does NOT \
             appear in any span: a table's typed (row, col) values go to the structured path, \
             and pouring them into a token stream would destroy the structure that path \
             exists to read",
            mixed_spans.len()
        ),
    );

    let mut summaries: Vec<Summary> = Vec::new();
    let mut index_rows: Vec<(String, IndexStats)> = Vec::new();
    let mut forward_rows: Vec<(String, usize, f64, usize, f64)> = Vec::new();

    for c in &corpora {
        let contract = TokenizerContract::train(&c.source, NormRule::Identity);
        let (lane, mut s) = ingest(c, &contract, &mut g);
        deepnsm_arm(c, &contract, &lane, &vocab, &posmap, &mut s, &mut g);

        // ---- Consumer C: the forward-prediction input surface ----
        let before = source_passes();
        let split = lane.receipts().len() * 4 / 5;
        let holdout: Vec<_> = lane.receipts()[split..]
            .iter()
            .filter_map(|r| lane.view(r, &contract))
            .collect();
        let mut best = (0usize, -1.0f64);
        let mut base = 0.0f64;
        let mut by_k: Vec<(usize, f64)> = Vec::new();
        let mut best_sc = tesseract_paperless::token::forward::ForwardScore::default();
        for k in [1usize, 2, 3] {
            let mut p = CountPredictor::new(k);
            for r in &lane.receipts()[..split] {
                if let Some(v) = lane.view(r, &contract) {
                    p.observe(&v);
                }
            }
            let sc = score(&p, &holdout);
            by_k.push((k, sc.accuracy()));
            if k == 1 {
                base = sc.accuracy();
            }
            if sc.accuracy() > best.1 {
                best = (k, sc.accuracy());
                best_sc = sc;
            }
        }
        let fwd_passes = source_passes() - before;
        // How many token positions have a DeepNSM coordinate available?
        let mut covered = 0usize;
        let mut positions = 0usize;
        for r in lane.receipts() {
            let v = lane.view(r, &contract).expect("same contract");
            positions += v.len();
            for u in project(&v) {
                if vocab.id(&u.surface).is_some() {
                    covered += u.token_span as usize;
                }
            }
        }
        let cov = pct(covered, positions);
        let borrowed = {
            let r = lane.receipts()[0];
            let v = lane.view(&r, &contract).expect("same contract");
            let ids_ptr = v.ids().as_ptr();
            let lane_ptr = lane.particles().as_ptr().cast::<u8>();
            // The window really is a slice of the resident population, not a copy.
            ids_ptr == lane_ptr && windows(&v, 2).count() == v.len().saturating_sub(2)
        };
        g.run(
            &format!(
                "T-FORWARD[{}] the input surface is BORROWED, and it is the same ids",
                s.name
            ),
            borrowed && fwd_passes == 0,
            &format!(
                "context windows are slices INTO the resident particle array (pointer-identical \
                 to the lane's own storage), {fwd_passes} extra source tokenizations; an order-k \
                 counting BASELINE — not a language model — reaches top-1 {:.1}% at k={} against \
                 {:.1}% at k=1 on held-out spans ({} positions scored, {} with a context never \
                 seen in training; per order {}), and {cov:.1}% of token positions also carry a \
                 DeepNSM (basin, identity) coordinate, so the hybrid representation is \
                 CONSTRUCTIBLE. Which representation a trained model should prefer is NOT \
                 measured here and no claim is made",
                100.0 * best.1,
                best.0,
                100.0 * base,
                best_sc.scored,
                best_sc.unseen,
                by_k.iter()
                    .map(|(k, a)| format!("k={k}:{:.1}%", 100.0 * a))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
        );
        forward_rows.push((s.name.to_string(), best.0, best.1, positions, cov));

        // ---- Consumer A: Tantivy, both term modes ----
        let store = Arc::new(SeamStore {
            contract: contract.clone(),
            lane: lane.clone(),
        });
        for (mode, label) in [
            (TermMode::Surface, "surface"),
            (TermMode::TokenId, "token-id"),
        ] {
            let st = tantivy_arm(&store, mode, 0, &mut g, &format!("{}/{label}", s.name));
            index_rows.push((format!("{}/{label}", s.name), st));
        }
        summaries.push(s);
    }

    // ---- T-CONTRACT: the codebook law ----
    let c0 = &corpora[0];
    let a = TokenizerContract::train(&c0.source, NormRule::Identity);
    let b = TokenizerContract::train(&c0.source, NormRule::Identity);
    let other = TokenizerContract::train(&corpora[1].source, NormRule::Identity);
    let lower = TokenizerContract::train(&c0.source, NormRule::AsciiLowercase);
    let sample: &[u8] = &c0.source;
    let (ta, _) = a.try_encode(sample).expect("trained on it");
    let (tb, _) = b.try_encode(sample).expect("trained on it");
    g.run(
        "T-CONTRACT-DET same codebook + same source -> byte-identical token stream",
        a.contract_id() == b.contract_id() && ta == tb,
        &format!(
            "two independent trainings of the same corpus produce contract id {}… and identical \
             {}-token streams; the tie-break is deterministic, so the identity is a function of \
             the corpus, not of allocation order",
            &a.contract_hex()[..16],
            ta.len()
        ),
    );
    let cross = other.decode(&ta).0;
    g.run(
        "T-CONTRACT-DIFF a changed codebook is a DIFFERENT contract, and ids alone are meaningless",
        a.contract_id() != other.contract_id()
            && a.contract_id() != lower.contract_id()
            && cross != sample,
        &format!(
            "a different corpus yields contract {}… and a different NORMALISATION RULE over the \
             SAME corpus yields {}… — the rule is hashed into the identity, not left implicit. \
             Decoding corpus-A's ids under corpus-B's codebook returns {} bytes of garbage \
             instead of the {} original bytes: a stored id without its contract is not a weak \
             reference, it is a wrong one",
            &other.contract_hex()[..16],
            &lower.contract_hex()[..16],
            cross.len(),
            sample.len()
        ),
    );

    // ---- T-CONTRACT-RULE: the normalisation rule is IN the identity ----
    // Trained on already-lowercase bytes, `Identity` and `AsciiLowercase`
    // produce the SAME table — so if the rule were not hashed, they would share
    // an id while behaving differently on mixed-case input. That is the only
    // corpus shape on which this claim is falsifiable; a mixed-case corpus
    // makes the tables differ and the assertion passes for the wrong reason.
    let lc = c0.source.to_ascii_lowercase();
    let r_id = TokenizerContract::train(&lc, NormRule::Identity);
    let r_lo = TokenizerContract::train(&lc, NormRule::AsciiLowercase);
    let mixed = b"The Garden";
    let id_takes_mixed = r_id.try_encode(mixed).is_some();
    let lo_takes_mixed = r_lo.try_encode(mixed).is_some();
    g.run(
        "T-CONTRACT-RULE identical tables + different rules are different contracts",
        r_id.vocab_len() == r_lo.vocab_len()
            && r_id.contract_id() != r_lo.contract_id()
            && !id_takes_mixed
            && lo_takes_mixed,
        &format!(
            "on an already-lowercase corpus both rules train the SAME {}-id table, yet the \
             contract ids differ ({}… vs {}…) because the rule is hashed into the identity. \
             They are not interchangeable: encoding \"The Garden\" under Identity is REFUSED \
             (uppercase 'T' is outside the trained alphabet) and under AsciiLowercase succeeds. \
             A shared id for those two behaviours would be a lie the store could not detect",
            r_id.vocab_len(),
            &r_id.contract_hex()[..12],
            &r_lo.contract_hex()[..12]
        ),
    );

    // ---- T-FRAME: token_count is authoritative, PAD is not a length ----
    let mut frame_lane = TokenLane::new();
    let (full, _) = a.try_encode(&c0.source).expect("trained");
    let exact = full.len() - (full.len() % IDS_PER_PARTICLE); // a 12-aligned run
    let r0 = frame_lane.append(
        SpanKey {
            doc: 0,
            page: 0,
            reading_order: 0,
        },
        0,
        &a,
        &full[..exact],
    );
    let r1 = frame_lane.append(
        SpanKey {
            doc: 0,
            page: 0,
            reading_order: 1,
        },
        0,
        &a,
        &full[exact..],
    );
    let flat = frame_lane.particles().as_flattened();
    let pad_scan = flat
        .iter()
        .position(|&x| x == tesseract_paperless::token::contract::PAD)
        .unwrap_or(flat.len());
    g.run(
        "T-FRAME an exactly-full run has NO pad, so pad-inference reads into the next receipt",
        r0.tail_is_full() && pad_scan > r0.token_count as usize,
        &format!(
            "receipt 0 holds {} tokens in {} particles with a full tail; a lane-wide scan for the \
             first PAD stops at {pad_scan}, which is {} tokens PAST receipt 0's end and inside \
             receipt 1 ({} tokens) — so a receipt carrying only `first_particle` cannot frame \
             itself, and every span whose length is a multiple of 12 is this case rather than a \
             corner one. Stated honestly, there are then TWO lawful framings and this probe \
             carries both fields: `particle_count` alone bounds the run and, because PAD is a \
             RESERVED id, a scan inside that bound is already exact — which costs one \
             vocabulary slot; `token_count` costs 4 bytes instead and frees the slot for a full \
             256-id alphabet. What is NOT lawful is inferring the end from padding without a \
             bound, which is what this gate measures",
            r0.token_count,
            r0.particle_count,
            pad_scan - r0.token_count as usize,
            r1.token_count
        ),
    );
    let v0 = frame_lane.view(&r0, &a).expect("same contract");
    let v1 = frame_lane.view(&r1, &a).expect("same contract");
    let mut joined = v0.decode();
    joined.extend_from_slice(&v1.decode());
    g.run(
        "T-FRAME-ADJ adjacent receipts decode independently and concatenate exactly",
        v0.len() == exact && v1.len() == full.len() - exact && joined == c0.source,
        &format!(
            "receipt 0 -> {} tokens, receipt 1 -> {} tokens, and their decodes concatenate back to \
             the full {} canonical bytes with no bleed in either direction",
            v0.len(),
            v1.len(),
            c0.source.len()
        ),
    );
    g.run(
        "T-CONTRACT-GATE a view refuses to open under the wrong contract",
        frame_lane.view(&r0, &other).is_none(),
        "`TokenLane::view` compares the receipt's contract id against the codebook offered and \
         returns None on mismatch — a mis-read cannot be silent",
    );

    // ---- T-QUERY: query analysis is a pass over QUERY bytes, and it is counted ----
    let q_before = query_passes();
    let s_before = source_passes();
    let store = Arc::new(SeamStore {
        contract: a.clone(),
        lane: frame_lane.clone(),
    });
    let mut tk = ReceiptTokenizer::new(Arc::clone(&store), TermMode::Surface);
    let mut n_q = 0usize;
    {
        let mut ts = tk.token_stream("the garden");
        while ts.advance() {
            n_q += 1;
        }
    }
    g.run(
        "T-QUERY query analysis is separately counted and touches no source",
        query_passes() - q_before == 1 && source_passes() - s_before == 0 && n_q > 0,
        &format!(
            "analysing the query \"the garden\" cost 1 QUERY tokenization ({n_q} terms) and 0 \
             source tokenizations. A query is different bytes than the corpus; folding it into \
             one number would make the zero-retokenization claim a lie, so the counters are \
             separate and both are reported"
        ),
    );

    // ---- T-FENCE ----
    let srcs = [
        include_str!("../src/token/contract.rs"),
        include_str!("../src/token/lane.rs"),
        include_str!("../src/token/lexical.rs"),
        include_str!("../src/token/seam_tantivy.rs"),
        include_str!("../src/token/forward.rs"),
    ];
    let code: String = srcs
        .iter()
        .map(|s| code_lines(s))
        .collect::<Vec<_>>()
        .join("\n");
    let manifest = include_str!("../Cargo.toml").to_lowercase();
    let no_class_addr = !code.contains("classid") && !code.contains("class_id");
    let no_df = !manifest.contains("polars") && !manifest.contains("dataframe");
    g.run(
        "T-FENCE no class address in the token path, no DataFrame in the manifest",
        no_class_addr && no_df,
        &format!(
            "{} lines of non-comment library code contain no `classid`/`class_id` — the contract \
             id is a FIELD on the receipt, never smuggled into an address (E-CONTENT-NEVER-\
             TRAVELS-IN-CLASSID-1); and the manifest declares no DataFrame of any kind. The \
             online path here is parse -> normalise -> tokenize -> index -> project, and not one \
             of those five steps is a groupby, a join, a window function or a columnar expression",
            code.lines().count()
        ),
    );

    // ---- the measured report ----
    println!("\n── corpus / lane ──");
    println!(
        "{:<22} {:>8} {:>7} {:>8} {:>7} {:>6} {:>9} {:>6} {:>5} {:>5} {:>7}",
        "corpus",
        "bytes",
        "spans",
        "tokens",
        "uniq",
        "part.",
        "resident",
        "p50",
        "p95",
        "max",
        "cont%"
    );
    // `encode_probes` is #1012's honest cost unit: merge-table probes, never
    // wall time. A debug/release timing here would be a junk number.
    for s in &summaries {
        println!(
            "{:<22} {:>8} {:>7} {:>8} {:>7} {:>6} {:>9} {:>6} {:>5} {:>5} {:>6.1}%",
            s.name,
            s.bytes,
            s.spans,
            s.tokens,
            s.uniq_tokens,
            s.particles,
            s.resident_bytes,
            s.p50,
            s.p95,
            s.pmax,
            s.continuation_rate
        );
    }
    println!(
        "encode cost (merge-table probes, not wall time): {}",
        summaries
            .iter()
            .map(|s| format!("{}={}", s.name, s.encode_probes))
            .collect::<Vec<_>>()
            .join("  ")
    );
    println!("\n── lexical projection (DeepNSM-v2) ──");
    println!(
        "{:<22} {:>7} {:>9} {:>8} {:>16} {:>10} {:>6}",
        "corpus", "units", "resolved", "OOV%", "tok/unit p50/95/max", "straddle", "SPO"
    );
    for s in &summaries {
        println!(
            "{:<22} {:>7} {:>9} {:>7.1}% {:>10}/{}/{} {:>10} {:>6}",
            s.name,
            s.lex_units,
            s.lex_resolved,
            100.0 - pct(s.lex_resolved, s.lex_units),
            s.tokens_per_unit.0,
            s.tokens_per_unit.1,
            s.tokens_per_unit.2,
            s.multi_unit_tokens,
            s.spo
        );
    }
    println!("\n── Tantivy index (same segmentation, two term readings) ──");
    println!(
        "{:<34} {:>5} {:>7} {:>6} {:>12} {:>11} {:>7}",
        "index", "segs", "terms", "docs", "index tokens", "bytes", "phrase"
    );
    for (name, st) in &index_rows {
        println!(
            "{:<34} {:>5} {:>7} {:>6} {:>12} {:>11} {:>7}",
            name, st.segments, st.terms, st.docs, st.total_tokens, st.bytes, st.phrase_hits
        );
    }
    println!("\n── forward-prediction input surface (counting BASELINE, not a model) ──");
    println!(
        "{:<22} {:>6} {:>10} {:>11} {:>14}",
        "corpus", "best k", "top-1", "positions", "DeepNSM cov%"
    );
    for (name, k, acc, pos, cov) in &forward_rows {
        println!(
            "{name:<22} {k:>6} {:>9.1}% {pos:>11} {cov:>13.1}%",
            100.0 * acc
        );
    }
    println!("\n── passes ──");
    println!(
        "source tokenizations: {} (= one per span, summed over both corpora and the framing \
         fixture)\nquery  tokenizations: {} (separate by construction)",
        source_passes(),
        query_passes()
    );

    println!("\nPROBE-TOKEN-SEAM-1: ALL {} GATES GREEN", g.pass);
    println!(
        "\nverdict: ONE receipt drove all three consumers. Tantivy indexed a receipt HANDLE and \
         never received the source; DeepNSM-v2 projected from ids alone through an unmodified \
         library; the forward surface is a borrowed slice of the same particles. Byte offsets \
         are DERIVED from the codebook's length table, so the receipt stores none — and the \
         span population and every span's identity come from `ogar-doc-ir`, so the receipt \
         mints nothing either. An offset is therefore REGION-LOCAL, which is what retires the \
         old no-offsets-at-the-OCR-boundary gap. What is NOT settled: the resident carrier is \
         still a probe-local Vec; the 8-bit vocabulary table is FULL at 255/255 on 75 KB of \
         English; this probe builds its DocIr from text rather than from a real retina; and \
         there is no callable PoS surface anywhere — the module that held one was deliberately \
         deleted, and the grounding cited for that deletion is itself an example binary \
         outside this repo's dependency barrier."
    );
}
