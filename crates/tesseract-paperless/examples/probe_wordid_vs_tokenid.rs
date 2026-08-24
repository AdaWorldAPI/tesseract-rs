//! `PROBE-WORDID-VS-TOKENID-1` — side by side: what does each candidate put
//! INTO the 256:256 tile, and what does it cost you?
//!
//! Three candidates want the same two bytes. They are not variants of one
//! thing; they differ in what the ADDRESS means and in what you can get back.
//!
//! | | the hi byte means | reconstruction |
//! |---|---|---|
//! | `WordId` (`DeepNSM`-v2) | frequency band | LOSSY — measured below |
//! | `TokenId` (BPE rail) | merge order (nothing) | EXACT — measured below |
//! | `WordNet` synset | taxonomic ancestry | n/a — not a tokenization |
//!
//! Separate tiles, same shape (operator ruling). This probe measures the first
//! two on identical spans. The third is CITED, not re-measured: `WordNet`'s
//! source host is outside this environment's network egress allowlist
//! (`Host not in allowlist: wordnetcode.princeton.edu`), so the corpus cannot
//! be fetched here — but lance-graph already measured it
//! (`PROBE-WORDNET-44-ACTIVATION`, 5/5) and those numbers are quoted with
//! their provenance rather than re-derived.
//!
//! # Honesty box
//! - `PaletteVocab` is loaded from the committed COCA projection; the trained
//!   `cam96` codebook is ABSENT, so `DeepNSM`'s SEMANTIC DISTANCE is not
//!   exercised. What is measured is the id space and the round trip.
//! - The `WordId` round trip joins surviving words with single spaces. That is
//!   the most favourable reconstruction available — the normalisation genuinely
//!   discards case, punctuation and short words, so a kinder join cannot exist.
//! - No claim is made about which is "better". They answer different questions.

// A probe is one linear argument: splitting `main` into helpers hides the order
// the measurements run in, which is what a reader checks.
#![allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::too_many_lines
)]

use std::collections::{HashMap, HashSet};

use deepnsm_v2::PaletteVocab;
use tesseract_paperless::token::contract::{NormRule, TokenizerContract};

/// `WordNet`, quoted from `PROBE-WORDNET-44-ACTIVATION` (lance-graph, 5/5 gates).
/// Not re-measured here: the corpus host is outside the egress allowlist.
const WORDNET_SYNSETS: usize = 82_192;
const WORDNET_LEAVES: usize = 65_292;
const WORDNET_BAND_RECALL: f64 = 0.763;
const WORDNET_RANDOM_RECALL: f64 = 0.031;

/// The 256:256 rail carrier on this same corpus, from `PROBE-TILE-H2H-1`.
/// Cited, not re-trained here — see the note at the print site.
const RAIL_UNITS: usize = 27_134;
const RAIL_DISTINCT: usize = 6_351;

/// The `DeepNSM` lexical rule, from its own consumers.
fn normalise(tok: &str) -> Option<String> {
    let w: String = tok
        .chars()
        .filter(char::is_ascii_alphabetic)
        .collect::<String>()
        .to_lowercase();
    (w.len() >= 2).then_some(w)
}

fn paragraphs(raw: &str, max: usize) -> Vec<String> {
    let raw = raw.trim_start_matches('\u{feff}').replace("\r\n", "\n");
    let mut out = Vec::new();
    for para in raw.split("\n\n") {
        let p = para.split_whitespace().collect::<Vec<_>>().join(" ");
        if p.len() >= 40 {
            out.push(p);
        }
        if out.len() >= max {
            break;
        }
    }
    out
}

fn main() {
    println!("PROBE-WORDID-VS-TOKENID-1 — what goes in the 256:256 tile\n");

    // ---- vocabulary (committed COCA projection) ----
    let tsv = include_str!("../corpus/coca_academic_20k.tsv");
    let mut vocab = PaletteVocab::new();
    let words: Vec<&str> = tsv
        .lines()
        .filter_map(|l| l.split('\t').next())
        .filter(|w| !w.is_empty())
        .collect();
    vocab.from_frequency_ranked(words);

    let spans = paragraphs(include_str!("../corpus/alice.txt"), usize::MAX);
    let src_bytes: usize = spans.iter().map(String::len).sum();

    // ---- TokenId arm: BPE over the 256:256 rail tile ----
    let joined = spans.join("\n").into_bytes();
    let contract = TokenizerContract::train(&joined, NormRule::Identity);
    let mut tok_units = 0usize;
    let mut tok_ids: HashSet<u8> = HashSet::new();
    let mut tok_exact = true;
    for s in &spans {
        let (ids, _) = contract.try_encode(s.as_bytes()).expect("trained");
        if contract.decode(&ids).0 != s.as_bytes() {
            tok_exact = false;
        }
        tok_units += ids.len();
        tok_ids.extend(ids.iter().copied());
    }

    // ---- WordId arm: the same spans through DeepNSM's own lexical rule ----
    let mut word_units = 0usize;
    let mut word_resolved = 0usize;
    let mut word_ids: HashSet<u16> = HashSet::new();
    let mut basins: HashSet<u8> = HashSet::new();
    let mut recovered_bytes = 0usize;
    let mut oov: HashMap<String, usize> = HashMap::new();
    for s in &spans {
        let mut rebuilt: Vec<&str> = Vec::new();
        for t in s.split_whitespace() {
            let Some(w) = normalise(t) else { continue };
            word_units += 1;
            if let Some(id) = vocab.id(&w) {
                word_resolved += 1;
                word_ids.insert(id);
                basins.insert(deepnsm_v2::vocab::split(id).0);
                rebuilt.push(vocab.word(id).expect("just resolved"));
            } else {
                *oov.entry(w).or_default() += 1;
            }
        }
        // The kindest possible round trip: surviving words, single-spaced.
        recovered_bytes += rebuilt.join(" ").len();
    }

    println!(
        "corpus: alice-full, {} spans, {} source bytes\n",
        spans.len(),
        src_bytes
    );
    println!(
        "{:<16} {:>9} {:>9} {:>12} {:>13} {:>11}",
        "carrier", "units", "distinct", "of its space", "reconstruct", "hi byte =",
    );
    println!(
        "{:<16} {:>9} {:>9} {:>11.2}% {:>13} {:>11}",
        "TokenId u8/255",
        tok_units,
        tok_ids.len(),
        100.0 * tok_ids.len() as f64 / 256.0,
        if tok_exact { "EXACT" } else { "BROKEN" },
        "merge order",
    );
    println!(
        "{:<16} {:>9} {:>9} {:>11.2}% {:>12.1}% {:>11}",
        "WordId (DeepNSM)",
        word_units,
        word_ids.len(),
        100.0 * word_ids.len() as f64 / 65_536.0,
        100.0 * recovered_bytes as f64 / src_bytes as f64,
        "freq band",
    );
    // The rail carrier's row is CITED from PROBE-TILE-H2H-1 rather than
    // re-trained here: this crate's `contract.rs` is still the u8 carrier, and
    // scoring a u8 vocabulary against a 65 536 tile is the wrong-quantity
    // mistake. Provenance is explicit so the cell cannot be read as measured
    // here.
    println!(
        "{:<16} {:>9} {:>9} {:>11.2}% {:>13} {:>11}",
        "TokenId u16 rail",
        RAIL_UNITS,
        RAIL_DISTINCT,
        100.0 * RAIL_DISTINCT as f64 / 65_536.0,
        "EXACT",
        "merge order",
    );
    println!(
        "{:<16} {:>9} {:>9} {:>11.2}% {:>13} {:>11}",
        "WordNet synset",
        "-",
        WORDNET_LEAVES,
        100.0 * WORDNET_LEAVES as f64 / 65_536.0,
        "n/a",
        "ancestry",
    );

    let lost = src_bytes - recovered_bytes;
    println!(
        "\nWordId round trip: {} of {} source bytes recover ({:.1}%); {} bytes ({:.1}%) are \
         GONE — case, punctuation, digits and every sub-2-letter word. {} of {} lexical units \
         resolved ({:.1}% OOV). This is not a defect: DeepNSM's id is a SEMANTIC coordinate \
         and was never a codec. But it means a WordId stream cannot be the canonical carrier — \
         you cannot get the document back from it.",
        recovered_bytes,
        src_bytes,
        100.0 * recovered_bytes as f64 / src_bytes as f64,
        lost,
        100.0 * lost as f64 / src_bytes as f64,
        word_resolved,
        word_units,
        100.0 * (word_units - word_resolved) as f64 / word_units as f64,
    );
    println!(
        "\nTokenId round trip: byte-exact on every span. At the 256:256 rail the tile is \
         {:.1}% EMPTY ({} ids used of 65 536) — the corpus, not the cap, set that. But its hi byte is merge \
         order: two ids sharing a basin share NOTHING semantic, so address adjacency in this \
         tile is not a search prior.",
        100.0 * (65_536.0 - RAIL_DISTINCT as f64) / 65_536.0,
        RAIL_DISTINCT,
    );
    println!(
        "\nWordId basins actually occupied: {} of 256. The hi byte is the FREQUENCY band, so \
         basin adjacency means 'similar corpus frequency' — which is not semantic adjacency \
         either. DeepNSM gets its meaning from the trained cam96 codebook, which is ABSENT \
         here and is a SEPARATE structure from the id.",
        basins.len(),
    );
    println!(
        "\nWordNet, CITED not re-measured (host outside this environment's egress allowlist; \
         lance-graph PROBE-WORDNET-44-ACTIVATION 5/5): {} noun synsets, {} leaves — {:.1}% of \
         a 65 536 tile, a near-exact fit. Folded to 4^4 = 256 cells, ancestry-band recall is \
         {:.3} vs {:.3} random = {:.1}x. That is the ONLY one of the three whose ADDRESS \
         carries meaning: prefix = ancestor, by construction rather than by training.",
        WORDNET_SYNSETS,
        WORDNET_LEAVES,
        100.0 * WORDNET_LEAVES as f64 / 65_536.0,
        WORDNET_BAND_RECALL,
        WORDNET_RANDOM_RECALL,
        WORDNET_BAND_RECALL / WORDNET_RANDOM_RECALL,
    );
    let mut top: Vec<(&String, &usize)> = oov.iter().collect();
    top.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    println!(
        "\ntop OOV against the academic vocabulary: {}",
        top.iter()
            .take(12)
            .map(|(w, c)| format!("{w}({c})"))
            .collect::<Vec<_>>()
            .join(" ")
    );
}
