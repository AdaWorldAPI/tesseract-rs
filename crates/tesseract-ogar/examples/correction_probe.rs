//! Does dictionary correction actually help? Measure it, per token, against a
//! REAL lexicon — and print what it declines as prominently as what it fixes.
//!
//! Diagnostic only. Two things this exists to answer honestly:
//!
//! 1. **Coverage.** How big is the loaded lexicon really, and how much of the
//!    document's vocabulary does it contain? An English COCA list against a
//!    German lab report answers "almost none", and that is the finding — not a
//!    reason to lower the guards until something changes.
//! 2. **Effect.** Which tokens change, which are declined, and by which guard.
//!    A corrector that silently rewrites is untrustworthy; the point is to see
//!    the edits.
//!
//! ```sh
//! cargo run -p tesseract-ogar --release --example correction_probe -- \
//!     "word1 word2 ..."            # ad-hoc tokens
//! ```
#![allow(clippy::print_stdout, reason = "diagnostic CLI")]

use std::path::PathBuf;

use tesseract_ogar::correction::{suggest, CorrectionPolicy, Lexicon};

fn vocab_dir() -> PathBuf {
    std::env::var("DEEPNSM_VOCAB_DIR").map_or_else(
        |_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../../lance-graph/crates/deepnsm/word_frequency")
        },
        PathBuf::from,
    )
}

fn main() {
    let dir = vocab_dir();
    let lex = match Lexicon::from_deepnsm_vocab_dir(&dir) {
        Ok(l) => l,
        Err(e) => {
            println!(
                "could not load the deepnsm vocabulary at {}: {e}",
                dir.display()
            );
            return;
        }
    };
    println!(
        "lexicon: {} distinct words (from {})",
        lex.len(),
        dir.display()
    );

    // Sweep the long-token budget: every English fix measured at distance 1,
    // while the one German corruption is distance 2 — so the budget is the
    // knob that separates "useful" from "harmful" here.
    for long_budget in [1usize, 2] {
        let policy = CorrectionPolicy {
            max_distance_long: long_budget,
            ..CorrectionPolicy::default()
        };
        println!("### max_distance_long = {long_budget} ###");
        let mut harm = 0;
        for (label, tokens) in [
            (
                "german",
                &[
                    "Parameter",
                    "Ergebnis",
                    "Einheit",
                    "Referenz",
                    "Haemoglobin",
                    "Glukose",
                    "Kalium",
                    "Natrium",
                    "Kreatinin",
                    "Calcium",
                ][..],
            ),
            (
                "english",
                &[
                    "beginnlng",
                    "thlnking",
                    "conversatlon",
                    "slster",
                    "pictures",
                    "rabblt",
                    "wondet",
                    "very",
                    "tired",
                ][..],
            ),
        ] {
            let mut fixed = Vec::new();
            for t in tokens {
                if let Some((to, d)) = suggest(t, &lex, &policy) {
                    fixed.push(format!("{t}->{to}(d{d})"));
                    if label == "german" {
                        harm += 1;
                    }
                }
            }
            println!("  {label}: {} changed  {:?}", fixed.len(), fixed);
        }
        println!("  german corruptions: {harm}\n");
    }

    let policy = CorrectionPolicy::default();
    println!(
        "policy: min_len={} budget(short)={} budget(>= {})={}\n",
        policy.min_len, policy.max_distance_short, policy.long_len, policy.max_distance_long
    );

    // The REAL degraded cell text measured on corpus/lab/lab_table_ruled.pgm
    // (stripped path), plus the header row. This is the actual material a
    // corrector would face on that fixture.
    let lab_tokens = [
        "Parameter",
        "Ergebnis",
        "Einheit",
        "Referenz",
        "Haemoglobin",
        "Glukose",
        "Kalium",
        "Natrium",
        "Kreatinin",
        "Calcium",
        "mg/dl",
        "g/dl",
        "mmol",
        "$142",
        "O09",
        "4mm",
        "ol",
        "885-5",
        "AAC",
        "lsd",
        "=—_—sO&Referenz",
    ];
    // A prose sample in the lexicon's OWN language, so the English-vs-German
    // point is measured rather than asserted.
    let prose_tokens = [
        "beginnlng",
        "thlnking",
        "conversatlon",
        "slster",
        "pictures",
        "rabblt",
        "wondet",
        "very",
        "tired",
    ];

    for (label, tokens) in [
        (
            "GERMAN LAB FIXTURE (real measured cell text)",
            &lab_tokens[..],
        ),
        (
            "ENGLISH PROSE (the lexicon's own language)",
            &prose_tokens[..],
        ),
    ] {
        println!("=== {label} ===");
        let (mut fixed, mut declined) = (0, 0);
        for t in tokens {
            match suggest(t, &lex, &policy) {
                Some((to, d)) => {
                    fixed += 1;
                    println!("  FIX   {t:>18?} -> {to:?}  (distance {d})");
                }
                None => {
                    declined += 1;
                    // Name WHICH guard declined it: that is the difference
                    // between "the lexicon lacks this word" and "the token is
                    // data we must never touch".
                    let why = if t.chars().any(|c| c.is_numeric()) {
                        "guard 1: contains a digit (never correctable)"
                    } else if lex.contains(t.trim_matches(|c: char| !c.is_alphabetic())) {
                        "guard 2: already in the lexicon"
                    } else {
                        let core_len = t.chars().filter(|c| c.is_alphabetic()).count();
                        if core_len < policy.min_len {
                            "guard 3: core below the length floor"
                        } else {
                            "guard 4: nothing within the distance budget"
                        }
                    };
                    println!("  keep  {t:>18?}      {why}");
                }
            }
        }
        println!("  -> {fixed} corrected, {declined} left alone\n");
    }
}
