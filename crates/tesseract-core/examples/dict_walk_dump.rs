//! Byte-parity dump for the Dict-lite walker
//! ([`tesseract_core::DictLite`]) — walks a word (space-separated
//! `UNICHAR_ID`s) through `default_dawgs` + `def_letter_is_okay`, printing the
//! SAME format as the C++ oracle `/tmp/def_letter_oracle.cpp`.
//!
//! ```sh
//! cargo run -q -p tesseract-core --example dict_walk_dump -- 91 97 92 > /tmp/rust_dw.tsv
//! /tmp/def_letter_oracle 91 97 92 > /tmp/oracle_dw.tsv
//! diff /tmp/oracle_dw.tsv /tmp/rust_dw.tsv   # byte-identical => D1.2b green
//! ```
#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use std::path::Path;

use tesseract_core::dawg::PermuterType;
use tesseract_core::{DawgPosition, DictLite, UniCharSet};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let word_ids: Vec<u32> = args[1..]
        .iter()
        .map(|s| s.parse().expect("unichar id"))
        .collect();
    if word_ids.is_empty() {
        eprintln!(
            "usage: {} <unichar_id_0> [<unichar_id_1> ...]",
            args.first().map_or("dict_walk_dump", String::as_str)
        );
        std::process::exit(2);
    }

    let word = std::fs::read("/tmp/eng.lstm-word-dawg").expect("read word dawg");
    let punc = std::fs::read("/tmp/eng.lstm-punc-dawg").expect("read punc dawg");
    let number = std::fs::read("/tmp/eng.lstm-number-dawg").expect("read number dawg");
    let dict = DictLite::from_components(&word, &punc, &number).expect("load dawgs");
    let charset =
        UniCharSet::load_from_file(Path::new("/tmp/eng.lstm-unicharset")).expect("load unicharset");

    let mut active = dict.default_dawgs(false);
    let n = word_ids.len();
    for (i, &unichar_id) in word_ids.iter().enumerate() {
        let word_end = i + 1 == n;
        let (updated, perm, valid_end) = dict.def_letter_is_okay(
            &active,
            &charset,
            unichar_id,
            word_end,
            PermuterType::NoPerm,
        );
        println!(
            "step\t{i}\t{unichar_id}\t{}\tperm={}\tvalid_end={}\tupdated={{{}}}",
            i32::from(word_end),
            perm.as_i32(),
            i32::from(valid_end),
            updated.len()
        );
        dump_positions(&updated);
        active = updated;
    }
}

/// Sorts lexicographically by `(dawg_index, dawg_ref, punc_index, punc_ref,
/// back_to_punc)`, matching the oracle's `DumpPositions` — makes the dump
/// order-independent of `Vec::push` insertion order.
fn dump_positions(positions: &[DawgPosition]) {
    let mut sorted: Vec<&DawgPosition> = positions.iter().collect();
    sorted.sort_by_key(|p| {
        (
            p.dawg_index,
            p.dawg_ref,
            p.punc_index,
            p.punc_ref,
            i32::from(p.back_to_punc),
        )
    });
    for p in sorted {
        println!(
            "p\t{}\t{}\t{}\t{}\t{}",
            p.dawg_index,
            p.dawg_ref,
            p.punc_index,
            p.punc_ref,
            i32::from(p.back_to_punc)
        );
    }
}
