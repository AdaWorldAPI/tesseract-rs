//! Dump the A2-A4 layer-forward byte-parity surface: `convolve_forward`
//! (half 1,1 — eng's `C3,3`), `maxpool_forward` (3,3 — eng's `Mp3,3`) and
//! `reconfig_forward` (2,2 — the `Ft` shape) over the shared ragged 3-image
//! batch, int8 then float. The libtesseract oracle (`/tmp/conv2d_oracle.cpp`)
//! constructs the REAL `Convolve`/`Maxpool`/`Reconfig` (public ctors), calls
//! the REAL `Forward` with a seeded `TRand`, and prints the same lines.
//!
//! ```sh
//! cargo run -p tesseract-recognizer --example conv2d_dump > /tmp/rust_conv2d.tsv
//! /tmp/conv2d_oracle > /tmp/oracle_conv2d.tsv
//! diff /tmp/oracle_conv2d.tsv /tmp/rust_conv2d.tsv   # byte-identical => A2-A4 green
//! ```
#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use tesseract_recognizer::{
    convolve_forward, maxpool_forward, reconfig_forward, FlexDim, NetworkIo, StrideMap, TRand,
};

const NF: usize = 2;

fn build_input(int_mode: bool) -> NetworkIo {
    let mut map = StrideMap::default();
    map.set_stride(&[(5, 7), (3, 4), (4, 6)]);
    let mut io = NetworkIo::default();
    io.resize_to_map(int_mode, &map, NF);
    let walk = map.clone();
    let mut idx = walk.index_first();
    loop {
        let t = idx.t() as usize;
        let vals: Vec<f32> = (0..NF as i32)
            .map(|f| ((t as i32 * 23 + f * 41) % 180 - 90) as f32 / 90.0)
            .collect();
        io.write_time_step(t, &vals);
        if !idx.increment() {
            break;
        }
    }
    io
}

fn dump(tag: &str, io: &NetworkIo) {
    println!(
        "{tag}shape\t{}\t{}\t{}\t{}\t{}",
        io.stride_map().size(FlexDim::Batch),
        io.stride_map().size(FlexDim::Height),
        io.stride_map().size(FlexDim::Width),
        io.width(),
        io.num_features()
    );
    for t in 0..io.width() {
        print!("{tag}\t{t}");
        if io.int_mode() {
            for &v in io.i(t) {
                print!("\t{v}");
            }
        } else {
            for &v in io.f(t) {
                print!("\t{:08x}", v.to_bits());
            }
        }
        println!();
    }
}

fn run(int_mode: bool) {
    println!("mode\t{}", i32::from(int_mode));
    let input = build_input(int_mode);
    dump("in", &input);

    let mut rng = TRand::default();
    rng.set_seed(4242);
    let conv = convolve_forward(&input, 1, 1, &mut rng);
    dump("c", &conv);

    let mp = maxpool_forward(&input, 3, 3);
    dump("p", &mp);

    let rc = reconfig_forward(&input, 2, 2);
    dump("g", &rc);
}

fn main() {
    run(true);
    run(false);
}
