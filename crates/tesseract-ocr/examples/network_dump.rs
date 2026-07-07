//! B1 byte-parity surface: load the REAL `eng.lstm` network tree, run a full
//! forward over a synthetic image grid, dump the output logits. The oracle
//! (`/tmp/network_forward_oracle.cpp`) loads the same file via libtesseract's
//! public `Network::CreateFromFile`, runs the REAL `net->Forward` on the SAME
//! input (read from a shared `.bin`, f32 quantized identically by both sides'
//! proven `WriteTimeStep`), and dumps the same lines.
//!
//! ```sh
//! cargo run -p tesseract-ocr --example network_dump -- /tmp/eng.lstm /tmp/net_input.bin > /tmp/rust_net.tsv
//! /tmp/network_forward_oracle /tmp/eng.lstm /tmp/net_input.bin > /tmp/oracle_net.tsv
//! diff /tmp/oracle_net.tsv /tmp/rust_net.tsv   # byte-identical => B1 green
//! ```
#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use std::io::Write;
use std::path::Path;

use tesseract_ocr::{Network, Node};
use tesseract_recognizer::{FlexDim, NetworkIo, StrideMap, TRand};

/// Recursively describe the loaded tree (Rust-side structural sanity; the
/// C++ children are private so the oracle proves structure via the forward
/// output + the top `num_weights`).
fn describe(node: &Node, depth: usize, ni: usize) {
    let (name, no) = match node {
        Node::Input { shape } => (
            format!(
                "Input[{},{},{},{}]",
                shape.batch, shape.height, shape.width, shape.depth
            ),
            shape.depth as usize,
        ),
        Node::Series(s) => (
            "Series".to_string(),
            s.iter().fold(ni, |n, c| c.num_outputs(n)),
        ),
        Node::Parallel(s) => (
            "Parallel".to_string(),
            s.iter().map(|c| c.num_outputs(ni)).sum(),
        ),
        Node::Reversed { kind, .. } => (format!("Reversed{kind:?}"), node.num_outputs(ni)),
        Node::Convolve { half_x, half_y } => {
            (format!("Convolve[{half_x},{half_y}]"), node.num_outputs(ni))
        }
        Node::Maxpool { x_scale, y_scale } => (format!("Maxpool[{x_scale},{y_scale}]"), ni),
        Node::Reconfig { x_scale, y_scale } => (
            format!("Reconfig[{x_scale},{y_scale}]"),
            node.num_outputs(ni),
        ),
        Node::Lstm { lstm, summary } => (
            format!(
                "Lstm{}[ni={},ns={}]",
                if *summary { "Summary" } else { "" },
                lstm.num_inputs(),
                lstm.state_size()
            ),
            lstm.state_size(),
        ),
        Node::FullyConnected {
            weights,
            activation,
            ..
        } => (
            format!("Fc{activation:?}[{}]", weights.num_outputs()),
            weights.num_outputs(),
        ),
    };
    println!("node\t{depth}\t{name}\tni={ni}\tno={no}");
    match node {
        Node::Series(s) | Node::Parallel(s) => {
            let mut n = ni;
            for c in s {
                describe(c, depth + 1, n);
                n = c.num_outputs(if matches!(node, Node::Series(_)) {
                    n
                } else {
                    ni
                });
            }
        }
        Node::Reversed { child, .. } => describe(child, depth + 1, ni),
        _ => {}
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let lstm_path = args.get(1).map_or("/tmp/eng.lstm", String::as_str);
    let bin_path = args.get(2).map_or("/tmp/net_input.bin", String::as_str);
    let width: i32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(24);

    let bytes = std::fs::read(Path::new(lstm_path)).expect("read eng.lstm");
    let (net, consumed) = Network::from_le_bytes(&bytes).expect("load network");
    println!("nw\t{}", net.num_weights);
    println!("ni\t{}\tno\t{}\tconsumed\t{}", net.ni, net.no, consumed);
    describe(&net.root, 0, net.ni as usize);

    // Synthetic image: the Input shape's height/depth, chosen width. f32 in
    // [-1,1] (deterministic), quantized to int8 by the proven WriteTimeStep.
    let shape = net.input_shape.expect("network has an Input node");
    let (height, depth) = (shape.height, shape.depth);
    let mut map = StrideMap::default();
    map.set_stride(&[(height, width)]);
    let mut input = NetworkIo::default();
    input.resize_to_map(true, &map, depth as usize);

    // Fill + capture f32 for the shared .bin (batch,height,width,depth + f32s).
    let mut buf = Vec::new();
    for v in [1_i32, height, width, depth] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    let walk = map.clone();
    let mut idx = walk.index_first();
    loop {
        let t = idx.t() as usize;
        let vals: Vec<f32> = (0..depth)
            .map(|f| ((t as i32 * 17 + f * 5) % 200 - 100) as f32 / 100.0)
            .collect();
        for &v in &vals {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        input.write_time_step(t, &vals);
        if !idx.increment() {
            break;
        }
    }
    std::fs::File::create(bin_path)
        .and_then(|mut f| f.write_all(&buf))
        .expect("write net_input.bin");

    // Full-tree forward with a fixed randomizer seed (Convolve's out-of-image
    // noise; the oracle seeds identically).
    let mut rng = TRand::default();
    rng.set_seed(1);
    let out = net.forward(&input, &mut rng).expect("forward");
    println!(
        "oshape\t{}\t{}\t{}\t{}\t{}",
        out.stride_map().size(FlexDim::Batch),
        out.stride_map().size(FlexDim::Height),
        out.stride_map().size(FlexDim::Width),
        out.width(),
        out.num_features()
    );
    for t in 0..out.width() {
        print!("o\t{t}");
        if out.int_mode() {
            for &v in out.i(t) {
                print!("\t{v}");
            }
        } else {
            for &v in out.f(t) {
                print!("\t{:08x}", v.to_bits());
            }
        }
        println!();
    }
}
