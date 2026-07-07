//! Dump the A1 `NetworkIo`/`StrideMap`/`TRand` byte-parity surface — a fixed
//! ragged 3-image scenario exercised twice (int8 then float), printed as TSV.
//! The libtesseract oracle (`/tmp/netio_oracle.cpp`) runs the IDENTICAL
//! operations through the REAL public `NetworkIO`/`StrideMap`/`TRand` API and
//! prints the same lines; `diff` == byte-parity for the whole grid substrate.
//!
//! ```sh
//! cargo run -p tesseract-recognizer --example netio_dump > /tmp/rust_netio.tsv
//! /tmp/netio_oracle > /tmp/oracle_netio.tsv
//! diff /tmp/oracle_netio.tsv /tmp/rust_netio.tsv   # byte-identical => A1 green
//! ```
#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use tesseract_recognizer::{FlexDim, NetworkIo, StrideMap, TRand};

const NF: usize = 3;

fn dim_char(d: FlexDim) -> char {
    match d {
        FlexDim::Batch => 'B',
        FlexDim::Height => 'H',
        FlexDim::Width => 'W',
    }
}

/// Fill every VALID cell deterministically: input[f] = ((t·31 + f·17) % 200 − 100)/100.
fn fill(io: &mut NetworkIo) {
    let map = io.stride_map().clone();
    let mut idx = map.index_first();
    loop {
        let t = idx.t() as usize;
        let vals: Vec<f32> = (0..NF as i32)
            .map(|f| ((t as i32 * 31 + f * 17) % 200 - 100) as f32 / 100.0)
            .collect();
        io.write_time_step(t, &vals);
        if !idx.increment() {
            break;
        }
    }
}

fn dump_store(tag: &str, io: &NetworkIo) {
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
    let mode = i32::from(int_mode);
    let mut map = StrideMap::default();
    map.set_stride(&[(5, 7), (3, 4), (4, 6)]);
    println!(
        "shape\t{mode}\t{}\t{}\t{}\t{}\t{NF}",
        map.size(FlexDim::Batch),
        map.size(FlexDim::Height),
        map.size(FlexDim::Width),
        map.width()
    );

    // Forward walk: t + [b,y,x] at every valid step.
    let mut idx = map.index_first();
    loop {
        println!(
            "w\t{}\t{}\t{}\t{}",
            idx.t(),
            idx.index(FlexDim::Batch),
            idx.index(FlexDim::Height),
            idx.index(FlexDim::Width)
        );
        if !idx.increment() {
            break;
        }
    }
    // Reverse walk.
    let mut idx = map.index_last();
    loop {
        println!("r\t{}", idx.t());
        if !idx.decrement() {
            break;
        }
    }
    // AddOffset probes (b, y, x, dim, off).
    let probes: [(i32, i32, i32, FlexDim, i32); 10] = [
        (0, 0, 0, FlexDim::Width, 3),
        (0, 4, 6, FlexDim::Width, 1),
        (1, 0, 3, FlexDim::Width, 1),
        (1, 2, 3, FlexDim::Height, 1),
        (2, 3, 5, FlexDim::Height, -1),
        (1, 0, 0, FlexDim::Batch, 1),
        (2, 0, 0, FlexDim::Batch, 1),
        (0, 2, 2, FlexDim::Height, 2),
        (1, 1, 1, FlexDim::Width, -2),
        (2, 3, 0, FlexDim::Width, 5),
    ];
    for (b, y, x, dim, off) in probes {
        let mut p = map.index_at(b, y, x);
        let valid = p.add_offset(off, dim);
        println!(
            "o\t{b}\t{y}\t{x}\t{}\t{off}\t{}\t{}",
            dim_char(dim),
            i32::from(valid),
            p.t()
        );
    }

    // The filled store (valid cells written, ragged padding zeroed).
    let mut io = NetworkIo::default();
    io.resize_to_map(int_mode, &map, NF);
    fill(&mut io);
    dump_store("s", &io);

    // XY transpose + the two reversals.
    let mut tr = NetworkIo::default();
    tr.copy_with_xy_transpose(&io);
    println!(
        "tshape\t{}\t{}\t{}\t{}",
        tr.stride_map().size(FlexDim::Batch),
        tr.stride_map().size(FlexDim::Height),
        tr.stride_map().size(FlexDim::Width),
        tr.width()
    );
    dump_store("t", &tr);
    let mut xr = NetworkIo::default();
    xr.copy_with_x_reversal(&io);
    dump_store("x", &xr);
    let mut yr = NetworkIo::default();
    yr.copy_with_y_reversal(&io);
    dump_store("y", &yr);

    // Maxpool primitive: 2x2-scaled dest, running max over 4 source steps.
    let mut mp = NetworkIo::default();
    mp.resize_scaled(&io, 2, 2, NF);
    println!(
        "mshape\t{}\t{}\t{}\t{}",
        mp.stride_map().size(FlexDim::Batch),
        mp.stride_map().size(FlexDim::Height),
        mp.stride_map().size(FlexDim::Width),
        mp.width()
    );
    let mut max_line = [0_i32; NF];
    mp.copy_time_step_from(0, &io, 0);
    for src_t in [1_usize, 7, 35, 36] {
        mp.maxpool_time_step(0, &io, src_t, &mut max_line);
    }
    print!("m\t0");
    if int_mode {
        for &v in mp.i(0) {
            print!("\t{v}");
        }
    } else {
        for &v in mp.f(0) {
            print!("\t{:08x}", v.to_bits());
        }
    }
    for m in max_line {
        print!("\t{m}");
    }
    println!();

    // TRand: 4 raw draws then a Randomize fill of one padding row.
    let mut rng = TRand::default();
    rng.set_seed(999);
    for _ in 0..4 {
        println!("rand\t{}", rng.int_rand());
    }
    io.randomize(39, 0, NF, &mut rng);
    print!("rnd\t39");
    if int_mode {
        for &v in io.i(39) {
            print!("\t{v}");
        }
    } else {
        for &v in io.f(39) {
            print!("\t{:08x}", v.to_bits());
        }
    }
    println!();

    // Shape echoes for the derived resizes.
    let mut x1 = NetworkIo::default();
    x1.resize_x_to_1(&io, NF);
    println!(
        "xshape\t{}\t{}\t{}\t{}",
        x1.stride_map().size(FlexDim::Batch),
        x1.stride_map().size(FlexDim::Height),
        x1.stride_map().size(FlexDim::Width),
        x1.width()
    );
}

fn main() {
    run(true);
    run(false);
}
