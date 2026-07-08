//! The network-tree loader + runnable forward — recognizer **B1**: the
//! transcode of `Network::CreateFromFile` (`network.cpp:214-283`) +
//! `Plumbing::DeSerialize` (recursive children) + the per-subclass payloads,
//! building a [`Node`] tree whose [`Node::forward_io`] runs the REAL model
//! (eng.lstm `[1,36,0,1[C3,3Ft16]Mp3,3TxyLfys48Lfx96RxLrx96Lfx192Fc111]`)
//! over a [`NetworkIo`] grid.
//!
//! Per Core-First: the per-node **header** parses via the Core's proven
//! [`tesseract_core::network::NetworkHeader`] (`E-OCR-NETWORK-SINK-1`); the
//! compute **payloads** parse via the recognizer's proven
//! [`WeightMatrix`]/[`Lstm`] readers; the walk composes the A1-A5 grid ops.
//!
//! ## Wire format (read from the C++ source, banked in the v2 plan)
//!
//! header → subclass payload:
//! - Series/Parallel/Reversed (all `Plumbing`): `u32 count` + recursive
//!   children (+ `learning_rates` f32 vec ONLY if
//!   `flags & NF_LAYER_SPECIFIC_LR` — rejected here; inference models don't
//!   carry it).
//! - Input: `StaticShape` = 5×i32 `batch,height,width,depth,loss_type`.
//! - FullyConnected (`Logistic..SoftmaxNoCtc`): one [`WeightMatrix`].
//! - Convolve: `i32 half_x, half_y` (`no = ni·(2hx+1)·(2hy+1)` recomputed).
//! - Reconfig/Maxpool: `i32 x_scale, y_scale` (Maxpool: `no = ni`).
//! - LSTM / LstmSummary: `i32 na` + gates CI,GI,GF1,GO (GFS skipped — 1-D
//!   only; 2-D LSTMs are rejected as unsupported, eng is 1-D).
//!
//! ## Forward semantics (from the C++ Forward bodies, all proven pieces)
//!
//! - `Series`: chain node outputs (`series.cpp:Forward` buffer plumbing is
//!   value-immaterial). Each node's output **inherits its input's int mode
//!   except softmax FCs (float)** — the inter-stage requant is
//!   `write_time_step`'s proven quantizer.
//! - `Parallel`: same input to every child, `copy_packing` the outputs.
//! - `Reversed`: reverse/transpose in → child → reverse/transpose out.
//! - `Lstm`: the grid walk is per-(batch,row): state+output are **zeroed at
//!   the end of every row** (`lstm.cpp` `IsLast(FD_WIDTH)`), so each row is an
//!   independent Leaf-5 [`Lstm::forward`] sequence. `LstmSummary` writes only
//!   each row's final output, into a `ResizeXTo1` map in row order.
//! - `FullyConnected`: per-timestep [`fully_connected_forward`].

use tesseract_core::network::{NetworkHeader, NetworkType};
use tesseract_recognizer::{
    convolve_forward, fully_connected_forward, maxpool_forward, reconfig_forward, FcActivation,
    FlexDim, Lstm, NetworkIo, TRand, WeightMatrix,
};

/// `NF_LAYER_SPECIFIC_LR` (`network.h` NetworkFlags) — training-side per-layer
/// learning rates; a model carrying it is not an inference snapshot.
const NF_LAYER_SPECIFIC_LR: i32 = 64;

/// The image-input shape (`StaticShape`, 5×i32 on the wire).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputShape {
    /// Batch size (usually 1).
    pub batch: i32,
    /// Target height (0 = variable).
    pub height: i32,
    /// Target width (0 = variable).
    pub width: i32,
    /// Depth / channels.
    pub depth: i32,
    /// `LossType` discriminant (carried, unused at inference).
    pub loss_type: i32,
}

/// How a `Reversed` wrapper reorients the grid around its child.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReverseKind {
    /// `NT_XREVERSED` — `CopyWithXReversal`.
    X,
    /// `NT_YREVERSED` — `CopyWithYReversal`.
    Y,
    /// `NT_XYTRANSPOSE` — `CopyWithXYTranspose`.
    Txy,
}

/// A runnable network node — the assembly-tier execution tree built from the
/// serialized model. NOT a parallel object model: headers/typing come from the
/// Core ([`NetworkType`]), compute from the recognizer's proven leaves.
#[derive(Debug)]
pub enum Node {
    /// `NT_INPUT` — declares the image shape; forward is identity (the caller
    /// supplies the image-shaped grid).
    Input {
        /// The declared shape.
        shape: InputShape,
    },
    /// `NT_SERIES`.
    Series(Vec<Node>),
    /// `NT_PARALLEL` / `NT_PAR_RL_LSTM` / `NT_REPLICATED` (same forward).
    Parallel(Vec<Node>),
    /// `NT_XREVERSED` / `NT_YREVERSED` / `NT_XYTRANSPOSE` (one child).
    Reversed {
        /// Which reorientation.
        kind: ReverseKind,
        /// The wrapped child.
        child: Box<Node>,
    },
    /// `NT_CONVOLVE`.
    Convolve {
        /// Window half-width.
        half_x: i32,
        /// Window half-height.
        half_y: i32,
    },
    /// `NT_MAXPOOL`.
    Maxpool {
        /// x window/scale.
        x_scale: i32,
        /// y window/scale.
        y_scale: i32,
    },
    /// `NT_RECONFIG`.
    Reconfig {
        /// x window/scale.
        x_scale: i32,
        /// y window/scale.
        y_scale: i32,
    },
    /// `NT_LSTM` / `NT_LSTM_SUMMARY` (1-D int path).
    Lstm {
        /// The Leaf-5 gate block (boxed — it dwarfs the other variants,
        /// `clippy::large_enum_variant`).
        lstm: Box<Lstm>,
        /// True for `NT_LSTM_SUMMARY` (keep only each row's final output).
        summary: bool,
    },
    /// The fully-connected family (`Logistic..SoftmaxNoCtc`).
    FullyConnected {
        /// The layer weights (boxed alongside `Lstm`).
        weights: Box<WeightMatrix>,
        /// The non-linearity.
        activation: FcActivation,
        /// True for `Softmax`/`SoftmaxNoCtc` — output is float
        /// (`ResizeFloat`), everything else inherits the input mode.
        float_output: bool,
    },
}

/// A loaded network: the runnable tree plus the top header facts.
#[derive(Debug)]
pub struct Network {
    /// The runnable tree.
    pub root: Node,
    /// Top-level input count (eng: 36).
    pub ni: i32,
    /// Top-level output count = the class/lattice width (eng: 111).
    pub no: i32,
    /// Cumulative weight count (eng: 385807) — the load self-check.
    pub num_weights: i32,
    /// The root's spec-ish name (diagnostic).
    pub name: String,
    /// The image input shape found in the tree (from the `NT_INPUT` node).
    pub input_shape: Option<InputShape>,
}

/// A failure loading or running the network tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetError {
    /// The byte stream ended mid-field.
    UnexpectedEof,
    /// A header failed to parse (Core error text).
    BadHeader(String),
    /// A node type this inference walk does not support (2-D LSTM,
    /// softmax-in-LSTM, TensorFlow, training-only flags...).
    Unsupported(&'static str),
    /// A recognizer payload failed to parse.
    BadPayload(String),
    /// A forward-pass shape/mode violation.
    Forward(&'static str),
}

impl std::fmt::Display for NetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "network stream ended mid-field"),
            Self::BadHeader(e) => write!(f, "bad node header: {e}"),
            Self::Unsupported(w) => write!(f, "unsupported network feature: {w}"),
            Self::BadPayload(e) => write!(f, "bad node payload: {e}"),
            Self::Forward(w) => write!(f, "forward error: {w}"),
        }
    }
}

impl std::error::Error for NetError {}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], NetError> {
        let end = self.pos.checked_add(n).ok_or(NetError::UnexpectedEof)?;
        let s = self
            .bytes
            .get(self.pos..end)
            .ok_or(NetError::UnexpectedEof)?;
        self.pos = end;
        Ok(s)
    }
    fn i32(&mut self) -> Result<i32, NetError> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, NetError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

impl Network {
    /// Load a network from the serialized bytes (the head of an `.lstm`
    /// component) — `Network::CreateFromFile` recursively. Returns the network
    /// and the number of bytes consumed (the recognizer stream continues with
    /// the charset/recoder for B2).
    ///
    /// # Errors
    ///
    /// [`NetError`] on truncation, unknown/unsupported node types, or payload
    /// parse failures.
    pub fn from_le_bytes(bytes: &[u8]) -> Result<(Network, usize), NetError> {
        let mut cur = Cursor { bytes, pos: 0 };
        let (root, hdr, shape) = read_node(&mut cur)?;
        Ok((
            Network {
                root,
                ni: hdr.ni,
                no: hdr.no,
                num_weights: hdr.num_weights,
                name: hdr.name,
                input_shape: shape,
            },
            cur.pos,
        ))
    }

    /// Run the tree over an input grid. `rng` feeds `Convolve`'s out-of-image
    /// noise (seed it exactly as the recognizer does for parity).
    ///
    /// # Errors
    ///
    /// [`NetError::Forward`] on mode/shape violations (e.g. a float grid into
    /// the int-only LSTM walk).
    pub fn forward(&self, input: &NetworkIo, rng: &mut TRand) -> Result<NetworkIo, NetError> {
        self.root.forward_io(input, rng)
    }

    /// `LSTMRecognizer::SimpleTextOutput()` (`lstmrecognizer.h:84-86`):
    /// `OutputLossType() == LT_SOFTMAX`. The loss type comes from the tree's
    /// `OutputShape` walk (`fullyconnected.cpp:47-58`): only the
    /// fully-connected family sets it — `NT_SOFTMAX → LT_CTC`,
    /// `NT_SOFTMAX_NO_CTC → LT_SOFTMAX`, `NT_LOGISTIC → LT_LOGISTIC`; every
    /// other node passes its input shape through, so the FINAL
    /// output-producing node decides (`Series` → last child, wrappers → their
    /// child).
    ///
    /// **The distinction that matters:** eng.lstm ends in `O1c111` — the `c`
    /// is `NT_SOFTMAX` = softmax activation WITH **CTC** loss → this returns
    /// `false`, and the beam must run the full CTC semantics (top-n flags,
    /// duplicate collapse). Softmax ACTIVATION does NOT imply `LT_SOFTMAX`
    /// LOSS — conflating the two makes the beam re-emit every per-timestep
    /// spike as a fresh character (systematic repeats like `TTTThhheee` on
    /// real text; unfalsifiable on noise fixtures).
    /// `Network::XScaleFactor()` (`network.h:214`, overrides in
    /// `series.cpp:90-96` / `reconfig.cpp` / `plumbing.cpp:124-126`): the
    /// tree's total x-subsampling — `Reconfig`/`Maxpool` contribute their
    /// `x_scale`, `Series` multiplies its children, plumbing wrappers take
    /// their first child, everything else is 1 (eng.lstm: 3, from `Mp3,3`).
    /// `Input::PrepareLSTMInputs` uses it as the minimum width/height a line
    /// image must have after `PreScale` — smaller lines are UNRECOGNIZABLE
    /// and the real pipeline skips them ("Image too small to scale!!").
    #[must_use]
    pub fn x_scale_factor(&self) -> i32 {
        fn xsf(node: &Node) -> i32 {
            match node {
                Node::Maxpool { x_scale, .. } | Node::Reconfig { x_scale, .. } => *x_scale,
                Node::Series(children) => children.iter().map(xsf).product(),
                Node::Parallel(children) => children.first().map_or(1, xsf),
                Node::Reversed { child, .. } => xsf(child),
                _ => 1,
            }
        }
        xsf(&self.root)
    }

    #[must_use]
    pub fn simple_text_output(&self) -> bool {
        fn loss_is_softmax(node: &Node) -> bool {
            match node {
                Node::FullyConnected { activation, .. } => {
                    matches!(activation, FcActivation::SoftmaxNoCtc)
                }
                Node::Series(children) => children.last().is_some_and(loss_is_softmax),
                Node::Reversed { child, .. } => loss_is_softmax(child),
                // Parallel joins outputs; loss stays LT_NONE unless a branch
                // sets it — eng never hits this, mirror the pass-through.
                Node::Parallel(children) => children.last().is_some_and(loss_is_softmax),
                _ => false,
            }
        }
        loss_is_softmax(&self.root)
    }
}

/// `Plumbing::DeSerialize` reads a trailing `learning_rates_` (`Vec<f32>`) —
/// `u32 count` + `count × f32` — after its children WHEN `NF_LAYER_SPECIFIC_LR`
/// is set (`plumbing.cpp`). Inference models still carry the flag; the rates
/// are training-only, so read past them. Non-Plumbing nodes never serialize it.
fn skip_layer_lr(cur: &mut Cursor<'_>, network_flags: i32) -> Result<(), NetError> {
    if network_flags & NF_LAYER_SPECIFIC_LR != 0 {
        let count = cur.u32()?;
        cur.take(count as usize * 4)?;
    }
    Ok(())
}

/// Read one node (header + payload + children), returning the top-level
/// header and any `NT_INPUT` shape found in the subtree.
fn read_node(cur: &mut Cursor<'_>) -> Result<(Node, NetworkHeader, Option<InputShape>), NetError> {
    let (hdr, used) = NetworkHeader::from_le_bytes(&cur.bytes[cur.pos..])
        .map_err(|e| NetError::BadHeader(format!("{e:?}")))?;
    cur.pos += used;
    let mut shape_found = None;
    let node = match hdr.ntype {
        NetworkType::Input => {
            let shape = InputShape {
                batch: cur.i32()?,
                height: cur.i32()?,
                width: cur.i32()?,
                depth: cur.i32()?,
                loss_type: cur.i32()?,
            };
            shape_found = Some(shape);
            Node::Input { shape }
        }
        NetworkType::Series
        | NetworkType::Parallel
        | NetworkType::Replicated
        | NetworkType::ParRlLstm
        | NetworkType::ParUdLstm => {
            let count = cur.u32()?;
            let mut children = Vec::with_capacity(count as usize);
            for _ in 0..count {
                let (child, _, s) = read_node(cur)?;
                if shape_found.is_none() {
                    shape_found = s;
                }
                children.push(child);
            }
            skip_layer_lr(cur, hdr.network_flags)?;
            if hdr.ntype == NetworkType::Series {
                Node::Series(children)
            } else {
                Node::Parallel(children)
            }
        }
        NetworkType::Par2dLstm => return Err(NetError::Unsupported("NT_PAR_2D_LSTM (2-D)")),
        NetworkType::XReversed | NetworkType::YReversed | NetworkType::XyTranspose => {
            let count = cur.u32()?;
            if count != 1 {
                return Err(NetError::Unsupported("Reversed with != 1 child"));
            }
            let (child, _, s) = read_node(cur)?;
            shape_found = s;
            skip_layer_lr(cur, hdr.network_flags)?;
            let kind = match hdr.ntype {
                NetworkType::XReversed => ReverseKind::X,
                NetworkType::YReversed => ReverseKind::Y,
                _ => ReverseKind::Txy,
            };
            Node::Reversed {
                kind,
                child: Box::new(child),
            }
        }
        NetworkType::Convolve => Node::Convolve {
            half_x: cur.i32()?,
            half_y: cur.i32()?,
        },
        NetworkType::Maxpool => Node::Maxpool {
            x_scale: cur.i32()?,
            y_scale: cur.i32()?,
        },
        NetworkType::Reconfig => Node::Reconfig {
            x_scale: cur.i32()?,
            y_scale: cur.i32()?,
        },
        NetworkType::Lstm | NetworkType::LstmSummary => {
            let (lstm, used) = Lstm::from_le_bytes(&cur.bytes[cur.pos..])
                .map_err(|e| NetError::BadPayload(format!("{e}")))?;
            cur.pos += used;
            // The Leaf-5 parser reads na + 4 gates (the 1-D layout). A 2-D
            // LSTM (na - nf == ni + 2*ns) carries a 5th gate (GFS) this walk
            // does not support — detect and reject rather than mis-parse.
            let (ni, ns) = (lstm.num_inputs(), lstm.state_size());
            if hdr.ni as usize + 2 * ns == ni + ns {
                return Err(NetError::Unsupported("2-D LSTM (GFS gate)"));
            }
            Node::Lstm {
                lstm: Box::new(lstm),
                summary: hdr.ntype == NetworkType::LstmSummary,
            }
        }
        NetworkType::LstmSoftmax | NetworkType::LstmSoftmaxEncoded => {
            return Err(NetError::Unsupported("LSTM-with-softmax variants"))
        }
        NetworkType::Logistic
        | NetworkType::PosClip
        | NetworkType::SymClip
        | NetworkType::Tanh
        | NetworkType::Relu
        | NetworkType::Linear
        | NetworkType::Softmax
        | NetworkType::SoftmaxNoCtc => {
            let (weights, used) = WeightMatrix::from_le_bytes_prefix(&cur.bytes[cur.pos..])
                .map_err(|e| NetError::BadPayload(format!("{e}")))?;
            cur.pos += used;
            let activation = match hdr.ntype {
                NetworkType::Logistic => FcActivation::Logistic,
                NetworkType::PosClip => FcActivation::PosClip,
                NetworkType::SymClip => FcActivation::SymClip,
                NetworkType::Tanh => FcActivation::Tanh,
                NetworkType::Relu => FcActivation::Relu,
                NetworkType::Linear => FcActivation::Linear,
                _ => FcActivation::Softmax,
            };
            Node::FullyConnected {
                weights: Box::new(weights),
                activation,
                float_output: matches!(hdr.ntype, NetworkType::Softmax | NetworkType::SoftmaxNoCtc),
            }
        }
        NetworkType::None | NetworkType::TensorFlow => {
            return Err(NetError::Unsupported("NT_NONE / NT_TENSORFLOW"))
        }
    };
    Ok((node, hdr, shape_found))
}

impl Node {
    /// The number of output features this node produces for `ni` inputs.
    #[must_use]
    pub fn num_outputs(&self, ni: usize) -> usize {
        match self {
            Node::Input { shape } => shape.depth as usize,
            Node::Series(stack) => stack.iter().fold(ni, |n, l| l.num_outputs(n)),
            Node::Parallel(stack) => stack.iter().map(|l| l.num_outputs(ni)).sum(),
            Node::Reversed { child, .. } => child.num_outputs(ni),
            Node::Convolve { half_x, half_y } => {
                ni * ((2 * half_x + 1) * (2 * half_y + 1)) as usize
            }
            Node::Maxpool { .. } => ni,
            Node::Reconfig { x_scale, y_scale } => ni * (x_scale * y_scale) as usize,
            Node::Lstm { lstm, .. } => lstm.state_size(),
            Node::FullyConnected { weights, .. } => weights.num_outputs(),
        }
    }

    /// Run this node over an input grid (`Forward` composed from the proven
    /// leaves; see the module docs for the exact C++ semantics carried).
    ///
    /// # Errors
    ///
    /// [`NetError::Forward`] on mode violations.
    pub fn forward_io(&self, input: &NetworkIo, rng: &mut TRand) -> Result<NetworkIo, NetError> {
        match self {
            Node::Input { .. } => Ok(input.clone()),
            Node::Series(stack) => {
                let Some((first, rest)) = stack.split_first() else {
                    return Err(NetError::Forward("empty Series"));
                };
                let mut out = first.forward_io(input, rng)?;
                for node in rest {
                    out = node.forward_io(&out, rng)?;
                }
                Ok(out)
            }
            Node::Parallel(stack) => {
                if stack.is_empty() {
                    return Err(NetError::Forward("empty Parallel"));
                }
                let outs: Vec<NetworkIo> = stack
                    .iter()
                    .map(|n| n.forward_io(input, rng))
                    .collect::<Result<_, _>>()?;
                let total: usize = outs.iter().map(NetworkIo::num_features).sum();
                let mut packed = NetworkIo::default();
                packed.resize_like(&outs[0], total);
                let mut off = 0;
                for o in &outs {
                    off = packed.copy_packing(o, off);
                }
                Ok(packed)
            }
            Node::Reversed { kind, child } => {
                let mut rev_in = NetworkIo::default();
                match kind {
                    ReverseKind::X => rev_in.copy_with_x_reversal(input),
                    ReverseKind::Y => rev_in.copy_with_y_reversal(input),
                    ReverseKind::Txy => rev_in.copy_with_xy_transpose(input),
                }
                let rev_out = child.forward_io(&rev_in, rng)?;
                let mut out = NetworkIo::default();
                match kind {
                    ReverseKind::X => out.copy_with_x_reversal(&rev_out),
                    ReverseKind::Y => out.copy_with_y_reversal(&rev_out),
                    ReverseKind::Txy => out.copy_with_xy_transpose(&rev_out),
                }
                Ok(out)
            }
            Node::Convolve { half_x, half_y } => Ok(convolve_forward(input, *half_x, *half_y, rng)),
            Node::Maxpool { x_scale, y_scale } => Ok(maxpool_forward(input, *x_scale, *y_scale)),
            Node::Reconfig { x_scale, y_scale } => Ok(reconfig_forward(input, *x_scale, *y_scale)),
            Node::Lstm { lstm, summary } => lstm_forward_io(lstm, *summary, input),
            Node::FullyConnected {
                weights,
                activation,
                float_output,
            } => {
                if !input.int_mode() {
                    return Err(NetError::Forward("float input to int FC walk"));
                }
                let no = weights.num_outputs();
                let mut out = NetworkIo::default();
                if *float_output {
                    out.resize_float(input, no);
                } else {
                    out.resize_like(input, no);
                }
                // Per-timestep over the stride walk (valid cells only, exactly
                // fullyconnected.cpp's src_index loop; padding cells stay 0).
                let map = input.stride_map().clone();
                let mut idx = map.index_first();
                loop {
                    let t = idx.t() as usize;
                    let line = fully_connected_forward(weights, input.i(t), *activation)
                        .map_err(|_| NetError::Forward("FC dimension mismatch"))?;
                    out.write_time_step(t, &line);
                    if !idx.increment() {
                        break;
                    }
                }
                Ok(out)
            }
        }
    }
}

/// The 1-D LSTM grid walk: state + output zero at every row start (the C++
/// zeroes them at each row END — identical since rows are independent), each
/// row an independent Leaf-5 sequence over the image's TRUE width; `summary`
/// keeps only each row's last output, packed into the `ResizeXTo1` map in
/// (batch, y) order.
fn lstm_forward_io(lstm: &Lstm, summary: bool, input: &NetworkIo) -> Result<NetworkIo, NetError> {
    if !input.int_mode() {
        return Err(NetError::Forward("float input to int LSTM walk"));
    }
    let ns = lstm.state_size();
    let mut out = NetworkIo::default();
    if summary {
        out.resize_x_to_1(input, ns);
    } else {
        out.resize_like(input, ns);
    }
    let map = input.stride_map().clone();
    let out_map = out.stride_map().clone();
    let mut dest_idx = out_map.index_first();
    let batches = map.size(FlexDim::Batch);
    for b in 0..batches {
        // Rows run to the image's true height; columns to its true width —
        // exactly the ragged src_index walk.
        let probe = map.index_at(b, 0, 0);
        let max_y = probe.max_index_of_dim(FlexDim::Height);
        let max_x = probe.max_index_of_dim(FlexDim::Width);
        for y in 0..=max_y {
            let t0 = map.index_at(b, y, 0).t() as usize;
            let row_len = (max_x + 1) as usize;
            let rows: Vec<&[i8]> = (0..row_len).map(|k| input.i(t0 + k)).collect();
            let outputs = lstm
                .forward(&rows)
                .map_err(|_| NetError::Forward("LSTM dimension mismatch"))?;
            if summary {
                let last = outputs.last().ok_or(NetError::Forward("empty LSTM row"))?;
                out.write_time_step(dest_idx.t() as usize, last);
                dest_idx.increment();
            } else {
                for (k, line) in outputs.iter().enumerate() {
                    out.write_time_step(t0 + k, line);
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize a node header in the C++ `Network::Serialize` wire form
    /// (`network.cpp`): `i8 tag`(NT_NONE=0), `string type_name` (`u32 len` +
    /// bytes), `i8 training`, `i8 needs_backprop`, `i32 flags`, `i32 ni`,
    /// `i32 no`, `i32 num_weights`, `string name`. The discriminant is the
    /// `kTypeNames` string, NOT a raw ordinal — mirrors the Core's own
    /// `header_bytes` helper (`lance-graph-contract::network`).
    fn header(
        type_name: &str,
        flags: i32,
        ni: i32,
        no: i32,
        num_weights: i32,
        name: &str,
    ) -> Vec<u8> {
        let mut b: Vec<u8> = vec![0]; // tag = NT_NONE
        b.extend_from_slice(&(type_name.len() as u32).to_le_bytes());
        b.extend_from_slice(type_name.as_bytes());
        b.push(0); // training = TS_DISABLED
        b.push(0); // needs_backprop = false
        b.extend_from_slice(&flags.to_le_bytes());
        b.extend_from_slice(&ni.to_le_bytes());
        b.extend_from_slice(&no.to_le_bytes());
        b.extend_from_slice(&num_weights.to_le_bytes());
        b.extend_from_slice(&(name.len() as u32).to_le_bytes());
        b.extend_from_slice(name.as_bytes());
        b
    }

    /// Build `Series[ Input[1,4,0,1], Maxpool[2,2] ]` — exercises the recursive
    /// Plumbing count+children path, the Input StaticShape, and the Maxpool
    /// 2×i32 payload. Series flags=0 so no trailing `learning_rates_` vec.
    #[test]
    fn loads_a_minimal_series_tree() {
        let mut b = header("Series", 0, 1, 1, 0, "root");
        b.extend_from_slice(&2_u32.to_le_bytes()); // child count
                                                   // child 0: Input[1,4,0,1]
        b.extend(header("Input", 0, 1, 1, 0, "in"));
        for v in [1_i32, 4, 0, 1, 0] {
            b.extend_from_slice(&v.to_le_bytes()); // StaticShape batch,h,w,d,loss
        }
        // child 1: Maxpool[2,2]
        b.extend(header("Maxpool", 0, 1, 1, 0, "mp"));
        for v in [2_i32, 2] {
            b.extend_from_slice(&v.to_le_bytes()); // x_scale, y_scale
        }

        let (net, consumed) = Network::from_le_bytes(&b).expect("load");
        assert_eq!(consumed, b.len(), "consumes exactly the tree");
        assert_eq!(net.ni, 1);
        assert_eq!(
            net.input_shape,
            Some(InputShape {
                batch: 1,
                height: 4,
                width: 0,
                depth: 1,
                loss_type: 0
            })
        );
        match &net.root {
            Node::Series(children) => {
                assert_eq!(children.len(), 2);
                assert!(matches!(children[0], Node::Input { .. }));
                assert!(matches!(
                    children[1],
                    Node::Maxpool {
                        x_scale: 2,
                        y_scale: 2
                    }
                ));
            }
            other => panic!("expected Series, got {other:?}"),
        }
    }

    /// A Plumbing node with `NF_LAYER_SPECIFIC_LR` carries a trailing
    /// `learning_rates_` (`u32 count` + `count × f32`) that must be read past.
    #[test]
    fn skips_layer_specific_learning_rates() {
        // Series with NF_LAYER_SPECIFIC_LR set → a trailing learning_rates_ vec
        // after the children that the loader must read past.
        let mut b = header("Series", NF_LAYER_SPECIFIC_LR, 1, 1, 0, "s");
        b.extend_from_slice(&1_u32.to_le_bytes()); // 1 child
        b.extend(header("Maxpool", 0, 1, 1, 0, "mp")); // Maxpool child
        for v in [2_i32, 2] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        // trailing learning_rates_: 3 floats
        b.extend_from_slice(&3_u32.to_le_bytes());
        for f in [0.1_f32, 0.2, 0.3] {
            b.extend_from_slice(&f.to_le_bytes());
        }
        let (net, consumed) = Network::from_le_bytes(&b).expect("load");
        assert_eq!(consumed, b.len(), "reads past the LR vec");
        assert!(matches!(net.root, Node::Series(_)));
    }

    #[test]
    fn num_outputs_composes_down_a_series() {
        // Convolve[1,1] on ni=1 -> 9; Maxpool keeps 9; Reconfig[2,2] -> 36.
        let tree = Node::Series(vec![
            Node::Convolve {
                half_x: 1,
                half_y: 1,
            },
            Node::Maxpool {
                x_scale: 3,
                y_scale: 3,
            },
            Node::Reconfig {
                x_scale: 2,
                y_scale: 2,
            },
        ]);
        assert_eq!(tree.num_outputs(1), 36);
    }
}
