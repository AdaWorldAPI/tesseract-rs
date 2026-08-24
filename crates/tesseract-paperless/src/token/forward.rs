//! The forward-prediction input surface.
//!
//! ```text
//!   TOKEN SEQUENCE = RESIDENT.  MODEL STATE = EPHEMERAL.  NEITHER OWNS THE OTHER.
//! ```
//!
//! This module does NOT train a language model, and the counting predictor
//! below is not one. Its whole job is to answer a narrow question: does the
//! resident lane hand a forward predictor a usable input surface — ordered ids,
//! borrowed, with no second population and no separate tokenization?
//!
//! [`windows`] is the surface: `(&[u8] context, u8 next)` pairs cut straight out
//! of the borrowed view. An LSTM would consume exactly this and keep its
//! weights, hidden state and logits entirely to itself.
//!
//! The predictor is an order-k count table so that the surface can be exercised
//! end to end and reported with a number instead of an assertion. A count table
//! is a BASELINE, not a claim about what a trained model would achieve; the
//! probe says so where it prints the number.

use std::collections::HashMap;

use crate::token::lane::TokenStreamView;

/// Borrowed `(context, next)` windows over a view. Allocation-free.
pub fn windows<'a>(
    view: &TokenStreamView<'a>,
    k: usize,
) -> impl Iterator<Item = (&'a [u8], u8)> + 'a {
    let ids = view.ids();
    (k..ids.len()).map(move |i| (&ids[i - k..i], ids[i]))
}

/// An order-k next-token count table. Ephemeral by construction: it is built
/// from borrowed windows and dropped; nothing about it is resident.
#[derive(Debug, Default)]
pub struct CountPredictor {
    k: usize,
    table: HashMap<Vec<u8>, [u32; 256]>,
    total: HashMap<Vec<u8>, u32>,
}

impl CountPredictor {
    /// Empty order-k predictor.
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            k,
            table: HashMap::new(),
            total: HashMap::new(),
        }
    }

    /// Observe every window of one view.
    pub fn observe(&mut self, view: &TokenStreamView<'_>) {
        for (ctx, next) in windows(view, self.k) {
            let e = self.table.entry(ctx.to_vec()).or_insert([0; 256]);
            e[next as usize] += 1;
            *self.total.entry(ctx.to_vec()).or_insert(0) += 1;
        }
    }

    /// Most likely next id for a context, if the context was seen.
    ///
    /// # Panics
    /// Never: the index comes from `enumerate` over a 256-element array, so the
    /// `u8` conversion cannot fail.
    #[must_use]
    pub fn predict(&self, ctx: &[u8]) -> Option<u8> {
        let counts = self.table.get(ctx)?;
        let (best, n) = counts
            .iter()
            .enumerate()
            .max_by_key(|(id, &c)| (c, std::cmp::Reverse(*id)))?;
        (*n > 0).then(|| u8::try_from(best).expect("index < 256"))
    }

    /// Distinct contexts observed.
    #[must_use]
    pub fn contexts(&self) -> usize {
        self.table.len()
    }

    /// Order.
    #[must_use]
    pub const fn order(&self) -> usize {
        self.k
    }
}

/// The outcome of scoring one held-out set.
#[derive(Clone, Copy, Debug, Default)]
pub struct ForwardScore {
    /// Positions scored.
    pub scored: usize,
    /// Positions whose context was never seen in training.
    pub unseen: usize,
    /// Correct top-1 predictions.
    pub hits: usize,
}

impl ForwardScore {
    /// Top-1 accuracy over scored positions.
    #[must_use]
    pub fn accuracy(&self) -> f64 {
        if self.scored == 0 {
            0.0
        } else {
            f64::from(u32::try_from(self.hits).unwrap_or(u32::MAX))
                / f64::from(u32::try_from(self.scored).unwrap_or(u32::MAX))
        }
    }
}

/// Score a predictor over held-out views.
#[must_use]
pub fn score(pred: &CountPredictor, views: &[TokenStreamView<'_>]) -> ForwardScore {
    let mut s = ForwardScore::default();
    for v in views {
        for (ctx, next) in windows(v, pred.order()) {
            s.scored += 1;
            match pred.predict(ctx) {
                Some(p) if p == next => s.hits += 1,
                Some(_) => {}
                None => s.unseen += 1,
            }
        }
    }
    s
}
