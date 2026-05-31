// Copyright (c) 2026 Vishal Verma. All rights reserved.
// Licensed under the Business Source License 1.1 (BSL 1.1).
// See LICENSE.md in the project root for full license terms.

//! state-harness: The Semantic Boundary Layer for Multi-Agent AI Systems.
//!
//! This crate provides the high-performance Rust compute engine for the
//! `state_harness` Python package. It exposes three core subsystems via PyO3:
//!
//! - **HolographicEngine** (`vsa.rs`): Vector Symbolic Architecture for
//!   storing and monitoring safety-critical invariants using bipolar hypervectors.
//!
//! - **LyapunovMonitor** (`lyapunov.rs`): Discrete-time energy tracker with
//!   circuit breaker semantics for detecting and halting runaway agent loops.
//!
//! - **RGDecimator** (`rg.rs`): Renormalization Group coarse-graining engine
//!   for compressing agent communication histories.
//!
//! The native module is exposed as `state_harness._core` and re-exported
//! through the Python SDK layer at `state_harness`.

use pyo3::prelude::*;

mod lyapunov;
mod rg;
mod vsa;

pub use lyapunov::{
    BudgetExhausted, LyapunovMonitor, MonitorGroup, PermanentFailure, StabilityStatus,
    StabilityViolation, TelemetrySnapshot,
};
pub use rg::{RGDecimator, ScoredMessage};
pub use vsa::{DefaultVsaCore, HolographicEngine, VsaCore};

/// Native extension module for state-harness.
///
/// Registered as `state_harness._core` by maturin and re-exported
/// through the Python `state_harness` package.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ── Core engine classes ────────────────────────────────────────────
    m.add_class::<HolographicEngine>()?;
    m.add_class::<LyapunovMonitor>()?;
    m.add_class::<MonitorGroup>()?;
    m.add_class::<StabilityStatus>()?;
    m.add_class::<TelemetrySnapshot>()?;
    m.add_class::<RGDecimator>()?;
    m.add_class::<ScoredMessage>()?;

    // ── Custom exception classes ───────────────────────────────────────
    m.add(
        "StabilityViolation",
        m.py().get_type::<StabilityViolation>(),
    )?;
    m.add("BudgetExhausted", m.py().get_type::<BudgetExhausted>())?;
    m.add("PermanentFailure", m.py().get_type::<PermanentFailure>())?;

    Ok(())
}
