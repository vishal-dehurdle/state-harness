// Copyright (c) 2026 Vishal Verma. All rights reserved.
// Licensed under the Business Source License 1.1 (BSL 1.1).
// See LICENSE.md in the project root for full license terms.

//! Lyapunov stability monitor and circuit breaker for multi-agent systems.
//!
//! Tracks system stability via a discrete scalar energy function:
//!
//!   V(k) = S(k) + λ · θ(k)
//!
//! Where:
//! - S(k) is the **instantaneous** token spend at step k (NOT cumulative).
//! - θ(k) is the error/retry weight at step k.
//! - λ is a positive coupling constant (default 1.0).
//!
//! The temporal derivative ΔV = V(k) - V(k-1) indicates stability:
//! - ΔV < 0 → system is converging (efficient execution)
//! - ΔV ≥ 0 → system energy is growing (potential runaway)
//!
//! If ΔV ≥ 0 for `window` consecutive steps, the circuit breaker trips
//! with a `StabilityViolation` exception to halt runaway execution.

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;

// ─── Custom Python Exception Classes ───────────────────────────────────────

create_exception!(
    state_harness._core,
    StabilityViolation,
    PyException,
    "Raised when the Lyapunov energy derivative indicates system instability.\n\n\
     This means ΔV ≥ 0 for `window` consecutive steps, indicating the agent \
     execution loop is not converging toward a stable, efficient state."
);

create_exception!(
    state_harness._core,
    BudgetExhausted,
    PyException,
    "Raised when cumulative token spend exceeds the configured budget ceiling.\n\n\
     The execution state is frozen for graceful recovery. Resume requires \
     explicit reset or human intervention."
);

create_exception!(
    state_harness._core,
    PermanentFailure,
    PyException,
    "Raised on unrecoverable errors (schema violations, invalid configurations).\n\n\
     The circuit breaker trips immediately. Post-mortem telemetry is compiled \
     to the exception message before propagation."
);

// ─── Constants ─────────────────────────────────────────────────────────────

/// Default coupling constant for error weighting in the energy function.
pub const DEFAULT_LAMBDA: f64 = 1.0;

/// Default number of consecutive non-negative ΔV steps before tripping.
pub const DEFAULT_WINDOW: usize = 3;

// ─── Stability Status ──────────────────────────────────────────────────────

/// Current stability status of the monitored system.
#[pyclass(name = "StabilityStatus")]
#[derive(Clone, Debug, PartialEq)]
pub enum StabilityStatus {
    /// ΔV < 0: System is converging normally.
    Stable,
    /// ΔV ≥ 0 for some steps (< window): Early warning.
    Warning,
    /// ΔV ≥ 0 for >= window steps: Breaker about to trip.
    Unstable,
    /// Circuit breaker has been tripped due to stability violation or permanent failure.
    Tripped,
    /// State frozen due to budget exhaustion.
    Frozen,
}

#[pymethods]
impl StabilityStatus {
    fn __repr__(&self) -> &'static str {
        match self {
            StabilityStatus::Stable => "StabilityStatus.Stable",
            StabilityStatus::Warning => "StabilityStatus.Warning",
            StabilityStatus::Unstable => "StabilityStatus.Unstable",
            StabilityStatus::Tripped => "StabilityStatus.Tripped",
            StabilityStatus::Frozen => "StabilityStatus.Frozen",
        }
    }

    fn __str__(&self) -> &'static str {
        match self {
            StabilityStatus::Stable => "Stable",
            StabilityStatus::Warning => "Warning",
            StabilityStatus::Unstable => "Unstable",
            StabilityStatus::Tripped => "Tripped",
            StabilityStatus::Frozen => "Frozen",
        }
    }
}

// ─── Snapshot for post-mortem diagnostics ───────────────────────────────────

/// Frozen telemetry snapshot captured before circuit breaker trips.
#[pyclass(name = "TelemetrySnapshot")]
#[derive(Clone, Debug)]
pub struct TelemetrySnapshot {
    #[pyo3(get)]
    pub step_count: usize,
    #[pyo3(get)]
    pub cumulative_tokens: u64,
    #[pyo3(get)]
    pub cumulative_errors: u64,
    #[pyo3(get)]
    pub energy_trajectory: Vec<f64>,
    #[pyo3(get)]
    pub final_status: StabilityStatus,
    #[pyo3(get)]
    pub consecutive_non_negative: usize,
}

#[pymethods]
impl TelemetrySnapshot {
    fn __repr__(&self) -> String {
        format!(
            "TelemetrySnapshot(steps={}, tokens={}, errors={}, status={:?})",
            self.step_count, self.cumulative_tokens, self.cumulative_errors, self.final_status
        )
    }
}

// ─── Lyapunov Monitor ──────────────────────────────────────────────────────

/// Discrete-time Lyapunov stability monitor with circuit breaker semantics.
///
/// Computes V(k) = S(k) + λ·θ(k) using **instantaneous** per-step metrics,
/// allowing ΔV to fluctuate dynamically as the agent transitions between
/// efficient and inefficient execution states.
///
/// Three failure classes:
/// - **Transient** (API timeouts): Adds +1.0 to θ(k) within the safety envelope.
/// - **Permanent** (schema violations): Immediate circuit breaker trip with post-mortem.
/// - **Budget exhaustion** (runaway loops): Freeze-and-snapshot for graceful recovery.
#[pyclass(name = "LyapunovMonitor")]
pub struct LyapunovMonitor {
    lambda: f64,
    window: usize,
    budget_ceiling: Option<u64>,
    energy_history: Vec<f64>,
    cumulative_tokens: u64,
    cumulative_errors: u64,
    consecutive_non_negative: usize,
    tripped: bool,
    frozen: bool,
    step_count: usize,
}

#[pymethods]
impl LyapunovMonitor {
    /// Create a new Lyapunov stability monitor.
    ///
    /// Args:
    ///     lambda_: Coupling constant for error weighting (default: 1.0).
    ///              Higher values make the monitor more sensitive to errors.
    ///     window: Consecutive non-negative ΔV steps before tripping (default: 3).
    ///     budget_ceiling: Optional maximum cumulative token spend.
    #[new]
    #[pyo3(signature = (lambda_=DEFAULT_LAMBDA, window=DEFAULT_WINDOW, budget_ceiling=None))]
    fn new(lambda_: f64, window: usize, budget_ceiling: Option<u64>) -> PyResult<Self> {
        if lambda_ <= 0.0 {
            return Err(PyValueError::new_err(
                "Coupling constant λ must be positive",
            ));
        }
        if window == 0 {
            return Err(PyValueError::new_err("Stability window must be ≥ 1"));
        }
        Ok(Self {
            lambda: lambda_,
            window,
            budget_ceiling,
            energy_history: Vec::new(),
            cumulative_tokens: 0,
            cumulative_errors: 0,
            consecutive_non_negative: 0,
            tripped: false,
            frozen: false,
            step_count: 0,
        })
    }

    /// Record metrics for a single execution step and check stability.
    ///
    /// This is the primary API. Call once per agent turn/tool call with the
    /// **instantaneous** metrics for that step.
    ///
    /// Args:
    ///     tokens_used: Tokens consumed in this step (instantaneous, not cumulative).
    ///     errors: Number of transient errors/retries in this step.
    ///
    /// Returns:
    ///     Current StabilityStatus.
    ///
    /// Raises:
    ///     StabilityViolation: If ΔV ≥ 0 for `window` consecutive steps.
    ///     BudgetExhausted: If cumulative tokens exceed the budget ceiling.
    fn record_step(&mut self, tokens_used: u64, errors: u64) -> PyResult<StabilityStatus> {
        if self.frozen {
            return Ok(StabilityStatus::Frozen);
        }
        if self.tripped {
            return Ok(StabilityStatus::Tripped);
        }

        // Compute instantaneous energy: V(k) = S(k) + λ · θ(k)
        let s_k = tokens_used as f64;
        let theta_k = errors as f64;
        let v_k = s_k + self.lambda * theta_k;

        // Update cumulative counters (for budget tracking and telemetry only)
        self.cumulative_tokens += tokens_used;
        self.cumulative_errors += errors;
        self.step_count += 1;

        // Check budget ceiling (cumulative comparison)
        if let Some(ceiling) = self.budget_ceiling {
            if self.cumulative_tokens >= ceiling {
                self.frozen = true;
                return Err(BudgetExhausted::new_err(format!(
                    "Token budget exhausted: {} / {} ceiling. State frozen for recovery. \
                     Steps: {}, errors: {}, energy: {:?}",
                    self.cumulative_tokens,
                    ceiling,
                    self.step_count,
                    self.cumulative_errors,
                    self.recent_energy(5)
                )));
            }
        }

        // Compute ΔV and track consecutive non-negative derivatives
        if let Some(&v_prev) = self.energy_history.last() {
            let delta_v = v_k - v_prev;
            if delta_v >= 0.0 {
                self.consecutive_non_negative += 1;
            } else {
                self.consecutive_non_negative = 0;
            }
        }

        self.energy_history.push(v_k);

        // Check stability: trip if ΔV ≥ 0 for `window` consecutive steps
        if self.consecutive_non_negative >= self.window {
            self.tripped = true;
            return Err(StabilityViolation::new_err(format!(
                "Lyapunov stability violation: ΔV ≥ 0 for {} consecutive steps (window: {}). \
                 Energy trajectory: {:?}. Total tokens: {}, total errors: {}.",
                self.consecutive_non_negative,
                self.window,
                self.recent_energy(self.window + 2),
                self.cumulative_tokens,
                self.cumulative_errors,
            )));
        }

        if self.consecutive_non_negative > 0 {
            Ok(StabilityStatus::Warning)
        } else {
            Ok(StabilityStatus::Stable)
        }
    }

    /// Report a transient failure (API timeout, rate limit, network error).
    ///
    /// Records a micro-step with 0 tokens and +1.0 error weight.
    /// This feeds into the Lyapunov energy function, incrementing θ(k)
    /// while remaining within the safety envelope bounds.
    fn report_transient_failure(&mut self) -> PyResult<StabilityStatus> {
        self.record_step(0, 1)
    }

    /// Report a permanent failure (schema violation, invalid configuration).
    ///
    /// Immediately trips the circuit breaker. A post-mortem telemetry snapshot
    /// is compiled and embedded in the exception message before propagation.
    ///
    /// Args:
    ///     reason: Human-readable description of the permanent failure.
    fn report_permanent_failure(&mut self, reason: &str) -> PyResult<()> {
        let snapshot = self.compile_snapshot(StabilityStatus::Tripped);
        self.tripped = true;
        Err(PermanentFailure::new_err(format!(
            "Permanent failure: {reason}. Post-mortem: {snapshot:?}"
        )))
    }

    /// Check current stability without recording new metrics.
    fn check_stability(&self) -> StabilityStatus {
        if self.frozen {
            StabilityStatus::Frozen
        } else if self.tripped {
            StabilityStatus::Tripped
        } else if self.consecutive_non_negative >= self.window {
            StabilityStatus::Unstable
        } else if self.consecutive_non_negative > 0 {
            StabilityStatus::Warning
        } else {
            StabilityStatus::Stable
        }
    }

    /// Compile a frozen telemetry snapshot for post-mortem analysis.
    fn snapshot(&self) -> TelemetrySnapshot {
        self.compile_snapshot(self.check_stability())
    }

    /// Get the full energy history V(k) for diagnostics.
    fn get_energy_history(&self) -> Vec<f64> {
        self.energy_history.clone()
    }

    /// Get the current cumulative token spend (for budget tracking).
    fn total_tokens(&self) -> u64 {
        self.cumulative_tokens
    }

    /// Get the total step count.
    fn total_steps(&self) -> usize {
        self.step_count
    }

    /// Get the most recent energy value V(k).
    fn current_energy(&self) -> Option<f64> {
        self.energy_history.last().copied()
    }

    /// Get the current consecutive non-negative ΔV count.
    fn consecutive_warnings(&self) -> usize {
        self.consecutive_non_negative
    }

    /// Check if the circuit breaker has been tripped.
    fn is_tripped(&self) -> bool {
        self.tripped
    }

    /// Check if the state has been frozen due to budget exhaustion.
    fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Reset the monitor to initial state.
    ///
    /// Clears all history, counters, and trip/freeze flags.
    /// Use after resolving the root cause of a stability violation.
    fn reset(&mut self) {
        self.energy_history.clear();
        self.cumulative_tokens = 0;
        self.cumulative_errors = 0;
        self.consecutive_non_negative = 0;
        self.tripped = false;
        self.frozen = false;
        self.step_count = 0;
    }
}

// ─── Internal implementation ───────────────────────────────────────────────

impl LyapunovMonitor {
    /// Get the most recent N energy values for diagnostic output.
    fn recent_energy(&self, n: usize) -> Vec<f64> {
        let start = self.energy_history.len().saturating_sub(n);
        self.energy_history[start..].to_vec()
    }

    /// Compile a complete telemetry snapshot.
    fn compile_snapshot(&self, status: StabilityStatus) -> TelemetrySnapshot {
        TelemetrySnapshot {
            step_count: self.step_count,
            cumulative_tokens: self.cumulative_tokens,
            cumulative_errors: self.cumulative_errors,
            energy_trajectory: self.energy_history.clone(),
            final_status: status,
            consecutive_non_negative: self.consecutive_non_negative,
        }
    }
}

// ─── Hierarchical Monitor Group ────────────────────────────────────────────

/// Hierarchical monitor composition for multi-agent topologies.
///
/// Aggregates multiple child ``LyapunovMonitor`` instances, each governing
/// a sub-agent or sub-graph within a larger multi-agent system. Provides
/// Input-to-State Stability (ISS) guarantees by monitoring aggregate energy
/// across the entire agent topology.
///
/// **Design**: A parent ``MonitorGroup`` can also impose its own budget ceiling
/// on the aggregate token spend of all children, catching runaway multi-agent
/// loops that would be invisible to individual child monitors.
///
/// Example (Python)::
///
///     group = MonitorGroup(budget_ceiling=200_000)
///     group.add_child("planner", lambda_=1.0, window=3, budget_ceiling=80_000)
///     group.add_child("executor", lambda_=1.5, window=5, budget_ceiling=80_000)
///
///     # Record steps on individual sub-agents
///     group.record_step("planner", tokens_used=1500, errors=0)
///     group.record_step("executor", tokens_used=3000, errors=1)
///
///     # Check aggregate stability across the topology
///     assert group.is_all_stable()
///     print(group.aggregate_energy())  # {'planner': 1500.0, 'executor': 4500.0}
#[pyclass(name = "MonitorGroup")]
pub struct MonitorGroup {
    children: Vec<(String, LyapunovMonitor)>,
    /// Optional aggregate budget ceiling across all children.
    aggregate_budget: Option<u64>,
}

#[pymethods]
impl MonitorGroup {
    /// Create a new hierarchical monitor group.
    ///
    /// Args:
    ///     budget_ceiling: Optional aggregate token budget across all child monitors.
    ///                     If total tokens across all children exceed this, the group
    ///                     raises ``BudgetExhausted``.
    #[new]
    #[pyo3(signature = (budget_ceiling=None))]
    fn new(budget_ceiling: Option<u64>) -> Self {
        Self {
            children: Vec::new(),
            aggregate_budget: budget_ceiling,
        }
    }

    /// Register a named child monitor for a sub-agent or sub-graph.
    ///
    /// Args:
    ///     name: Unique identifier for this child monitor.
    ///     lambda_: Coupling constant for this child's energy function.
    ///     window: Stability window for this child.
    ///     budget_ceiling: Optional per-child token budget.
    ///
    /// Raises:
    ///     ValueError: If a child with this name already exists.
    #[pyo3(signature = (name, lambda_=DEFAULT_LAMBDA, window=DEFAULT_WINDOW, budget_ceiling=None))]
    fn add_child(
        &mut self,
        name: &str,
        lambda_: f64,
        window: usize,
        budget_ceiling: Option<u64>,
    ) -> PyResult<()> {
        if self.children.iter().any(|(n, _)| n == name) {
            return Err(PyValueError::new_err(format!(
                "Child monitor '{name}' already exists"
            )));
        }
        let monitor = LyapunovMonitor::new(lambda_, window, budget_ceiling)?;
        self.children.push((name.to_string(), monitor));
        Ok(())
    }

    /// Remove a named child monitor.
    ///
    /// Returns:
    ///     True if the child was found and removed, False otherwise.
    fn remove_child(&mut self, name: &str) -> bool {
        let len_before = self.children.len();
        self.children.retain(|(n, _)| n != name);
        self.children.len() < len_before
    }

    /// Record metrics for a step on a specific child monitor.
    ///
    /// Also checks the aggregate budget ceiling across all children.
    ///
    /// Args:
    ///     name: Name of the child monitor.
    ///     tokens_used: Instantaneous tokens consumed in this step.
    ///     errors: Number of transient errors in this step.
    ///
    /// Returns:
    ///     Current StabilityStatus of the specific child.
    ///
    /// Raises:
    ///     ValueError: If the named child doesn't exist.
    ///     StabilityViolation: If the child's stability threshold is breached.
    ///     BudgetExhausted: If child or aggregate budget is exceeded.
    fn record_step(
        &mut self,
        name: &str,
        tokens_used: u64,
        errors: u64,
    ) -> PyResult<StabilityStatus> {
        let child = self
            .children
            .iter_mut()
            .find(|(n, _)| n == name)
            .map(|(_, m)| m)
            .ok_or_else(|| PyValueError::new_err(format!("No child monitor named '{name}'")))?;

        let status = child.record_step(tokens_used, errors)?;

        // Check aggregate budget
        if let Some(ceiling) = self.aggregate_budget {
            let total: u64 = self.children.iter().map(|(_, m)| m.cumulative_tokens).sum();
            if total >= ceiling {
                return Err(BudgetExhausted::new_err(format!(
                    "Aggregate budget exhausted: {} / {} ceiling across {} children",
                    total,
                    ceiling,
                    self.children.len(),
                )));
            }
        }

        Ok(status)
    }

    /// Report a transient failure on a specific child.
    fn report_transient(&mut self, name: &str) -> PyResult<StabilityStatus> {
        self.record_step(name, 0, 1)
    }

    /// Report a permanent failure on a specific child.
    fn report_permanent(&mut self, name: &str, reason: &str) -> PyResult<()> {
        let child = self
            .children
            .iter_mut()
            .find(|(n, _)| n == name)
            .map(|(_, m)| m)
            .ok_or_else(|| PyValueError::new_err(format!("No child monitor named '{name}'")))?;
        child.report_permanent_failure(reason)
    }

    /// Get the current energy V(k) for each child monitor.
    ///
    /// Returns:
    ///     Dict mapping child name → most recent energy value.
    fn aggregate_energy(&self) -> std::collections::HashMap<String, f64> {
        self.children
            .iter()
            .map(|(name, m)| (name.clone(), m.current_energy().unwrap_or(0.0)))
            .collect()
    }

    /// Get the sum of all children's current energy values.
    ///
    /// This aggregate scalar provides a single ISS metric for the
    /// entire multi-agent topology.
    fn total_energy(&self) -> f64 {
        self.children
            .iter()
            .map(|(_, m)| m.current_energy().unwrap_or(0.0))
            .sum()
    }

    /// Get the aggregate cumulative token spend across all children.
    fn aggregate_tokens(&self) -> u64 {
        self.children.iter().map(|(_, m)| m.cumulative_tokens).sum()
    }

    /// Check if ALL children are in a stable state (ΔV < 0 or no steps yet).
    fn is_all_stable(&self) -> bool {
        self.children
            .iter()
            .all(|(_, m)| matches!(m.check_stability(), StabilityStatus::Stable))
    }

    /// Check if ANY child has tripped its circuit breaker.
    fn is_any_tripped(&self) -> bool {
        self.children.iter().any(|(_, m)| m.tripped)
    }

    /// Check if ANY child has been frozen due to budget exhaustion.
    fn is_any_frozen(&self) -> bool {
        self.children.iter().any(|(_, m)| m.frozen)
    }

    /// Get the stability status of a specific child monitor.
    fn child_status(&self, name: &str) -> PyResult<StabilityStatus> {
        self.children
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, m)| m.check_stability())
            .ok_or_else(|| PyValueError::new_err(format!("No child monitor named '{name}'")))
    }

    /// Get the stability status of all children.
    fn all_statuses(&self) -> std::collections::HashMap<String, StabilityStatus> {
        self.children
            .iter()
            .map(|(name, m)| (name.clone(), m.check_stability()))
            .collect()
    }

    /// Compile telemetry snapshots for all children.
    fn aggregate_snapshots(&self) -> Vec<TelemetrySnapshot> {
        self.children
            .iter()
            .map(|(_, m)| m.compile_snapshot(m.check_stability()))
            .collect()
    }

    /// Get the snapshot for a specific child.
    fn child_snapshot(&self, name: &str) -> PyResult<TelemetrySnapshot> {
        self.children
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, m)| m.compile_snapshot(m.check_stability()))
            .ok_or_else(|| PyValueError::new_err(format!("No child monitor named '{name}'")))
    }

    /// List all registered child monitor names.
    fn children_names(&self) -> Vec<String> {
        self.children.iter().map(|(n, _)| n.clone()).collect()
    }

    /// Get the number of registered child monitors.
    fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Reset all child monitors to their initial state.
    fn reset_all(&mut self) {
        for (_, m) in self.children.iter_mut() {
            m.reset();
        }
    }

    /// Reset a specific child monitor.
    fn reset_child(&mut self, name: &str) -> PyResult<()> {
        self.children
            .iter_mut()
            .find(|(n, _)| n == name)
            .map(|(_, m)| m.reset())
            .ok_or_else(|| PyValueError::new_err(format!("No child monitor named '{name}'")))
    }

    fn __repr__(&self) -> String {
        let statuses: Vec<String> = self
            .children
            .iter()
            .map(|(name, m)| format!("{}={:?}", name, m.check_stability()))
            .collect();
        format!(
            "MonitorGroup(children={}, total_energy={:.1}, statuses=[{}])",
            self.children.len(),
            self.total_energy(),
            statuses.join(", ")
        )
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_monitor() -> LyapunovMonitor {
        LyapunovMonitor::new(1.0, 3, None).unwrap()
    }

    fn make_budgeted_monitor(ceiling: u64) -> LyapunovMonitor {
        LyapunovMonitor::new(1.0, 3, Some(ceiling)).unwrap()
    }

    #[test]
    fn test_stable_decreasing_energy() {
        let mut mon = make_monitor();
        // Decreasing token usage → ΔV < 0 → stable
        assert!(matches!(
            mon.record_step(1000, 0),
            Ok(StabilityStatus::Stable)
        ));
        assert!(matches!(
            mon.record_step(800, 0),
            Ok(StabilityStatus::Stable)
        ));
        assert!(matches!(
            mon.record_step(600, 0),
            Ok(StabilityStatus::Stable)
        ));
        assert!(matches!(
            mon.record_step(400, 0),
            Ok(StabilityStatus::Stable)
        ));
    }

    #[test]
    fn test_warning_on_increasing_energy() {
        let mut mon = make_monitor();
        mon.record_step(100, 0).unwrap(); // baseline
        let status = mon.record_step(200, 0).unwrap(); // ΔV = 100 > 0
        assert_eq!(status, StabilityStatus::Warning);
    }

    #[test]
    fn test_trip_on_consecutive_increase() {
        let mut mon = make_monitor();
        mon.record_step(100, 0).unwrap(); // baseline
        mon.record_step(200, 0).unwrap(); // ΔV > 0, consecutive = 1
        mon.record_step(300, 0).unwrap(); // ΔV > 0, consecutive = 2
        let result = mon.record_step(400, 0); // ΔV > 0, consecutive = 3 → trip!
        assert!(result.is_err());
        assert!(mon.is_tripped());
    }

    #[test]
    fn test_reset_clears_consecutive() {
        let mut mon = make_monitor();
        mon.record_step(100, 0).unwrap();
        mon.record_step(200, 0).unwrap(); // warning
                                          // Drop back down → resets consecutive counter
        let status = mon.record_step(50, 0).unwrap();
        assert_eq!(status, StabilityStatus::Stable);
        assert_eq!(mon.consecutive_warnings(), 0);
    }

    #[test]
    fn test_error_weighting() {
        let mut mon = LyapunovMonitor::new(2.0, 3, None).unwrap();
        mon.record_step(100, 0).unwrap(); // V(0) = 100 + 2*0 = 100
        let status = mon.record_step(50, 30).unwrap(); // V(1) = 50 + 2*30 = 110 > 100 → warning
        assert_eq!(status, StabilityStatus::Warning);
    }

    #[test]
    fn test_budget_exhaustion() {
        let mut mon = make_budgeted_monitor(500);
        mon.record_step(200, 0).unwrap();
        mon.record_step(200, 0).unwrap();
        let result = mon.record_step(200, 0); // cumulative: 600 > 500
        assert!(result.is_err());
        assert!(mon.is_frozen());
    }

    #[test]
    fn test_permanent_failure_trips_immediately() {
        let mut mon = make_monitor();
        mon.record_step(100, 0).unwrap();
        let result = mon.report_permanent_failure("Schema validation failed");
        assert!(result.is_err());
        assert!(mon.is_tripped());
    }

    #[test]
    fn test_transient_failure_increments_theta() {
        let mut mon = make_monitor();
        mon.record_step(100, 0).unwrap(); // V(0) = 100
                                          // Transient failure: record_step(0, 1) → V(1) = 0 + 1.0 = 1.0
                                          // ΔV = 1.0 - 100.0 = -99.0 < 0 → stable (energy dropped)
        let status = mon.report_transient_failure().unwrap();
        assert_eq!(status, StabilityStatus::Stable);
    }

    #[test]
    fn test_frozen_state_rejects_new_steps() {
        let mut mon = make_budgeted_monitor(100);
        let _ = mon.record_step(200, 0); // triggers freeze
        assert!(mon.is_frozen());
        // Subsequent steps should return Frozen without error
        let status = mon.record_step(50, 0).unwrap();
        assert_eq!(status, StabilityStatus::Frozen);
    }

    #[test]
    fn test_tripped_state_rejects_new_steps() {
        let mut mon = make_monitor();
        mon.record_step(100, 0).unwrap();
        mon.record_step(200, 0).unwrap();
        mon.record_step(300, 0).unwrap();
        let _ = mon.record_step(400, 0); // trips
                                         // Subsequent steps should return Tripped
        let status = mon.record_step(50, 0).unwrap();
        assert_eq!(status, StabilityStatus::Tripped);
    }

    #[test]
    fn test_snapshot_compilation() {
        let mut mon = make_monitor();
        mon.record_step(100, 0).unwrap();
        mon.record_step(200, 1).unwrap();
        let snap = mon.snapshot();
        assert_eq!(snap.step_count, 2);
        assert_eq!(snap.cumulative_tokens, 300);
        assert_eq!(snap.cumulative_errors, 1);
        assert_eq!(snap.energy_trajectory.len(), 2);
    }

    #[test]
    fn test_reset_full_clear() {
        let mut mon = make_monitor();
        mon.record_step(100, 1).unwrap();
        mon.record_step(200, 0).unwrap();
        mon.reset();
        assert_eq!(mon.total_tokens(), 0);
        assert_eq!(mon.total_steps(), 0);
        assert!(!mon.is_tripped());
        assert!(!mon.is_frozen());
        assert_eq!(mon.get_energy_history().len(), 0);
    }

    #[test]
    fn test_invalid_lambda() {
        assert!(LyapunovMonitor::new(0.0, 3, None).is_err());
        assert!(LyapunovMonitor::new(-1.0, 3, None).is_err());
    }

    #[test]
    fn test_invalid_window() {
        assert!(LyapunovMonitor::new(1.0, 0, None).is_err());
    }

    // ── MonitorGroup Tests ─────────────────────────────────────────────

    #[test]
    fn test_group_add_and_list_children() {
        let mut group = MonitorGroup::new(None);
        group.add_child("planner", 1.0, 3, None).unwrap();
        group.add_child("executor", 1.5, 5, None).unwrap();
        assert_eq!(group.child_count(), 2);
        assert_eq!(group.children_names(), vec!["planner", "executor"]);
    }

    #[test]
    fn test_group_duplicate_child_rejected() {
        let mut group = MonitorGroup::new(None);
        group.add_child("agent", 1.0, 3, None).unwrap();
        assert!(group.add_child("agent", 1.0, 3, None).is_err());
    }

    #[test]
    fn test_group_record_step_per_child() {
        let mut group = MonitorGroup::new(None);
        group.add_child("a", 1.0, 3, None).unwrap();
        group.add_child("b", 1.0, 3, None).unwrap();

        group.record_step("a", 500, 0).unwrap();
        group.record_step("b", 300, 0).unwrap();

        let energies = group.aggregate_energy();
        assert!((energies["a"] - 500.0).abs() < f64::EPSILON);
        assert!((energies["b"] - 300.0).abs() < f64::EPSILON);
        assert_eq!(group.aggregate_tokens(), 800);
    }

    #[test]
    fn test_group_all_stable() {
        let mut group = MonitorGroup::new(None);
        group.add_child("a", 1.0, 3, None).unwrap();
        group.add_child("b", 1.0, 3, None).unwrap();

        // Decreasing energy → stable
        group.record_step("a", 1000, 0).unwrap();
        group.record_step("a", 500, 0).unwrap();
        group.record_step("b", 800, 0).unwrap();
        group.record_step("b", 400, 0).unwrap();

        assert!(group.is_all_stable());
        assert!(!group.is_any_tripped());
    }

    #[test]
    fn test_group_child_trip_detected() {
        let mut group = MonitorGroup::new(None);
        group.add_child("a", 1.0, 3, None).unwrap();

        group.record_step("a", 100, 0).unwrap();
        group.record_step("a", 200, 0).unwrap();
        group.record_step("a", 300, 0).unwrap();
        let _ = group.record_step("a", 400, 0); // trips

        assert!(group.is_any_tripped());
        assert!(!group.is_all_stable());
    }

    #[test]
    fn test_group_aggregate_budget() {
        let mut group = MonitorGroup::new(Some(1000));
        group.add_child("a", 1.0, 10, None).unwrap();
        group.add_child("b", 1.0, 10, None).unwrap();

        group.record_step("a", 400, 0).unwrap();
        group.record_step("b", 400, 0).unwrap();
        // Aggregate is 800, ceiling is 1000 → OK
        let result = group.record_step("a", 300, 0);
        // Aggregate is now 1100 ≥ 1000 → BudgetExhausted
        assert!(result.is_err());
    }

    #[test]
    fn test_group_remove_child() {
        let mut group = MonitorGroup::new(None);
        group.add_child("temp", 1.0, 3, None).unwrap();
        assert_eq!(group.child_count(), 1);
        assert!(group.remove_child("temp"));
        assert_eq!(group.child_count(), 0);
        assert!(!group.remove_child("nonexistent"));
    }

    #[test]
    fn test_group_reset_all() {
        let mut group = MonitorGroup::new(None);
        group.add_child("a", 1.0, 3, None).unwrap();
        group.record_step("a", 500, 1).unwrap();
        assert!(group.aggregate_tokens() > 0);

        group.reset_all();
        assert_eq!(group.aggregate_tokens(), 0);
    }

    #[test]
    fn test_group_total_energy() {
        let mut group = MonitorGroup::new(None);
        group.add_child("a", 1.0, 3, None).unwrap();
        group.add_child("b", 2.0, 3, None).unwrap();

        group.record_step("a", 100, 0).unwrap(); // V = 100
        group.record_step("b", 200, 1).unwrap(); // V = 200 + 2*1 = 202

        let total = group.total_energy();
        assert!((total - 302.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_group_unknown_child_error() {
        let mut group = MonitorGroup::new(None);
        assert!(group.record_step("ghost", 100, 0).is_err());
    }
}
