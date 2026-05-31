# Copyright (c) 2026 Vishal Verma. All rights reserved.
# Python SDK licensed under Apache License 2.0.
# The Rust compute engine (state_harness._core) is licensed under BSL 1.1.

"""state-harness: The Semantic Boundary Layer for Multi-Agent AI Systems.

Eliminates trajectory drift, context dilution, and runaway token tsunamis
in autonomous multi-agent systems using physics-inspired safety mechanisms:

- **Holographic Invariant Storage (VSA)**: Binds and monitors safety-critical
  invariants outside the LLM context window using high-dimensional bipolar vectors.

- **Lyapunov Circuit Breaker**: Tracks system stability via a discrete energy
  derivative to proactively intercept collapsing execution loops.

- **Renormalization Group Compression**: Condenses conversational history into
  scale-invariant task objectives by filtering high-frequency linguistic noise.
"""

from __future__ import annotations

from state_harness._core import (
    BudgetExhausted,
    HolographicEngine,
    LyapunovMonitor,
    MonitorGroup,
    PermanentFailure,
    RGDecimator,
    ScoredMessage,
    StabilityStatus,
    StabilityViolation,
    TelemetrySnapshot,
)
from state_harness.sdk import (
    BoundaryGuard,
    CoarseGrainer,
    FailureType,
    GrowthRatioConfig,
    GrowthRatioGuard,
    GuardConfig,
    StepMetrics,
    boundary_guard,
)
from state_harness.diagnostics import (
    FailurePattern,
    FailureReport,
    Suggestion,
)

__version__ = "0.2.0"

__all__ = [
    # Core engine classes (from Rust _core)
    "HolographicEngine",
    "LyapunovMonitor",
    "MonitorGroup",
    "StabilityStatus",
    "TelemetrySnapshot",
    "RGDecimator",
    "ScoredMessage",
    # Exceptions (from Rust _core)
    "StabilityViolation",
    "BudgetExhausted",
    "PermanentFailure",
    # SDK (Python layer)
    "BoundaryGuard",
    "GrowthRatioGuard",
    "GrowthRatioConfig",
    "boundary_guard",
    "CoarseGrainer",
    "StepMetrics",
    "GuardConfig",
    "FailureType",
    # Diagnostics
    "FailureReport",
    "FailurePattern",
    "Suggestion",
]
