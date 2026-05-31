// Copyright (c) 2026 Vishal Verma. All rights reserved.
// Licensed under the Business Source License 1.1 (BSL 1.1).
// See LICENSE.md in the project root for full license terms.

//! High-dimensional Vector Symbolic Architecture (VSA) engine for
//! holographic invariant storage using bipolar vectors v ∈ {-1, 1}^D.
//!
//! Implements three core algebraic operations:
//! - **Bind** (⊗): Element-wise multiplication, producing near-orthogonal vectors.
//! - **Bundle** (+): Element-wise summation + signum, creating holographic superpositions.
//! - **Cosine Similarity**: Real-time drift detection via angular distance.
//!
//! Safety invariants are stored as bound pairs H_inv = K_goal ⊗ V_safe outside
//! the LLM context window. Continuous cosine monitoring detects context corruption,
//! and algebraic unbinding recovers clean goal vectors.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

#[cfg(feature = "numpy")]
use numpy::{PyArray1, PyReadonlyArray1};

/// Character n-gram size for text encoding.
const NGRAM_SIZE: usize = 3;

/// Default vector dimensionality for the Holographic Engine.
pub const DEFAULT_DIMENSIONALITY: usize = 10_000;

/// Maximum number of simultaneously registered invariants.
/// Beyond this limit, algebraic recovery fidelity degrades below useful thresholds.
const MAX_INVARIANTS: usize = 20;

/// Warning threshold for invariant count.
/// At K > 5, recovery fidelity drops to ≈ 1/(K+1) ≈ 16.7%.
const WARN_INVARIANTS: usize = 5;

/// High-dimensional Vector Symbolic Architecture engine.
///
/// Manages bipolar hypervectors of configurable dimensionality D (default 10,000)
/// using i8 storage for cache-efficient SIMD-friendly operations.
///
/// Stores named safety invariants as bound vector pairs on the Rust heap,
/// minimizing FFI boundary crossings for real-time drift detection.
#[pyclass(name = "HolographicEngine")]
pub struct HolographicEngine {
    dim: usize,
    invariants: HashMap<String, Vec<i8>>,
    /// Key vectors stored separately for recovery (unbinding).
    keys: HashMap<String, Vec<i8>>,
    rng: SmallRng,
}

#[pymethods]
impl HolographicEngine {
    /// Create a new HolographicEngine with the specified dimensionality.
    ///
    /// Args:
    ///     dim: Vector dimensionality (default: 10,000). Higher dimensions
    ///          improve pseudo-orthogonality but increase memory usage.
    #[new]
    #[pyo3(signature = (dim=DEFAULT_DIMENSIONALITY))]
    fn new(dim: usize) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("Dimensionality must be > 0"));
        }
        Ok(Self {
            dim,
            invariants: HashMap::new(),
            keys: HashMap::new(),
            rng: SmallRng::from_entropy(),
        })
    }

    /// Generate a random bipolar vector with elements uniformly in {-1, 1}.
    ///
    /// Returns:
    ///     A list of `dim` elements, each -1 or 1.
    fn generate_random_vector(&mut self) -> Vec<i8> {
        (0..self.dim)
            .map(|_| if self.rng.gen_bool(0.5) { 1i8 } else { -1i8 })
            .collect()
    }

    /// Bind two bipolar vectors via element-wise multiplication.
    ///
    /// This operation is its own inverse: bind(a, bind(a, b)) ≈ b.
    /// The result is near-orthogonal to both inputs, enabling role-filler encoding.
    ///
    /// Args:
    ///     a: First bipolar vector.
    ///     b: Second bipolar vector.
    ///
    /// Returns:
    ///     Element-wise product a ⊗ b.
    fn bind(&self, py: Python<'_>, a: Vec<i8>, b: Vec<i8>) -> PyResult<Vec<i8>> {
        self.validate_vec(&a, "a")?;
        self.validate_vec(&b, "b")?;
        Ok(py.allow_threads(|| Self::bind_internal(&a, &b)))
    }

    /// Bundle multiple bipolar vectors via element-wise sum + signum projection.
    ///
    /// Creates a holographic superposition that maintains maximum cosine
    /// similarity to all constituent vectors. Ties (zero sums) project to +1.
    ///
    /// Args:
    ///     vectors: List of bipolar vectors to bundle.
    ///
    /// Returns:
    ///     Bundled superposition vector projected back to {-1, 1}.
    fn bundle(&self, py: Python<'_>, vectors: Vec<Vec<i8>>) -> PyResult<Vec<i8>> {
        if vectors.is_empty() {
            return Err(PyValueError::new_err("Cannot bundle empty vector list"));
        }
        for (i, v) in vectors.iter().enumerate() {
            self.validate_vec(v, &format!("vectors[{i}]"))?;
        }
        let dim = self.dim;
        Ok(py.allow_threads(move || Self::bundle_internal(&vectors, dim)))
    }

    /// Compute cosine similarity between two bipolar vectors.
    ///
    /// For bipolar vectors where ||v|| = √D, this simplifies to dot(a,b) / D.
    /// Random vectors concentrate around 0.0 at high D; meaningful similarity
    /// appears as significant deviation from this baseline.
    ///
    /// Args:
    ///     a: First bipolar vector.
    ///     b: Second bipolar vector.
    ///
    /// Returns:
    ///     Cosine similarity in [-1.0, 1.0].
    fn cosine_similarity(&self, py: Python<'_>, a: Vec<i8>, b: Vec<i8>) -> PyResult<f64> {
        self.validate_vec(&a, "a")?;
        self.validate_vec(&b, "b")?;
        Ok(py.allow_threads(|| Self::cosine_internal(&a, &b)))
    }

    /// Register a named safety invariant by binding a key vector to a value vector.
    ///
    /// The bound result H_inv = K ⊗ V is stored on the Rust heap for continuous
    /// drift detection. The key vector is retained for later recovery via unbinding.
    ///
    /// Warns when the invariant count exceeds 5 (recovery fidelity ≈ 16.7%).
    /// Hard-caps at 20 invariants to prevent algebraic breakdown.
    ///
    /// Args:
    ///     name: Unique identifier for this invariant.
    ///     key: Key vector (role identifier).
    ///     value: Value vector (safety constraint encoding).
    fn register_invariant(
        &mut self,
        name: String,
        key: Vec<i8>,
        value: Vec<i8>,
    ) -> PyResult<()> {
        self.validate_vec(&key, "key")?;
        self.validate_vec(&value, "value")?;

        if !self.invariants.contains_key(&name) && self.invariants.len() >= MAX_INVARIANTS {
            return Err(PyValueError::new_err(format!(
                "Cannot register more than {MAX_INVARIANTS} invariants. \
                 Algebraic recovery degrades as ≈1/(K+1) beyond this limit.",
            )));
        }

        if self.invariants.len() >= WARN_INVARIANTS && !self.invariants.contains_key(&name) {
            eprintln!(
                "[state-harness WARNING] Registering invariant '{}' (count: {}). \
                 Recovery fidelity degrades as ≈1/(K+1). Consider reducing active invariants.",
                name,
                self.invariants.len() + 1
            );
        }

        let bound = Self::bind_internal(&key, &value);
        self.invariants.insert(name.clone(), bound);
        self.keys.insert(name, key);
        Ok(())
    }

    /// Remove a registered invariant by name.
    ///
    /// Args:
    ///     name: The invariant identifier to remove.
    ///
    /// Returns:
    ///     True if the invariant existed and was removed, False otherwise.
    fn remove_invariant(&mut self, name: &str) -> bool {
        let removed = self.invariants.remove(name).is_some();
        self.keys.remove(name);
        removed
    }

    /// Check semantic drift by computing cosine similarity between
    /// the current context vector and a stored invariant.
    ///
    /// Values near 0.0 indicate severe drift from the invariant.
    /// Values near 1.0 indicate strong alignment with the stored constraint.
    ///
    /// Args:
    ///     name: Name of the stored invariant.
    ///     context_vector: Current running context as a bipolar vector.
    ///
    /// Returns:
    ///     Cosine similarity in [-1.0, 1.0].
    fn check_drift(&self, name: &str, context_vector: Vec<i8>) -> PyResult<f64> {
        self.validate_vec(&context_vector, "context_vector")?;
        let invariant = self
            .invariants
            .get(name)
            .ok_or_else(|| PyValueError::new_err(format!("No invariant registered as '{name}'")))?;
        Ok(Self::cosine_internal(invariant, &context_vector))
    }

    /// Recover the original value vector from a stored invariant by unbinding.
    ///
    /// Exploits the self-inverse property of bipolar binding:
    /// K ⊗ (K ⊗ V) = V (exact recovery for single-signal invariants).
    ///
    /// Args:
    ///     name: Name of the stored invariant to recover from.
    ///
    /// Returns:
    ///     Recovered value vector (clean, uncorrupted goal state).
    fn recover(&self, name: &str) -> PyResult<Vec<i8>> {
        let invariant = self
            .invariants
            .get(name)
            .ok_or_else(|| PyValueError::new_err(format!("No invariant registered as '{name}'")))?;
        let key = self
            .keys
            .get(name)
            .ok_or_else(|| PyValueError::new_err(format!("No key stored for invariant '{name}'")))?;
        Ok(Self::bind_internal(invariant, key))
    }

    /// Return the number of currently registered invariants.
    fn invariant_count(&self) -> usize {
        self.invariants.len()
    }

    /// Return the configured dimensionality.
    fn dimensionality(&self) -> usize {
        self.dim
    }

    /// List all registered invariant names.
    fn list_invariants(&self) -> Vec<String> {
        self.invariants.keys().cloned().collect()
    }

    /// Batch cosine similarity: compute similarity of one vector against many.
    ///
    /// This amortizes FFI crossing overhead compared to calling cosine_similarity
    /// in a Python loop.
    ///
    /// Args:
    ///     target: The reference vector.
    ///     candidates: List of vectors to compare against.
    ///
    /// Returns:
    ///     List of cosine similarity values, one per candidate.
    fn batch_cosine_similarity(
        &self,
        py: Python<'_>,
        target: Vec<i8>,
        candidates: Vec<Vec<i8>>,
    ) -> PyResult<Vec<f64>> {
        self.validate_vec(&target, "target")?;
        for (i, c) in candidates.iter().enumerate() {
            self.validate_vec(c, &format!("candidates[{i}]"))?;
        }
        Ok(py.allow_threads(|| {
            candidates
                .iter()
                .map(|c| Self::cosine_internal(&target, c))
                .collect()
        }))
    }

    /// Check drift against all registered invariants at once.
    ///
    /// Returns:
    ///     Dict mapping invariant name → cosine similarity.
    fn check_all_drift(
        &self,
        context_vector: Vec<i8>,
    ) -> PyResult<HashMap<String, f64>> {
        self.validate_vec(&context_vector, "context_vector")?;
        Ok(self
            .invariants
            .iter()
            .map(|(name, inv)| (name.clone(), Self::cosine_internal(inv, &context_vector)))
            .collect())
    }

    /// Encode a text string into a bipolar hypervector via character n-gram hashing.
    ///
    /// Each character n-gram is deterministically hashed to seed a random bipolar
    /// vector. All n-gram vectors are bundled (superposed) via element-wise summation
    /// and signum projection, producing a single holographic representation of the
    /// text's distributional character statistics.
    ///
    /// This enables the RG coarse-graining engine to use VSA cosine similarity
    /// for semantic redundancy detection across agent messages.
    ///
    /// Args:
    ///     text: The text string to encode.
    ///
    /// Returns:
    ///     A bipolar vector of dimensionality `dim` representing the text.
    fn encode_text(&self, py: Python<'_>, text: &str) -> Vec<i8> {
        let dim = self.dim;
        py.allow_threads(move || Self::encode_text_internal(text, dim))
    }

    /// Batch-encode multiple text strings into bipolar hypervectors.
    ///
    /// Amortizes FFI crossing overhead compared to calling `encode_text` in a
    /// Python loop.
    ///
    /// Args:
    ///     texts: List of text strings to encode.
    ///
    /// Returns:
    ///     List of bipolar vectors, one per input text.
    fn batch_encode_texts(&self, py: Python<'_>, texts: Vec<String>) -> Vec<Vec<i8>> {
        let dim = self.dim;
        py.allow_threads(move || {
            texts
                .iter()
                .map(|t| Self::encode_text_internal(t, dim))
                .collect()
        })
    }
}

// ─── numpy zero-copy methods (feature-gated) ───────────────────────────────

#[cfg(feature = "numpy")]
#[pymethods]
impl HolographicEngine {
    /// Bind two bipolar vectors using zero-copy numpy arrays.
    ///
    /// Avoids copying 10K-element arrays across the FFI boundary by reading
    /// directly from the numpy array's memory buffer.
    ///
    /// Args:
    ///     a: First bipolar vector as a numpy int8 array.
    ///     b: Second bipolar vector as a numpy int8 array.
    ///
    /// Returns:
    ///     Element-wise product a ⊗ b as a numpy int8 array.
    fn bind_numpy<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray1<'py, i8>,
        b: PyReadonlyArray1<'py, i8>,
    ) -> PyResult<Bound<'py, PyArray1<i8>>> {
        let a_slice = a.as_slice().map_err(|e| {
            PyValueError::new_err(format!("Array 'a' is not contiguous: {e}"))
        })?;
        let b_slice = b.as_slice().map_err(|e| {
            PyValueError::new_err(format!("Array 'b' is not contiguous: {e}"))
        })?;
        self.validate_vec(a_slice, "a")?;
        self.validate_vec(b_slice, "b")?;
        let result = py.allow_threads(|| Self::bind_internal(a_slice, b_slice));
        Ok(PyArray1::from_vec_bound(py, result))
    }

    /// Cosine similarity using zero-copy numpy arrays.
    fn cosine_similarity_numpy<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray1<'py, i8>,
        b: PyReadonlyArray1<'py, i8>,
    ) -> PyResult<f64> {
        let a_slice = a.as_slice().map_err(|e| {
            PyValueError::new_err(format!("Array 'a' is not contiguous: {e}"))
        })?;
        let b_slice = b.as_slice().map_err(|e| {
            PyValueError::new_err(format!("Array 'b' is not contiguous: {e}"))
        })?;
        self.validate_vec(a_slice, "a")?;
        self.validate_vec(b_slice, "b")?;
        Ok(py.allow_threads(|| Self::cosine_internal(a_slice, b_slice)))
    }

    /// Batch cosine similarity using a zero-copy numpy target vector.
    fn batch_cosine_similarity_numpy<'py>(
        &self,
        py: Python<'py>,
        target: PyReadonlyArray1<'py, i8>,
        candidates: Vec<Vec<i8>>,
    ) -> PyResult<Vec<f64>> {
        let target_slice = target.as_slice().map_err(|e| {
            PyValueError::new_err(format!("Array 'target' is not contiguous: {e}"))
        })?;
        self.validate_vec(target_slice, "target")?;
        for (i, c) in candidates.iter().enumerate() {
            self.validate_vec(c, &format!("candidates[{i}]"))?;
        }
        Ok(py.allow_threads(|| {
            candidates
                .iter()
                .map(|c| Self::cosine_internal(target_slice, c))
                .collect()
        }))
    }

    /// Register a safety invariant using zero-copy numpy arrays for key and value.
    fn register_invariant_numpy<'py>(
        &mut self,
        name: &str,
        key_vector: PyReadonlyArray1<'py, i8>,
        value_vector: PyReadonlyArray1<'py, i8>,
    ) -> PyResult<()> {
        let key_slice = key_vector.as_slice().map_err(|e| {
            PyValueError::new_err(format!("key_vector is not contiguous: {e}"))
        })?;
        let val_slice = value_vector.as_slice().map_err(|e| {
            PyValueError::new_err(format!("value_vector is not contiguous: {e}"))
        })?;
        self.validate_vec(key_slice, "key_vector")?;
        self.validate_vec(val_slice, "value_vector")?;
        // Convert slices to owned Vecs for storage
        self.register_invariant(name.to_string(), key_slice.to_vec(), val_slice.to_vec())
    }

    /// Check drift using a zero-copy numpy context vector.
    fn check_drift_numpy<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        context_vector: PyReadonlyArray1<'py, i8>,
    ) -> PyResult<f64> {
        let ctx_slice = context_vector.as_slice().map_err(|e| {
            PyValueError::new_err(format!("context_vector is not contiguous: {e}"))
        })?;
        self.validate_vec(ctx_slice, "context_vector")?;
        let inv = self.invariants.get(name).ok_or_else(|| {
            PyValueError::new_err(format!("No invariant named '{name}'"))
        })?;
        let inv_clone = inv.clone();
        Ok(py.allow_threads(move || Self::cosine_internal(&inv_clone, ctx_slice)))
    }
}

// ─── Internal implementation (not exposed to Python) ───────────────────────

impl HolographicEngine {
    /// Validate that a vector has the correct dimensionality and bipolar values.
    fn validate_vec(&self, v: &[i8], name: &str) -> PyResult<()> {
        if v.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "Vector '{name}' has dimension {} but engine requires {dim}",
                v.len(),
                dim = self.dim
            )));
        }
        for (i, &val) in v.iter().enumerate() {
            if val != -1 && val != 1 {
                return Err(PyValueError::new_err(format!(
                    "Vector '{name}' element [{i}] = {val} is not bipolar (must be -1 or 1)"
                )));
            }
        }
        Ok(())
    }

    /// Element-wise multiplication of two bipolar vectors (binding).
    ///
    /// For bipolar {-1, 1} values, multiplication is equivalent to XNOR
    /// and the operation is self-inverse: bind(a, bind(a, b)) == b.
    #[inline]
    fn bind_internal(a: &[i8], b: &[i8]) -> Vec<i8> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
    }

    /// Element-wise summation followed by signum projection to {-1, 1}.
    ///
    /// Ties (sum == 0) project to +1 by convention, maintaining the bipolar domain.
    /// Uses i32 accumulator to prevent overflow when bundling many vectors.
    fn bundle_internal(vectors: &[Vec<i8>], dim: usize) -> Vec<i8> {
        let mut accum = vec![0i32; dim];
        for v in vectors {
            for (acc, &val) in accum.iter_mut().zip(v.iter()) {
                *acc += val as i32;
            }
        }
        accum
            .iter()
            .map(|&sum| if sum >= 0 { 1i8 } else { -1i8 })
            .collect()
    }

    /// Cosine similarity optimized for bipolar vectors.
    ///
    /// For bipolar vectors, ||v|| = √D for all v, so:
    ///   cos(a, b) = dot(a, b) / (||a|| · ||b||) = dot(a, b) / D
    ///
    /// This avoids the sqrt computation entirely.
    ///
    /// Exposed as `pub(crate)` so the RG decimator can use it for
    /// VSA-based semantic redundancy detection.
    #[inline]
    pub(crate) fn cosine_internal(a: &[i8], b: &[i8]) -> f64 {
        let dot: i64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x as i64) * (y as i64))
            .sum();
        dot as f64 / a.len() as f64
    }

    /// Encode text to a bipolar vector using character n-gram random projection.
    ///
    /// Algorithm:
    /// 1. Extract overlapping character n-grams (trigrams by default).
    /// 2. Hash each n-gram to a deterministic u64 seed (FNV-1a).
    /// 3. Use each seed to generate a bipolar vector via `SmallRng`.
    /// 4. Bundle (superpose) all n-gram vectors via summation + signum.
    ///
    /// For very short texts (< NGRAM_SIZE characters), falls back to
    /// a hash of the entire string.
    pub(crate) fn encode_text_internal(text: &str, dim: usize) -> Vec<i8> {
        let lower = text.to_lowercase();
        let chars: Vec<char> = lower.chars().collect();

        if chars.len() < NGRAM_SIZE {
            // Fallback: hash the entire short string
            let seed = Self::hash_ngram(&lower);
            let mut rng = SmallRng::seed_from_u64(seed);
            return (0..dim)
                .map(|_| if rng.gen_bool(0.5) { 1i8 } else { -1i8 })
                .collect();
        }

        let mut accum = vec![0i32; dim];

        for window in chars.windows(NGRAM_SIZE) {
            let ngram: String = window.iter().collect();
            let seed = Self::hash_ngram(&ngram);
            let mut rng = SmallRng::seed_from_u64(seed);
            for acc in accum.iter_mut() {
                *acc += if rng.gen_bool(0.5) { 1 } else { -1 };
            }
        }

        // Signum projection back to bipolar domain
        accum
            .iter()
            .map(|&sum| if sum >= 0 { 1i8 } else { -1i8 })
            .collect()
    }

    /// FNV-1a hash of an n-gram string, producing a deterministic u64 seed.
    #[inline]
    fn hash_ngram(ngram: &str) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        for byte in ngram.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }
}

// ─── Const-Generic VSA Core (Pure Rust API) ────────────────────────────────

/// Compile-time dimensionality-specialized VSA engine for pure Rust usage.
///
/// When `D` is known at compile time, the Rust compiler can:
/// - Unroll inner loops for SIMD auto-vectorization
/// - Emit tighter bounds-checking code (often elided entirely)
/// - Optimize the `D`-based cosine denominator as a constant divisor
///
/// **PyO3 constraint**: `#[pyclass]` does not support const generics, so the
/// Python-facing [`HolographicEngine`] uses runtime dimensionality. Use
/// `VsaCore<D>` directly from Rust for maximum performance.
///
/// Default: `VsaCore<10_000>` (alias: [`DefaultVsaCore`]).
///
/// # Example
///
/// ```ignore
/// use state_harness::VsaCore;
///
/// let mut engine = VsaCore::<10_000>::new();
/// let key = engine.generate_random_vector();
/// let val = engine.generate_random_vector();
/// let bound = VsaCore::<10_000>::bind(&key, &val);
/// let similarity = VsaCore::<10_000>::cosine(&key, &key);
/// assert!((similarity - 1.0).abs() < f64::EPSILON);
/// ```
pub struct VsaCore<const D: usize = 10_000> {
    invariants: HashMap<String, Vec<i8>>,
    keys: HashMap<String, Vec<i8>>,
    rng: SmallRng,
}

/// Type alias for the default-dimensionality VSA core (D = 10,000).
pub type DefaultVsaCore = VsaCore<{ DEFAULT_DIMENSIONALITY }>;

impl<const D: usize> VsaCore<D> {
    /// Create a new const-generic VSA core.
    pub fn new() -> Self {
        Self {
            invariants: HashMap::new(),
            keys: HashMap::new(),
            rng: SmallRng::from_entropy(),
        }
    }

    /// Returns the compile-time dimensionality.
    #[inline(always)]
    pub const fn dim() -> usize {
        D
    }

    /// Generate a random bipolar vector of dimensionality D.
    pub fn generate_random_vector(&mut self) -> Vec<i8> {
        (0..D)
            .map(|_| if self.rng.gen_bool(0.5) { 1i8 } else { -1i8 })
            .collect()
    }

    /// Element-wise multiplication (binding) of two bipolar vectors.
    ///
    /// The compiler knows the loop bound at compile time, enabling
    /// SIMD auto-vectorization and potential loop unrolling.
    #[inline(always)]
    pub fn bind(a: &[i8], b: &[i8]) -> Vec<i8> {
        debug_assert_eq!(a.len(), D, "bind: vector 'a' has wrong dimension");
        debug_assert_eq!(b.len(), D, "bind: vector 'b' has wrong dimension");
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
    }

    /// Element-wise summation + signum projection (bundling).
    #[inline(always)]
    pub fn bundle(vectors: &[Vec<i8>]) -> Vec<i8> {
        let mut accum = vec![0i32; D];
        for v in vectors {
            debug_assert_eq!(v.len(), D, "bundle: vector has wrong dimension");
            for (acc, &val) in accum.iter_mut().zip(v.iter()) {
                *acc += val as i32;
            }
        }
        accum
            .iter()
            .map(|&sum| if sum >= 0 { 1i8 } else { -1i8 })
            .collect()
    }

    /// Cosine similarity optimized for bipolar vectors.
    ///
    /// For bipolar vectors, `cos(a, b) = dot(a, b) / D`. The denominator `D`
    /// is a compile-time constant, allowing the compiler to emit a
    /// multiply-by-reciprocal instead of a runtime division.
    #[inline(always)]
    pub fn cosine(a: &[i8], b: &[i8]) -> f64 {
        debug_assert_eq!(a.len(), D, "cosine: vector 'a' has wrong dimension");
        debug_assert_eq!(b.len(), D, "cosine: vector 'b' has wrong dimension");
        let dot: i64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x as i64) * (y as i64))
            .sum();
        dot as f64 / D as f64
    }

    /// Encode text to a bipolar vector using character n-gram hashing.
    pub fn encode_text(text: &str) -> Vec<i8> {
        HolographicEngine::encode_text_internal(text, D)
    }

    /// Register a named invariant as a bound key-value pair.
    pub fn register_invariant(
        &mut self,
        name: &str,
        key: Vec<i8>,
        value: Vec<i8>,
    ) -> Result<(), String> {
        if self.invariants.len() >= MAX_INVARIANTS {
            return Err(format!(
                "Maximum invariant capacity ({MAX_INVARIANTS}) reached"
            ));
        }
        if key.len() != D || value.len() != D {
            return Err(format!("Vectors must have dimension {D}"));
        }
        let bound = Self::bind(&key, &value);
        self.invariants.insert(name.to_string(), bound);
        self.keys.insert(name.to_string(), key);
        Ok(())
    }

    /// Check drift by computing cosine similarity between a stored
    /// invariant and a context vector.
    pub fn check_drift(&self, name: &str, context: &[i8]) -> Option<f64> {
        self.invariants
            .get(name)
            .map(|inv| Self::cosine(inv, context))
    }

    /// Recover (unbind) the value vector from a stored invariant.
    pub fn recover(&self, name: &str) -> Option<Vec<i8>> {
        let inv = self.invariants.get(name)?;
        let key = self.keys.get(name)?;
        Some(Self::bind(inv, key))
    }

    /// Number of registered invariants.
    pub fn invariant_count(&self) -> usize {
        self.invariants.len()
    }
}

impl<const D: usize> Default for VsaCore<D> {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_engine(dim: usize) -> HolographicEngine {
        HolographicEngine {
            dim,
            invariants: HashMap::new(),
            keys: HashMap::new(),
            rng: SmallRng::seed_from_u64(42),
        }
    }

    #[test]
    fn test_generate_random_vector_dimension() {
        let mut engine = make_engine(100);
        let v = engine.generate_random_vector();
        assert_eq!(v.len(), 100);
        assert!(v.iter().all(|&x| x == -1 || x == 1));
    }

    #[test]
    fn test_bind_self_inverse_property() {
        let mut engine = make_engine(1000);
        let a = engine.generate_random_vector();
        let b = engine.generate_random_vector();

        let bound = HolographicEngine::bind_internal(&a, &b);
        let recovered = HolographicEngine::bind_internal(&a, &bound);

        // For bipolar vectors, bind is exactly self-inverse
        assert_eq!(recovered, b);
    }

    #[test]
    fn test_cosine_identical_vectors() {
        let _engine = make_engine(100);
        let v = vec![1i8; 100];
        let sim = HolographicEngine::cosine_internal(&v, &v);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_opposite_vectors() {
        let v1 = vec![1i8; 100];
        let v2 = vec![-1i8; 100];
        let sim = HolographicEngine::cosine_internal(&v1, &v2);
        assert!((sim + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_random_near_zero() {
        let mut engine = make_engine(10_000);
        let a = engine.generate_random_vector();
        let b = engine.generate_random_vector();
        let sim = HolographicEngine::cosine_internal(&a, &b);
        // Random 10K-dim bipolar vectors are pseudo-orthogonal
        assert!(sim.abs() < 0.1, "Expected near-zero similarity, got {sim}");
    }

    #[test]
    fn test_bundle_majority_vote() {
        let dim = 100;
        let v1 = vec![1i8; dim];
        let v2 = vec![1i8; dim];
        let v3 = vec![-1i8; dim];
        let bundled = HolographicEngine::bundle_internal(&[v1, v2, v3], dim);
        // Majority is +1 for all elements
        assert!(bundled.iter().all(|&x| x == 1));
    }

    #[test]
    fn test_register_and_recover_invariant() {
        let mut engine = make_engine(1000);
        let key = engine.generate_random_vector();
        let value = engine.generate_random_vector();
        let original_value = value.clone();

        engine
            .register_invariant("test".to_string(), key, value)
            .unwrap();

        let recovered = engine.recover("test").unwrap();
        assert_eq!(recovered, original_value);
    }

    #[test]
    fn test_invariant_hard_cap() {
        let mut engine = make_engine(100);
        for i in 0..MAX_INVARIANTS {
            let k = engine.generate_random_vector();
            let v = engine.generate_random_vector();
            engine
                .register_invariant(format!("inv_{i}"), k, v)
                .unwrap();
        }
        let k = engine.generate_random_vector();
        let v = engine.generate_random_vector();
        let result = engine.register_invariant("overflow".to_string(), k, v);
        assert!(result.is_err());
    }

    #[test]
    fn test_check_drift_aligned() {
        let mut engine = make_engine(1000);
        let key = engine.generate_random_vector();
        let value = engine.generate_random_vector();
        let invariant = HolographicEngine::bind_internal(&key, &value);

        engine
            .register_invariant("goal".to_string(), key, value)
            .unwrap();

        // Context identical to invariant should have similarity 1.0
        let drift = engine.check_drift("goal", invariant).unwrap();
        assert!((drift - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_remove_invariant() {
        let mut engine = make_engine(100);
        let k = engine.generate_random_vector();
        let v = engine.generate_random_vector();
        engine
            .register_invariant("temp".to_string(), k, v)
            .unwrap();
        assert_eq!(engine.invariant_count(), 1);
        assert!(engine.remove_invariant("temp"));
        assert_eq!(engine.invariant_count(), 0);
        assert!(!engine.remove_invariant("temp"));
    }

    #[test]
    fn test_validation_wrong_dimension() {
        let engine = make_engine(100);
        let a = vec![1i8; 100];
        let b = vec![1i8; 50]; // wrong dimension
        assert!(engine.validate_vec(&a, "a").is_ok());
        assert!(engine.validate_vec(&b, "b").is_err());
    }

    #[test]
    fn test_validation_non_bipolar() {
        let engine = make_engine(3);
        let a = vec![1i8, -1, 1];
        let b = vec![1i8, 0, 1]; // 0 is not bipolar
        assert!(engine.validate_vec(&a, "a").is_ok());
        assert!(engine.validate_vec(&b, "b").is_err());
    }

    #[test]
    fn test_encode_text_deterministic() {
        // Same text always produces the same vector
        let v1 = HolographicEngine::encode_text_internal("hello world", 100);
        let v2 = HolographicEngine::encode_text_internal("hello world", 100);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_encode_text_bipolar() {
        // Output is always bipolar {-1, 1}
        let v = HolographicEngine::encode_text_internal("some test message", 100);
        assert_eq!(v.len(), 100);
        for &x in &v {
            assert!(x == 1 || x == -1, "Non-bipolar element: {x}");
        }
    }

    #[test]
    fn test_encode_text_similar_texts_correlated() {
        // Similar texts should have higher cosine similarity than dissimilar texts
        let v_a = HolographicEngine::encode_text_internal("the quick brown fox jumps", 1000);
        let v_b = HolographicEngine::encode_text_internal("the quick brown fox leaps", 1000);
        let v_c = HolographicEngine::encode_text_internal("database connection error timeout", 1000);

        let sim_ab = HolographicEngine::cosine_internal(&v_a, &v_b);
        let sim_ac = HolographicEngine::cosine_internal(&v_a, &v_c);

        // Similar texts (a,b) should be more similar than dissimilar texts (a,c)
        assert!(
            sim_ab > sim_ac,
            "Expected similar texts to be more correlated: sim_ab={sim_ab:.3} vs sim_ac={sim_ac:.3}"
        );
    }

    #[test]
    fn test_encode_text_short_fallback() {
        // Very short text (< NGRAM_SIZE) should still produce valid bipolar vector
        let v = HolographicEngine::encode_text_internal("ab", 100);
        assert_eq!(v.len(), 100);
        for &x in &v {
            assert!(x == 1 || x == -1);
        }
    }

    // ── VsaCore<const D> Tests ─────────────────────────────────────────

    #[test]
    fn test_vsacore_const_dim() {
        assert_eq!(VsaCore::<500>::dim(), 500);
        assert_eq!(VsaCore::<10_000>::dim(), 10_000);
        assert_eq!(DefaultVsaCore::dim(), 10_000);
    }

    #[test]
    fn test_vsacore_bind_self_inverse() {
        let mut core = VsaCore::<100>::new();
        let a = core.generate_random_vector();
        let b = core.generate_random_vector();
        let bound = VsaCore::<100>::bind(&a, &b);
        let recovered = VsaCore::<100>::bind(&bound, &a);
        assert_eq!(recovered, b, "bind(bind(a,b), a) must recover b");
    }

    #[test]
    fn test_vsacore_cosine_identical() {
        let v = vec![1i8; 200];
        let sim = VsaCore::<200>::cosine(&v, &v);
        assert!((sim - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vsacore_cosine_opposite() {
        let a = vec![1i8; 200];
        let b = vec![-1i8; 200];
        let sim = VsaCore::<200>::cosine(&a, &b);
        assert!((sim - (-1.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vsacore_bundle() {
        let v1 = vec![1i8, 1, 1, -1, -1];
        let v2 = vec![1, -1, 1, -1, 1];
        let v3 = vec![1, 1, -1, -1, -1];
        let bundled = VsaCore::<5>::bundle(&[v1, v2, v3]);
        assert_eq!(bundled, vec![1, 1, 1, -1, -1]); // majority vote
    }

    #[test]
    fn test_vsacore_register_and_recover() {
        let mut core = VsaCore::<100>::new();
        let key = core.generate_random_vector();
        let val = core.generate_random_vector();
        core.register_invariant("test", key.clone(), val.clone()).unwrap();
        let recovered = core.recover("test").unwrap();
        assert_eq!(recovered, val);
    }

    #[test]
    fn test_vsacore_encode_text() {
        let v = VsaCore::<500>::encode_text("hello world");
        assert_eq!(v.len(), 500);
        for &x in &v {
            assert!(x == 1 || x == -1);
        }
    }

    #[test]
    fn test_vsacore_default_trait() {
        let core: VsaCore<100> = VsaCore::default();
        assert_eq!(core.invariant_count(), 0);
    }
}
