// Copyright (c) 2026 Vishal Verma. All rights reserved.
// Licensed under the Business Source License 1.1 (BSL 1.1).
// See LICENSE.md in the project root for full license terms.

//! Renormalization Group (RG) coarse-graining engine for agent communication.
//!
//! Implements intentional decimation of fine-grained conversational histories
//! by filtering high-frequency linguistic noise and preserving structural
//! semantic operators (tool calls, error signals, state transitions).
//!
//! The decimation pipeline operates entirely in Rust with zero external
//! dependencies, using three scoring dimensions:
//!
//! 1. **Keyword Density**: Measures concentration of structural action verbs,
//!    error indicators, and system signals within each message.
//! 2. **Information Density**: Lexical diversity × length normalization to
//!    identify substantive content versus conversational filler.
//! 3. **Redundancy Penalty**: Bigram Jaccard overlap against recent context
//!    to detect and penalize repetitive/looping messages.
//!
//! The composite score is: relevance = w₁·keyword + w₂·info + w₃·(1 - redundancy)

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashSet;

use crate::vsa::HolographicEngine;

// ─── Constants ─────────────────────────────────────────────────────────────

/// Default relevance threshold for message retention (0.0 to 1.0).
pub const DEFAULT_THRESHOLD: f64 = 0.3;

/// Default maximum number of messages to retain after decimation.
pub const DEFAULT_MAX_RETAINED: usize = 50;

/// Number of recent context messages to consider for redundancy detection.
const REDUNDANCY_LOOKBACK: usize = 10;

/// Scoring weights for the composite relevance function.
const W_KEYWORD: f64 = 0.35;
const W_INFO: f64 = 0.30;
const W_NOVELTY: f64 = 0.35;

// ─── Scored Message ────────────────────────────────────────────────────────

/// A message annotated with its RG relevance score and retention status.
#[pyclass(name = "ScoredMessage")]
#[derive(Clone, Debug)]
pub struct ScoredMessage {
    /// Original index in the input message list.
    #[pyo3(get)]
    pub index: usize,
    /// Composite relevance score in [0.0, 1.0].
    #[pyo3(get)]
    pub score: f64,
    /// Original message content.
    #[pyo3(get)]
    pub content: String,
    /// Whether this message survives decimation.
    #[pyo3(get)]
    pub retained: bool,
}

#[pymethods]
impl ScoredMessage {
    fn __repr__(&self) -> String {
        format!(
            "ScoredMessage(index={}, score={:.3}, retained={}, content={:?})",
            self.index,
            self.score,
            self.retained,
            if self.content.len() > 60 {
                format!("{}...", &self.content[..60])
            } else {
                self.content.clone()
            }
        )
    }
}

// ─── RG Decimator ──────────────────────────────────────────────────────────

/// Renormalization Group decimation engine for agent communication compression.
///
/// Downsamples fine-grained message histories (UV microstates) into compressed,
/// scale-invariant summaries (IR macrostates) by filtering high-frequency noise
/// while preserving structural semantic operators.
///
/// Each message is scored along three axes: keyword density, information density,
/// and novelty (inverse redundancy). Messages below the threshold are discarded.
///
/// The first and last messages are always retained (system prompt + most recent turn).
#[pyclass(name = "RGDecimator")]
pub struct RGDecimator {
    threshold: f64,
    max_retained: usize,
    structural_keywords: Vec<String>,
}

#[pymethods]
impl RGDecimator {
    /// Create a new RG Decimator.
    ///
    /// Args:
    ///     threshold: Minimum relevance score for retention (default: 0.3).
    ///     max_retained: Maximum messages to keep after decimation (default: 50).
    ///     structural_keywords: Custom keywords indicating high-relevance content.
    ///                          If None, uses a curated default set of action verbs,
    ///                          error signals, and state transition markers.
    #[new]
    #[pyo3(signature = (threshold=DEFAULT_THRESHOLD, max_retained=DEFAULT_MAX_RETAINED, structural_keywords=None))]
    fn new(
        threshold: f64,
        max_retained: usize,
        structural_keywords: Option<Vec<String>>,
    ) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(PyValueError::new_err(
                "Threshold must be in [0.0, 1.0]",
            ));
        }
        if max_retained == 0 {
            return Err(PyValueError::new_err("max_retained must be ≥ 1"));
        }

        let keywords = structural_keywords.unwrap_or_else(default_structural_keywords);

        Ok(Self {
            threshold,
            max_retained,
            structural_keywords: keywords,
        })
    }

    /// Score and decimate a list of messages.
    ///
    /// Returns all messages annotated with relevance scores and retention flags.
    /// The first and last messages are always retained regardless of score.
    ///
    /// Args:
    ///     messages: List of message strings (conversational turns).
    ///
    /// Returns:
    ///     List of ScoredMessage objects with index, score, content, and retained flag.
    fn decimate(&self, messages: Vec<String>) -> Vec<ScoredMessage> {
        if messages.is_empty() {
            return Vec::new();
        }

        let scores = self.score_all(&messages);
        let mut scored: Vec<ScoredMessage> = messages
            .into_iter()
            .enumerate()
            .zip(scores.iter())
            .map(|((i, content), &score)| ScoredMessage {
                index: i,
                score,
                content,
                retained: score >= self.threshold,
            })
            .collect();

        // Always retain first message (system prompt) and last (most recent turn)
        if let Some(first) = scored.first_mut() {
            first.retained = true;
        }
        if scored.len() > 1 {
            if let Some(last) = scored.last_mut() {
                last.retained = true;
            }
        }

        // Enforce max_retained: if too many pass threshold, keep top-K by score
        let retained_count = scored.iter().filter(|s| s.retained).count();
        if retained_count > self.max_retained {
            let mut retained_scores: Vec<(usize, f64)> = scored
                .iter()
                .filter(|s| s.retained)
                .map(|s| (s.index, s.score))
                .collect();
            retained_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let to_drop: HashSet<usize> = retained_scores
                .iter()
                .skip(self.max_retained)
                .map(|(idx, _)| *idx)
                .collect();

            let scored_len = scored.len();
            for msg in scored.iter_mut() {
                if to_drop.contains(&msg.index) {
                    // Never drop first or last
                    if msg.index != 0 && msg.index != scored_len - 1 {
                        msg.retained = false;
                    }
                }
            }
        }

        scored
    }

    /// Compress messages by returning only those that survive decimation.
    ///
    /// This is a convenience method equivalent to filtering decimate() results.
    ///
    /// Args:
    ///     messages: List of message strings.
    ///
    /// Returns:
    ///     List of retained message strings in original order.
    fn compress(&self, messages: Vec<String>) -> Vec<String> {
        self.decimate(messages)
            .into_iter()
            .filter(|s| s.retained)
            .map(|s| s.content)
            .collect()
    }

    /// Score a single message for relevance against optional context.
    ///
    /// Args:
    ///     message: The message to score.
    ///     context: Optional list of preceding messages for redundancy detection.
    ///
    /// Returns:
    ///     Relevance score in [0.0, 1.0].
    #[pyo3(signature = (message, context=None))]
    fn score_message(&self, message: &str, context: Option<Vec<String>>) -> f64 {
        let ctx = context.unwrap_or_default();
        let ctx_refs: Vec<&str> = ctx.iter().map(|s| s.as_str()).collect();
        self.compute_score(message, &ctx_refs)
    }

    /// Get the current structural keywords list.
    fn get_keywords(&self) -> Vec<String> {
        self.structural_keywords.clone()
    }

    /// Get the configured threshold.
    fn get_threshold(&self) -> f64 {
        self.threshold
    }

    /// Get the configured max_retained.
    fn get_max_retained(&self) -> usize {
        self.max_retained
    }

    /// Score and decimate messages using pre-computed VSA bipolar embeddings.
    ///
    /// This method integrates the Holographic Engine with the RG decimator.
    /// Instead of using bigram Jaccard overlap for redundancy detection, it
    /// computes VSA cosine similarity between message embeddings, fulfilling
    /// the paper's specification of "TF-IDF + VSA semantic similarity".
    ///
    /// The embeddings should be generated via `HolographicEngine.batch_encode_texts()`
    /// or `HolographicEngine.encode_text()` before calling this method.
    ///
    /// Args:
    ///     messages: List of message strings (conversational turns).
    ///     embeddings: List of bipolar vectors, one per message, from HolographicEngine.
    ///
    /// Returns:
    ///     List of ScoredMessage objects with VSA-enhanced scoring.
    fn decimate_with_embeddings(
        &self,
        messages: Vec<String>,
        embeddings: Vec<Vec<i8>>,
    ) -> PyResult<Vec<ScoredMessage>> {
        if messages.len() != embeddings.len() {
            return Err(PyValueError::new_err(format!(
                "messages length ({}) must match embeddings length ({})",
                messages.len(),
                embeddings.len(),
            )));
        }
        if messages.is_empty() {
            return Ok(Vec::new());
        }

        let scores = self.score_all_with_embeddings(&messages, &embeddings);
        let mut scored: Vec<ScoredMessage> = messages
            .into_iter()
            .enumerate()
            .zip(scores.iter())
            .map(|((i, content), &score)| ScoredMessage {
                index: i,
                score,
                content,
                retained: score >= self.threshold,
            })
            .collect();

        // Always retain first and last messages
        if let Some(first) = scored.first_mut() {
            first.retained = true;
        }
        if scored.len() > 1 {
            if let Some(last) = scored.last_mut() {
                last.retained = true;
            }
        }

        // Enforce max_retained (same logic as decimate)
        let retained_count = scored.iter().filter(|s| s.retained).count();
        if retained_count > self.max_retained {
            let mut retained_scores: Vec<(usize, f64)> = scored
                .iter()
                .filter(|s| s.retained)
                .map(|s| (s.index, s.score))
                .collect();
            retained_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let to_drop: HashSet<usize> = retained_scores
                .iter()
                .skip(self.max_retained)
                .map(|(idx, _)| *idx)
                .collect();

            let scored_len = scored.len();
            for msg in scored.iter_mut() {
                if to_drop.contains(&msg.index) {
                    if msg.index != 0 && msg.index != scored_len - 1 {
                        msg.retained = false;
                    }
                }
            }
        }

        Ok(scored)
    }
}

// ─── Internal scoring engine ───────────────────────────────────────────────

impl RGDecimator {
    /// Score all messages considering inter-message redundancy (text-only).
    fn score_all(&self, messages: &[String]) -> Vec<f64> {
        let mut scores = Vec::with_capacity(messages.len());
        for (i, msg) in messages.iter().enumerate() {
            let context: Vec<&str> = messages[..i].iter().map(|s| s.as_str()).collect();
            let score = self.compute_score(msg, &context);
            scores.push(score);
        }
        scores
    }

    /// Score all messages using VSA cosine similarity for redundancy detection.
    ///
    /// Replaces bigram Jaccard overlap with holographic vector cosine distance,
    /// integrating the VSA subsystem with the RG coarse-graining pipeline.
    fn score_all_with_embeddings(
        &self,
        messages: &[String],
        embeddings: &[Vec<i8>],
    ) -> Vec<f64> {
        let mut scores = Vec::with_capacity(messages.len());
        for (i, msg) in messages.iter().enumerate() {
            let keyword_score = self.keyword_density(msg);
            let info_score = Self::information_density(msg);

            // VSA cosine redundancy: max cosine similarity against recent context
            let vsa_redundancy = if i > 0 {
                let lookback_start = i.saturating_sub(REDUNDANCY_LOOKBACK);
                let mut max_sim = 0.0f64;
                for j in lookback_start..i {
                    let sim = HolographicEngine::cosine_internal(
                        &embeddings[i],
                        &embeddings[j],
                    );
                    // Cosine of bipolar vectors is in [-1, 1]; map to [0, 1]
                    let normalized_sim = (sim + 1.0) / 2.0;
                    max_sim = max_sim.max(normalized_sim);
                }
                max_sim
            } else {
                0.0
            };

            let raw = W_KEYWORD * keyword_score
                + W_INFO * info_score
                + W_NOVELTY * (1.0 - vsa_redundancy);
            scores.push(raw.clamp(0.0, 1.0));
        }
        scores
    }

    /// Compute a composite relevance score for a single message.
    ///
    /// Score = w₁·keyword_density + w₂·information_density + w₃·(1 - redundancy)
    fn compute_score(&self, message: &str, context: &[&str]) -> f64 {
        let keyword_score = self.keyword_density(message);
        let info_score = Self::information_density(message);
        let redundancy = Self::redundancy_penalty(message, context);

        let raw = W_KEYWORD * keyword_score + W_INFO * info_score + W_NOVELTY * (1.0 - redundancy);
        raw.clamp(0.0, 1.0)
    }

    /// Measure density of structural keywords in the message.
    ///
    /// Looks for matches of each structural keyword within the tokenized
    /// lowercase message. Returns the fraction of tokens matching any keyword,
    /// clamped to [0.0, 1.0].
    fn keyword_density(&self, message: &str) -> f64 {
        let lower = message.to_lowercase();
        let tokens = tokenize(&lower);
        if tokens.is_empty() {
            return 0.0;
        }
        let hits: usize = tokens
            .iter()
            .filter(|t| {
                self.structural_keywords
                    .iter()
                    .any(|kw| t.contains(kw.as_str()))
            })
            .count();
        (hits as f64 / tokens.len() as f64).min(1.0)
    }

    /// Measure information density using lexical diversity and length normalization.
    ///
    /// Lexical diversity = unique_tokens / total_tokens.
    /// Length factor penalizes both trivially short messages (filler) and
    /// extremely long messages (dumps/logs).
    fn information_density(message: &str) -> f64 {
        let tokens = tokenize(&message.to_lowercase());
        if tokens.is_empty() {
            return 0.0;
        }

        let unique: HashSet<&str> = tokens.iter().map(|s| s.as_str()).collect();
        let diversity = unique.len() as f64 / tokens.len() as f64;

        let length_factor = match tokens.len() {
            0..=2 => 0.2,
            3..=10 => 0.6,
            11..=50 => 1.0,
            51..=200 => 0.8,
            _ => 0.5,
        };

        diversity * length_factor
    }

    /// Compute redundancy penalty based on bigram Jaccard overlap with recent context.
    ///
    /// Compares the current message's word bigrams against the most recent
    /// `REDUNDANCY_LOOKBACK` messages. Returns the maximum overlap found,
    /// heavily penalizing near-duplicate or looping messages.
    fn redundancy_penalty(message: &str, context: &[&str]) -> f64 {
        if context.is_empty() {
            return 0.0;
        }

        let msg_bigrams = bigrams(&message.to_lowercase());
        if msg_bigrams.is_empty() {
            return 0.0;
        }

        let lookback_start = context.len().saturating_sub(REDUNDANCY_LOOKBACK);
        let recent = &context[lookback_start..];

        let mut max_overlap = 0.0f64;
        for ctx_msg in recent {
            let ctx_bigrams = bigrams(&ctx_msg.to_lowercase());
            if ctx_bigrams.is_empty() {
                continue;
            }

            let intersection = msg_bigrams.intersection(&ctx_bigrams).count();
            let union = msg_bigrams.len() + ctx_bigrams.len() - intersection;
            let jaccard = if union > 0 {
                intersection as f64 / union as f64
            } else {
                0.0
            };
            max_overlap = max_overlap.max(jaccard);
        }

        max_overlap
    }
}

// ─── Utility functions ─────────────────────────────────────────────────────

/// Default set of structural keywords indicating high-relevance content.
fn default_structural_keywords() -> Vec<String> {
    [
        // Action verbs (tool invocations)
        "execute", "create", "update", "delete", "insert", "query",
        "deploy", "migrate", "validate", "transform", "invoke", "call",
        // Error signals
        "error", "failed", "exception", "timeout", "retry",
        "rejected", "invalid", "violation", "denied", "abort", "crashed",
        // Tool/system signals
        "tool_call", "function_call", "api_response", "result",
        "observation", "action", "thought", "plan", "response",
        // State transitions
        "completed", "started", "pending", "approved", "blocked",
        "resolved", "committed", "rolled_back", "cancelled",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// Whitespace tokenizer with punctuation stripping.
fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|t| {
            t.trim_matches(|c: char| !c.is_alphanumeric() && c != '_')
                .to_string()
        })
        .filter(|t| !t.is_empty())
        .collect()
}

/// Generate word-level bigrams for overlap detection.
fn bigrams(text: &str) -> HashSet<String> {
    let tokens = tokenize(text);
    if tokens.len() < 2 {
        return HashSet::new();
    }
    tokens
        .windows(2)
        .map(|w| format!("{} {}", w[0], w[1]))
        .collect()
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_decimator() -> RGDecimator {
        RGDecimator::new(0.3, 50, None).unwrap()
    }

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("hello, world! this is a test.");
        assert_eq!(tokens, vec!["hello", "world", "this", "is", "a", "test"]);
    }

    #[test]
    fn test_tokenize_preserves_underscores() {
        let tokens = tokenize("tool_call function_call api_response");
        assert_eq!(tokens, vec!["tool_call", "function_call", "api_response"]);
    }

    #[test]
    fn test_bigrams_generation() {
        let bg = bigrams("the quick brown fox");
        assert!(bg.contains("the quick"));
        assert!(bg.contains("quick brown"));
        assert!(bg.contains("brown fox"));
        assert_eq!(bg.len(), 3);
    }

    #[test]
    fn test_bigrams_too_short() {
        let bg = bigrams("hello");
        assert!(bg.is_empty());
    }

    #[test]
    fn test_keyword_density_high() {
        let dec = make_decimator();
        let msg = "execute query failed with error timeout retry";
        let density = dec.keyword_density(msg);
        // 6 out of 7 tokens are keywords
        assert!(density > 0.7, "Expected high keyword density, got {density}");
    }

    #[test]
    fn test_keyword_density_zero() {
        let dec = make_decimator();
        let msg = "the weather is nice today";
        let density = dec.keyword_density(msg);
        assert!(
            density < 0.01,
            "Expected near-zero keyword density, got {density}"
        );
    }

    #[test]
    fn test_information_density_short() {
        let density = RGDecimator::information_density("ok");
        // Short messages get a low length factor (0.2)
        assert!(density < 0.3, "Expected low info density for short message, got {density}");
    }

    #[test]
    fn test_information_density_substantive() {
        let msg = "The database migration completed successfully with 42 rows transformed \
                   and 3 validation errors detected in the schema mapping layer";
        let density = RGDecimator::information_density(msg);
        assert!(density > 0.4, "Expected high info density, got {density}");
    }

    #[test]
    fn test_redundancy_no_context() {
        let penalty = RGDecimator::redundancy_penalty("hello world", &[]);
        assert!((penalty - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_redundancy_duplicate() {
        let msg = "execute the database query and validate results";
        let context = vec![msg];
        let ctx_refs: Vec<&str> = context.iter().map(|s| s.as_ref()).collect();
        let penalty = RGDecimator::redundancy_penalty(msg, &ctx_refs);
        // Exact duplicate should have very high overlap
        assert!(penalty > 0.9, "Expected high redundancy for duplicate, got {penalty}");
    }

    #[test]
    fn test_redundancy_different() {
        let msg = "the weather forecast shows sunny skies tomorrow";
        let context = vec!["execute database migration with schema validation"];
        let ctx_refs: Vec<&str> = context.iter().map(|s| s.as_ref()).collect();
        let penalty = RGDecimator::redundancy_penalty(msg, &ctx_refs);
        assert!(penalty < 0.2, "Expected low redundancy for different messages, got {penalty}");
    }

    #[test]
    fn test_decimate_empty() {
        let dec = make_decimator();
        let result = dec.decimate(Vec::new());
        assert!(result.is_empty());
    }

    #[test]
    fn test_decimate_preserves_first_and_last() {
        let dec = RGDecimator::new(0.99, 50, None).unwrap(); // very high threshold
        let messages = vec![
            "system prompt".to_string(),
            "filler".to_string(),
            "latest message".to_string(),
        ];
        let result = dec.decimate(messages);
        assert!(result[0].retained, "First message should always be retained");
        assert!(result[2].retained, "Last message should always be retained");
    }

    #[test]
    fn test_compress_returns_only_retained() {
        let dec = make_decimator();
        let messages = vec![
            "System: You are a helpful assistant. Execute tool calls as needed.".to_string(),
            "ok".to_string(),
            "sure".to_string(),
            "Tool call: execute query on database, validate results, check error status".to_string(),
            "The migration completed with 3 errors and 42 rows transformed successfully".to_string(),
        ];
        let compressed = dec.compress(messages.clone());
        // First and last are always retained
        assert!(compressed.len() >= 2);
        assert!(compressed.len() <= messages.len());
    }

    #[test]
    fn test_score_message_standalone() {
        let dec = make_decimator();
        let score = dec.score_message(
            "execute query failed with timeout error",
            None,
        );
        assert!(score > 0.3, "Keyword-rich message should score above threshold, got {score}");
    }

    #[test]
    fn test_max_retained_enforcement() {
        let dec = RGDecimator::new(0.0, 3, None).unwrap(); // threshold 0 = keep all, but cap at 3
        let messages: Vec<String> = (0..10)
            .map(|i| format!("Message {i} with execute action and error handling"))
            .collect();
        let result = dec.decimate(messages);
        let retained_count = result.iter().filter(|s| s.retained).count();
        assert!(
            retained_count <= 3 + 2,
            "Expected at most 5 retained (max_retained + 2 for first/last), got {retained_count}"
        );
    }

    #[test]
    fn test_invalid_threshold() {
        assert!(RGDecimator::new(-0.1, 50, None).is_err());
        assert!(RGDecimator::new(1.1, 50, None).is_err());
    }

    #[test]
    fn test_invalid_max_retained() {
        assert!(RGDecimator::new(0.3, 0, None).is_err());
    }

    #[test]
    fn test_score_all_with_embeddings_redundancy() {
        let dec = make_decimator();
        // Simulate messages where msg[2] is a repeat of msg[1] (same embedding)
        let messages = vec![
            "System prompt".to_string(),
            "Execute tool call with database query".to_string(),
            "Execute tool call with database query".to_string(), // duplicate
            "Final result summary".to_string(),
        ];
        let dim = 100;
        let embeddings: Vec<Vec<i8>> = messages
            .iter()
            .map(|m| HolographicEngine::encode_text_internal(m, dim))
            .collect();

        let scores = dec.score_all_with_embeddings(&messages, &embeddings);

        // The duplicate message (index 2) should score lower than the original (index 1)
        // because VSA cosine detects the redundancy
        assert!(
            scores[2] < scores[1],
            "Duplicate message should have lower score: original={:.3} vs duplicate={:.3}",
            scores[1],
            scores[2],
        );
    }

    #[test]
    fn test_decimate_with_embeddings_preserves_first_last() {
        let dec = RGDecimator::new(0.99, 50, None).unwrap(); // very high threshold
        let messages = vec![
            "system prompt".to_string(),
            "filler".to_string(),
            "latest message".to_string(),
        ];
        let dim = 100;
        let embeddings: Vec<Vec<i8>> = messages
            .iter()
            .map(|m| HolographicEngine::encode_text_internal(m, dim))
            .collect();

        let result = dec
            .decimate_with_embeddings(messages, embeddings)
            .unwrap();
        assert!(result[0].retained, "First message must always be retained");
        assert!(result[2].retained, "Last message must always be retained");
    }

    #[test]
    fn test_decimate_with_embeddings_length_mismatch() {
        let dec = make_decimator();
        let messages = vec!["a".to_string(), "b".to_string()];
        let embeddings = vec![vec![1i8; 10]]; // wrong length

        let result = dec.decimate_with_embeddings(messages, embeddings);
        assert!(result.is_err(), "Mismatched lengths should produce an error");
    }
}
