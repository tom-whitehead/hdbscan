use crate::data_wrappers::CondensedNode;

/// Detailed clustering result exposing diagnostics equivalent to Python's HDBSCAN.
///
/// Contains cluster labels, membership probabilities, the condensed tree,
/// and outlier scores (GLOSH).
#[derive(Debug, Clone)]
pub struct HdbscanResult<T> {
    /// Cluster labels for each data point. -1 indicates noise.
    pub labels: Vec<i32>,
    /// Membership probability for each point in its assigned cluster. 0 for noise points.
    pub probabilities: Vec<T>,
    /// The condensed cluster hierarchy.
    pub condensed_tree: Vec<CondensedNode<T>>,
    /// GLOSH outlier scores for each point. Range \[0, 1\], higher = more outlier-like.
    pub outlier_scores: Vec<T>,
}

impl<T> HdbscanResult<T> {
    pub(crate) fn new(
        labels: Vec<i32>,
        probabilities: Vec<T>,
        condensed_tree: Vec<CondensedNode<T>>,
        outlier_scores: Vec<T>,
    ) -> Self {
        HdbscanResult {
            labels,
            probabilities,
            condensed_tree,
            outlier_scores,
        }
    }
}
