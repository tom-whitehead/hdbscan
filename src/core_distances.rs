const BRUTE_FORCE_N_SAMPLES_LIMIT: usize = 250;

#[cfg(feature = "parallel")]
pub(super) mod parallel;
#[cfg(feature = "serial")]
pub(super) mod serial;

/// The nearest neighbour algorithm options
#[derive(Debug, Clone, PartialEq)]
pub enum NnAlgorithm {
    /// HDBSCAN internally selects the nearest neighbour based on size
    /// and dimensionality of the input data
    Auto,
    /// Computes a distance matrix between each point and all others
    BruteForce,
    /// K-dimensional tree algorithm.
    KdTree,
}
