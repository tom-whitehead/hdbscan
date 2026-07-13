#[derive(Clone, Debug)]
pub(crate) struct MSTEdge<T> {
    pub(crate) left_node_id: usize,
    pub(crate) right_node_id: usize,
    pub(crate) distance: T,
}

pub(crate) struct SLTNode<T> {
    pub(crate) left_child: usize,
    pub(crate) right_child: usize,
    pub(crate) distance: T,
    pub(crate) size: usize,
}

/// A node in the condensed cluster tree.
///
/// Maps to the Python HDBSCAN library's condensed tree DataFrame columns:
/// - `node_id` → `child`
/// - `parent_node_id` → `parent`
/// - `lambda_birth` → `lambda_val`
/// - `size` → `child_size`
#[derive(Clone, Debug, PartialEq)]
pub struct CondensedNode<T> {
    /// The ID of this node. For individual data points this is the point index (< n_samples).
    /// For cluster nodes this is >= n_samples. Equivalent to Python's `child` column.
    pub node_id: usize,
    /// The ID of the parent cluster (always >= n_samples). Equivalent to Python's `parent` column.
    pub parent_node_id: usize,
    /// The lambda value (1/distance) at which this node was born. Equivalent to Python's `lambda_val`.
    pub lambda_birth: T,
    /// The number of points in this node (1 for individual points). Equivalent to Python's `child_size`.
    pub size: usize,
}
