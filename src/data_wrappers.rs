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

/// A node in the condensed cluster tree produced by HDBSCAN. Exposed
/// publicly to enable external `approximate_predict`-style inference on
/// new points (assigning previously-unseen samples to existing clusters
/// without re-running the full algorithm).
///
/// Marked `#[non_exhaustive]` so additional fields can be added in
/// future revisions without breaking external consumers.
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "T: serde::Serialize",
        deserialize = "T: serde::Deserialize<'de>"
    ))
)]
pub struct CondensedNode<T> {
    pub node_id: usize,
    pub parent_node_id: usize,
    pub lambda_birth: T,
    pub size: usize,
}
