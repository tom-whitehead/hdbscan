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

pub(crate) struct CondensedNode<T> {
    pub(crate) node_id: usize,
    pub(crate) parent_node_id: usize,
    pub(crate) lambda_birth: T,
    pub(crate) size: usize,
}
