//! Hierarchical Density-Based Spatial Clustering of Applications with Noise ("HDBSCAN") clustering
//! algorithm in Rust. Generic over floating point numeric types.
//!
//! HDBSCAN is a powerful clustering algorithm that can be used to effectively find clusters in
//! real world data. The main benefits of HDBSCAN are that:
//!  1. It does not assume that all data points belong to a cluster, as many clustering algorithms
//!     do. I.e. a data set can contain "noise" points. This is important for modelling real world
//!     data, which is inherently noisy;
//!  2. It allows clusters of varying densities, unlike the plain DBSCAN algorithm which uses a
//!     static density threshold. The winning clusters are those that persist the longest at all
//!     densities. This is also crucial for modelling real world data; and
//!  3. It makes no assumptions about the number of clusters there have to be, unlike KMeans
//!     clustering. The algorithm will select just select the clusters that are the most persistent
//!     at all densities.
//!
//! This implementation owes a debt to the Python scikit-learn implementation of this algorithm,
//! without which this algorithm would not have been possible. The "How HDBSCAN works" article
//! below is invaluable in understanding this algorithm better.
//!
//! # Examples
//! ```
//!use hdbscan::Hdbscan;
//!
//!let data: Vec<Vec<f32>> = vec![
//!    vec![1.5, 2.2],
//!    vec![1.0, 1.1],
//!    vec![1.2, 1.4],
//!    vec![0.8, 1.0],
//!    vec![1.1, 1.0],
//!    vec![3.7, 4.0],
//!    vec![3.9, 3.9],
//!    vec![3.6, 4.1],
//!    vec![3.8, 3.9],
//!    vec![4.0, 4.1],
//!    vec![10.0, 10.0],
//!];
//!let clusterer = Hdbscan::default(&data);
//!let result = clusterer.cluster().unwrap();
//!assert_eq!(result, vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1]);
//! ```
//!
//! # References
//! * [Campello, R.J.G.B.; Moulavi, D.; Sander, J. Density-based clustering based on hierarchical density estimates.](https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14)
//! * [How HDBSCAN Works](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html)

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, VecDeque};
use kdtree::KdTree;
use num_traits::Float;
use crate::data_wrappers::{CondensedNode, MSTEdge, SLTNode};
use crate::union_find::UnionFind;

pub use crate::centers::Center;
pub use crate::distance::DistanceMetric;
pub use crate::hyper_parameters::{HdbscanHyperParams, HyperParamBuilder};
pub use crate::error::HdbscanError;

mod hyper_parameters;
mod data_wrappers;
mod distance;
mod error;
mod union_find;
mod centers;

/// The HDBSCAN clustering algorithm in Rust. Generic over floating point numeric types.
pub struct Hdbscan<'a, T> {
    data: &'a Vec<Vec<T>>,
    n_samples: usize,
    n_dims: usize,
    hyper_params: HdbscanHyperParams,
    labels: RefCell<Option<Vec<i32>>>,
}

impl<'a, T: Float> Hdbscan<'a, T> {

    /// Creates an instance of HDBSCAN clustering model using a custom hyper parameter
    /// configuration.
    ///
    /// # Parameters
    /// * `data` - a reference to the data to cluster, a collection of vectors of floating points
    ///            numbers. The vectors must all be of the same dimensionality and contain no
    ///            infinite values.
    /// * `config` - the hyper parameter configuration.
    ///
    /// # Returns
    /// * The HDBSCAN model instance.
    ///
    /// # Examples
    /// ```
    ///use hdbscan::{DistanceMetric, Hdbscan, HdbscanHyperParams};
    ///
    ///let data: Vec<Vec<f32>> = vec![
    ///    vec![1.3, 1.1],
    ///    vec![1.3, 1.2],
    ///    vec![1.0, 1.1],
    ///    vec![1.2, 1.2],
    ///    vec![0.9, 1.0],
    ///    vec![0.9, 1.0],
    ///    vec![3.7, 4.0],
    ///    vec![3.9, 3.9],
    ///];
    ///let config = HdbscanHyperParams::builder()
    ///    .min_cluster_size(3)
    ///    .min_samples(2)
    ///    .dist_metric(DistanceMetric::Manhattan)
    ///    .build();
    ///let clusterer = Hdbscan::new(&data, config);
    /// ```
    pub fn new(data: &'a Vec<Vec<T>>, hyper_params: HdbscanHyperParams) -> Self {
        let n_samples = data.len();
        let n_dims = if data.is_empty() {0} else { data[0].len() };
        Hdbscan { data, n_samples, n_dims , hyper_params, labels: RefCell::new(None) }
    }

    /// Creates an instance of HDBSCAN clustering model using the default hyper parameters.
    ///
    /// # Parameters
    /// * `data` - a reference to the data to cluster, a collection of vectors of floating points
    ///            numbers. The vectors must all be of the same dimensionality and contain no
    ///            infinite values.
    ///
    /// # Returns
    /// * The HDBSCAN model instance.
    ///
    /// # Examples
    /// ```
    ///use hdbscan::Hdbscan;
    ///
    ///let data: Vec<Vec<f32>> = vec![
    ///    vec![1.3, 1.1],
    ///    vec![1.3, 1.2],
    ///    vec![1.0, 1.1],
    ///    vec![1.2, 1.2],
    ///    vec![0.9, 1.0],
    ///    vec![0.9, 1.0],
    ///    vec![3.7, 4.0],
    ///    vec![3.9, 3.9],
    ///];
    ///let clusterer = Hdbscan::default(&data);
    /// ```
    pub fn default(data: &'a Vec<Vec<T>>) -> Self {
        let hyper_params = HdbscanHyperParams::default();
        Hdbscan::new(data, hyper_params)
    }

    /// Performs clustering on the list of vectors passed to the constructor.
    ///
    /// # Returns
    /// * A result that, if successful, contains a list of cluster labels, with a length equal to
    ///   the numbe of samples passed to the constructor. Positive integers mean a data point
    ///   belongs to a cluster of that label. -1 labels mean that a data point is noise and does
    ///   not belong to any cluster. An Error will be returned if the dimensionality of the input
    ///   vectors are mismatched, if any vector contains non-finite coordinates, or if the passed
    ///   data set is empty.
    ///
    /// # Examples
    /// ```
    ///use hdbscan::Hdbscan;
    ///
    ///let data: Vec<Vec<f32>> = vec![
    ///    vec![1.5, 2.2],
    ///    vec![1.0, 1.1],
    ///    vec![1.2, 1.4],
    ///    vec![0.8, 1.0],
    ///    vec![1.1, 1.0],
    ///    vec![3.7, 4.0],
    ///    vec![3.9, 3.9],
    ///    vec![3.6, 4.1],
    ///    vec![3.8, 3.9],
    ///    vec![4.0, 4.1],
    ///    vec![10.0, 10.0],
    ///];
    ///let clusterer = Hdbscan::default(&data);
    ///let result = clusterer.cluster().unwrap();
    ///assert_eq!(result, vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1]);
    /// ```
    pub fn cluster(&self) -> Result<Vec<i32>, HdbscanError> {
        self.validate_input_data()?;
        let core_distances = self.calc_core_distances();
        let min_spanning_tree = self.prims_min_spanning_tree(&core_distances);
        let single_linkage_tree = self.make_single_linkage_tree(&min_spanning_tree);
        let condensed_tree = self.condense_tree(&single_linkage_tree);
        let winning_clusters = self.extract_winning_clusters(&condensed_tree);
        let labelled_data = self.label_data(&winning_clusters, &condensed_tree);
        Ok(labelled_data)
    }

    pub fn calc_centers(&self, center: Center) -> Result<Vec<Vec<T>>, HdbscanError> {
        if self.labels.borrow().is_none() {
            panic!(
                "Clustering must be completed before cluster centers can be calculated. \
                Call Hdbscan::cluster to perform clustering.");
        }
        let bound_labels = self.labels.borrow();
        let labels = bound_labels.as_ref().unwrap();
        match center {
            Center::Centroid => Ok(center.calc_centers(self.data, labels))
        }
    }

    fn validate_input_data(&self) -> Result<(), HdbscanError> {
        if self.data.is_empty() {
            return Err(HdbscanError::EmptyDataset);
        }
        let dims_0th = self.data[0].len();
        for (n, datapoint) in self.data.iter().enumerate() {
            for element in datapoint {
                if element.is_infinite() {
                    return Err(HdbscanError::NonFiniteCoordinate(
                        format!("{n}th vector contains non-finite element(s)")));
                }
            }

            let dims_nth = datapoint.len();
            if dims_nth != dims_0th {
                return Err(HdbscanError::WrongDimension(
                    format!("Oth data point has {dims_0th} dimensions, but {n}th has {dims_nth}")));
            }
        }
        Ok(())
    }

    fn calc_core_distances(&self) -> Vec<T> {
        // TODO: Wrap KdTree in standardised interface and inject
        let mut tree = KdTree::new(self.n_dims);
        self.data.iter().enumerate()
            // Unwrap should be safe due to data validation above
            // TODO: Address duplication of validation above and in KdTree
            .for_each(|(n, datum)| tree.add(datum, n).unwrap());

        let k = self.hyper_params.min_samples;
        let dist_func = distance::get_dist_func::<T>(&self.hyper_params.dist_metric);
        self.data.iter()
            .map(|datum| {
                // Unwrap should be safe due to data validation above
                let result = tree.nearest(datum, k, &dist_func).unwrap();
                result.into_iter()
                    .map(|(dist, _idx)| dist)
                    .last()
                    .unwrap()
            })
            .collect()
    }

    fn prims_min_spanning_tree(&self, core_distances: &Vec<T>) -> Vec<MSTEdge<T>> {
        // TODO: Move out to a dedicated object and inject to enable use of other mst algorithms
        let mut in_tree = vec![false; self.n_samples];
        let mut distances = vec![T::infinity(); self.n_samples];
        let mut parents = vec![0; self.n_samples];

        distances[0] = T::zero();

        for _ in 1..self.n_samples {
            let left_node = self.select_min_node(&distances, &in_tree);
            in_tree[left_node] = true;

            for right_node in 1..self.n_samples {
                if in_tree[right_node] {
                    continue;
                }
                let mrd = self.calc_mutual_reachability_dist(left_node, right_node, core_distances);
                if mrd < distances[right_node] {
                    distances[right_node] = mrd;
                    parents[right_node] = left_node;
                }
            }
        }
        let mut mst = self.collect_mst(&parents, &distances);
        self.sort_mst_by_dist(&mut mst);
        mst
    }

    fn select_min_node(&self, distances: &Vec<T>, in_tree: &Vec<bool>) -> usize {
        let mut min_dist = T::infinity();
        let mut node = 0;
        for (i, (dist, is_in_tree)) in distances.iter().zip(in_tree).enumerate() {
            if !is_in_tree && dist < &min_dist {
                min_dist = *dist;
                node = i;
            }
        }
        node
    }

    fn calc_mutual_reachability_dist(&self, a: usize, b: usize, core_distances: &Vec<T>) -> T {
        let core_dist_a = core_distances[a];
        let core_dist_b = core_distances[b];
        let dist_a_b = self.hyper_params.dist_metric.calc_dist(&self.data[a], &self.data[b]);

        core_dist_a.max(core_dist_b).max(dist_a_b)
    }

    fn collect_mst(&self, parents: &Vec<usize>, distances: &Vec<T>) -> Vec<MSTEdge<T>> {
        parents.iter().zip(distances).enumerate()
            .skip(1)
            .map(|(right, (left, dist))| {
                MSTEdge { left_node_id: *left, right_node_id: right, distance: dist.clone() }
            })
            .collect()
    }

    fn sort_mst_by_dist(&self, min_spanning_tree: &mut Vec<MSTEdge<T>>) {
        min_spanning_tree.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
    }

    fn make_single_linkage_tree(&self, min_spanning_tree: &Vec<MSTEdge<T>>) -> Vec<SLTNode<T>> {
        let mut single_linkage_tree: Vec<SLTNode<T>> = Vec::new();

        let mut union_find = UnionFind::new(self.n_samples);

        for i in 0..(self.n_samples - 1) {
            let mst_edge = &min_spanning_tree[i];

            let left_node = mst_edge.left_node_id;
            let right_node = mst_edge.right_node_id;
            let distance = mst_edge.distance;

            let left_child = union_find.find(left_node);
            let right_child = union_find.find(right_node);
            let size = union_find.size_of(left_child) + union_find.size_of(right_child);

            single_linkage_tree.push(SLTNode { left_child, right_child, distance, size });

            union_find.union(left_child, right_child);
        }

        single_linkage_tree
    }

    fn condense_tree(&self, single_linkage_tree: &Vec<SLTNode<T>>) -> Vec<CondensedNode<T>> {
        let top_node = (self.n_samples - 1) * 2;
        let node_ids = self.find_slt_children_breadth_first(single_linkage_tree, top_node);

        let mut new_node_ids = vec![0_usize; top_node + 1];
        new_node_ids[top_node] = self.n_samples;
        let mut next_parent_id = self.n_samples + 1;

        let mut visited = vec![false; node_ids.len()];
        let mut condensed_tree: Vec<CondensedNode<T>> = Vec::new();

        for node_id in node_ids {
            let has_been_visited = visited[node_id];
            if has_been_visited || self.is_individual_sample(&node_id) {
                continue;
            }

            let node = &single_linkage_tree[node_id - self.n_samples];
            let left_child_id = node.left_child;
            let right_child_id = node.right_child;
            let lambda_birth = self.calc_lambda(node.distance);

            let left_child_size = self.extract_cluster_size(left_child_id, single_linkage_tree);
            let right_child_size = self.extract_cluster_size(right_child_id, single_linkage_tree);

            let is_left_a_cluster = self.is_cluster_big_enough(left_child_size);
            let is_right_a_cluster = self.is_cluster_big_enough(right_child_size);

            if is_left_a_cluster && is_right_a_cluster {
                for (child_id, child_size) in [left_child_id, right_child_id].iter()
                    .zip([left_child_size, right_child_size]) {
                    new_node_ids[*child_id] = next_parent_id;
                    next_parent_id += 1;
                    condensed_tree.push(CondensedNode { node_id: new_node_ids[*child_id],
                        parent_node_id: new_node_ids[node_id], lambda_birth, size: child_size });
                }

            } else if !is_left_a_cluster && !is_right_a_cluster {
                let new_node_id = new_node_ids[node_id];
                self.process_individual_children(
                    left_child_id, new_node_id, &single_linkage_tree, &mut condensed_tree,
                    &mut visited, lambda_birth);
                self.process_individual_children(
                    right_child_id, new_node_id, &single_linkage_tree, &mut condensed_tree,
                    &mut visited, lambda_birth);

            } else if !is_left_a_cluster {
                new_node_ids[right_child_id] = new_node_ids[node_id];
                self.process_individual_children(
                    left_child_id, new_node_ids[node_id], &single_linkage_tree,
                    &mut condensed_tree, &mut visited, lambda_birth);

            } else {
                new_node_ids[left_child_id] = new_node_ids[node_id];
                self.process_individual_children(
                    right_child_id, new_node_ids[node_id], &single_linkage_tree,
                    &mut condensed_tree, &mut visited, lambda_birth);
            }
        }
        condensed_tree
    }

    fn find_slt_children_breadth_first(
        &self,
        single_linkage_tree: &Vec<SLTNode<T>>,
        root: usize
    ) -> Vec<usize> {
        let mut process_queue = VecDeque::from([root]);
        let mut child_nodes: Vec<usize> = Vec::new();

        while !process_queue.is_empty() {
            let mut current_node_num = process_queue.pop_front().unwrap();
            child_nodes.push(current_node_num);
            if self.is_individual_sample(&current_node_num) {
                continue;
            }
            current_node_num -= self.n_samples;
            let current_node = &single_linkage_tree[current_node_num];
            process_queue.push_back(current_node.left_child);
            process_queue.push_back(current_node.right_child);
        }

        child_nodes
    }

    fn is_individual_sample(&self, node_id: &usize) -> bool {
        node_id < &self.n_samples
    }

    fn is_cluster(&self, node_id: &usize) -> bool {
        !self.is_individual_sample(&node_id)
    }


    fn calc_lambda(&self, dist: T) -> T {
        if dist > T::zero() { T::one() / dist } else { T::infinity() }
    }

    fn extract_cluster_size(&self, node_id: usize, single_linkage_tree: &Vec<SLTNode<T>>) -> usize {
        if self.is_individual_sample(&node_id) {
            1
        } else {
            single_linkage_tree[node_id - self.n_samples].size
        }
    }

    fn is_cluster_big_enough(&self, cluster_size: usize) -> bool {
        cluster_size >= self.hyper_params.min_cluster_size
    }

    fn process_individual_children(
        &self,
        node_id: usize,
        new_node_id: usize,
        single_linkage_tree: &Vec<SLTNode<T>>,
        condensed_tree: &mut Vec<CondensedNode<T>>,
        visited: &mut Vec<bool>,
        lambda_birth: T
    ) {

        for child_id in self.find_slt_children_breadth_first(&single_linkage_tree, node_id) {
            if self.is_individual_sample(&child_id) {
                condensed_tree.push(CondensedNode {
                    node_id: child_id, parent_node_id: new_node_id, lambda_birth, size: 1 })
            }
            visited[child_id] = true
        }
    }

    fn extract_winning_clusters(&self, condensed_tree: &Vec<CondensedNode<T>>) -> Vec<usize> {
        let n_clusters = condensed_tree.len() - self.n_samples + 1;
        let stabilities = self.calc_all_stabilities(n_clusters, &condensed_tree);
        let mut selected_clusters: HashMap<usize, bool> =
            stabilities.keys().map(|id| (id.clone(), false)).collect();

        for (cluster_id, stability) in stabilities.iter().rev() {
            let immediate_children =
                self.get_immediate_child_clusters(*cluster_id, &condensed_tree);

            let combined_child_stability = immediate_children.iter()
                .map(|node| stabilities.get(&node.node_id)
                    .unwrap_or(&RefCell::new(T::zero())).borrow().clone())
                .fold(T::zero(), std::ops::Add::add);

            if *stability.borrow() > combined_child_stability
                && !self.is_cluster_too_big(cluster_id, condensed_tree) {
                *selected_clusters.get_mut(&cluster_id).unwrap() = true;

                // If child clusters were already marked as winning clusters reverse
                immediate_children.iter().for_each(|node| {
                    let is_child_selected = selected_clusters.get(&node.node_id);
                    if let Some(true) = is_child_selected {
                        *selected_clusters.get_mut(&node.node_id).unwrap() = false;
                    }
                });
            } else {
                stabilities.get(&cluster_id).unwrap().replace(combined_child_stability);
            }
        }

        selected_clusters.into_iter()
            .filter(|(_id, should_keep)| *should_keep)
            .map(|(id, _should_keep)| id)
            .collect()
    }

    fn calc_all_stabilities(&self, n_clusters: usize, condensed_tree: &Vec<CondensedNode<T>>)
        -> BTreeMap<usize, RefCell<T>> {
        (0..n_clusters).into_iter()
            .filter(|n|
                { if !self.hyper_params.allow_single_cluster && *n == 0 { false } else { true } })
            .map(|n| self.n_samples + n)
            .map(|cluster_id|
                (cluster_id, RefCell::new(self.calc_stability(cluster_id, &condensed_tree))))
            .collect()
    }

    fn calc_stability(&self, cluster_id: usize, condensed_tree: &Vec<CondensedNode<T>>) -> T {
        let lambda_birth = self.extract_lambda_birth(cluster_id, &condensed_tree);
        condensed_tree.iter()
            .filter(|node| node.parent_node_id == cluster_id)
            .map(|node| (node.lambda_birth - lambda_birth) * T::from(node.size).unwrap_or(T::one()))
            .fold(T::zero(), std::ops::Add::add)
    }

    fn extract_lambda_birth(&self, cluster_id: usize, condensed_tree: &Vec<CondensedNode<T>>) -> T {
        if self.is_top_cluster(cluster_id) {
            T::zero()
        } else {
            condensed_tree.iter()
                .find(|node| node.node_id == cluster_id)
                .unwrap()
                .lambda_birth
        }
    }

    fn is_top_cluster(&self, cluster_id: usize) -> bool {
        cluster_id == self.n_samples
    }

    fn get_immediate_child_clusters<'b>(
        &'b self,
        cluster_id: usize,
        condensed_tree: &'b Vec<CondensedNode<T>>
    ) -> Vec<&CondensedNode<T>> {
        condensed_tree.iter()
            .filter(|node| node.parent_node_id == cluster_id)
            .filter(|node| self.is_cluster(&node.node_id))
            .collect()
    }

    fn is_cluster_too_big(&self, cluster_id: &usize, condensed_tree: &Vec<CondensedNode<T>>)
        -> bool {
        let cluster_size = condensed_tree.iter()
            .find(|node| node.node_id == *cluster_id)
            .unwrap() // The cluster has to be in the tree
            .size;
        cluster_size > self.hyper_params.max_cluster_size
    }

    fn label_data(
        &self,
        winning_clusters: &Vec<usize>,
        condensed_tree: &Vec<CondensedNode<T>>
    ) -> Vec<i32> {
        // Assume all data points are noise by default then label the ones in clusters
        let mut current_cluster_id = 0;
        let mut labels = vec![-1; self.n_samples];

        for node in condensed_tree {
            if winning_clusters.contains(&node.node_id) {
                let child_samples = self.find_child_samples(node.node_id, &condensed_tree);
                child_samples.into_iter().for_each(|id| labels[id] = current_cluster_id);
                current_cluster_id += 1;
            }
        }
        self.labels.replace(Some(labels.clone()));
        labels
    }

    fn find_child_samples(&self, root: usize, condensed_tree: &Vec<CondensedNode<T>>)
        -> Vec<usize> {
        let mut process_queue = VecDeque::from([root]);
        let mut child_nodes: Vec<usize> = Vec::new();

        while !process_queue.is_empty() {
            let current_node_num = process_queue.pop_front().unwrap();
            for node in condensed_tree {
                if node.parent_node_id == current_node_num {
                    if self.is_individual_sample(&node.node_id) {
                        child_nodes.push(node.node_id);
                    } else {
                        // Else it is a cluster not an individual data point
                        // so need to find its children
                        process_queue.push_back(node.node_id);
                    }
                }
            }
        }
        child_nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster() {
        let data: Vec<Vec<f32>> = vec![
            vec![1.5, 2.2],
            vec![1.0, 1.1],
            vec![1.2, 1.4],
            vec![0.8, 1.0],
            vec![1.1, 1.0],
            vec![3.7, 4.0],
            vec![3.9, 3.9],
            vec![3.6, 4.1],
            vec![3.8, 3.9],
            vec![4.0, 4.1],
            vec![10.0, 10.0],
        ];
        let clusterer = Hdbscan::default(&data);
        let result = clusterer.cluster().unwrap();
        assert_eq!(result, vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1]);
    }

    #[test]
    fn builder_cluster() {
        let data: Vec<Vec<f32>> = vec![
            vec![1.3, 1.1],
            vec![1.3, 1.2],
            vec![1.0, 1.1],
            vec![1.2, 1.2],
            vec![0.9, 1.0],
            vec![0.9, 1.0],
            vec![3.7, 4.0],
            vec![3.9, 3.9],
        ];
        let hyper_params = HdbscanHyperParams::builder()
            .min_cluster_size(3)
            .min_samples(2)
            .dist_metric(DistanceMetric::Manhattan)
            .build();
        let clusterer = Hdbscan::new(&data, hyper_params);
        let result = clusterer.cluster().unwrap();
        assert_eq!(result, vec![0, 0, 1, 0, 1, 1, -1, -1]);
    }

    #[test]
    fn empty_data() {
        let data: Vec<Vec<f32>> = Vec::new();
        let clusterer = Hdbscan::default(&data);
        let result = clusterer.cluster();
        assert!(matches!(result, Err(HdbscanError::EmptyDataset)));
    }

    #[test]
    fn non_finite_coordinate() {
        let data: Vec<Vec<f32>> = vec![vec![1.5, f32::infinity()]];
        let clusterer = Hdbscan::default(&data);
        let result = clusterer.cluster();
        assert!(matches!(result, Err(HdbscanError::NonFiniteCoordinate(..))));
    }

    #[test]
    fn mismatched_dimensions() {
        let data: Vec<Vec<f32>> = vec![
            vec![1.5, 2.2],
            vec![1.0, 1.1],
            vec![1.2],
        ];
        let clusterer = Hdbscan::default(&data);
        let result = clusterer.cluster();
        assert!(matches!(result, Err(HdbscanError::WrongDimension(..))));
    }

}
