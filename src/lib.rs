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
//!     clustering. The algorithm will just select the clusters that are the most persistent
//!     at all densities.
//!
//! This implementation owes a debt to the Python scikit-learn implementation of this algorithm,
//! without which this algorithm would not have been possible. The "How HDBSCAN works" article
//! below is invaluable in understanding this algorithm better.
//!
//! # Examples
//!```
//!use std::collections::HashSet;
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
//!let labels = clusterer.cluster().unwrap();
//!//First five points form one cluster
//!assert_eq!(1, labels[..5].iter().collect::<HashSet<_>>().len());
//!// Next five points are a second cluster
//!assert_eq!(1, labels[5..10].iter().collect::<HashSet<_>>().len());
//!// The final point is noise
//!assert_eq!(-1, labels[10]);
//!```
//!
//! # References
//! * [Campello, R.J.G.B.; Moulavi, D.; Sander, J. Density-based clustering based on hierarchical density estimates.](https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14)
//! * [How HDBSCAN Works](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html)

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, VecDeque};
use num_traits::Float;
use crate::core_distances::{CoreDistance, KdTree, BruteForce};
use crate::data_wrappers::{CondensedNode, MSTEdge, SLTNode};
use crate::union_find::UnionFind;

pub use crate::centers::Center;
pub use crate::distance::DistanceMetric;
pub use crate::hyper_parameters::{HdbscanHyperParams, HyperParamBuilder};
pub use crate::error::HdbscanError;
pub use crate::core_distances::NnAlgorithm;

mod centers;
mod core_distances;
mod data_wrappers;
mod distance;
mod error;
mod hyper_parameters;
mod union_find;

const BRUTE_FORCE_N_SAMPLES_LIMIT: usize = 250;

type CondensedTree<T> = Vec<CondensedNode<T>>;

/// The HDBSCAN clustering algorithm in Rust. Generic over floating point numeric types.
pub struct Hdbscan<'a, T> {
    data: &'a Vec<Vec<T>>,
    n_samples: usize,
    hp: HdbscanHyperParams,
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
    ///use hdbscan::{DistanceMetric, Hdbscan, HdbscanHyperParams, NnAlgorithm};
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
    ///    .nn_algorithm(NnAlgorithm::BruteForce)
    ///    .build();
    ///let clusterer = Hdbscan::new(&data, config);
    /// ```
    pub fn new(data: &'a Vec<Vec<T>>, hyper_params: HdbscanHyperParams) -> Self {
        let n_samples = data.len();
        Hdbscan { data, n_samples, hp: hyper_params, }
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
    ///use std::collections::HashSet;
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
    ///let labels = clusterer.cluster().unwrap();
    /// //First five points form one cluster
    ///assert_eq!(1, labels[..5].iter().collect::<HashSet<_>>().len());
    /// // Next five points are a second cluster
    ///assert_eq!(1, labels[5..10].iter().collect::<HashSet<_>>().len());
    /// // The final point is noise
    ///assert_eq!(-1, labels[10]);
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

    /// Calculates the centers of the clusters just calculate.
    ///
    /// # Parameters
    /// * `center` - the type of center to calculate. Currently only centroid (the element wise mean
    ///              of all the data points in a cluster) is supported.
    /// * `labels` - a reference to the labels calculated by a call to `Hdbscan::cluster`.
    ///
    /// # Returns
    /// * A vector of the cluster centers, of shape num clusters by num dimensions/features.
    ///
    /// # Panics
    /// * If the labels are of different length to the data passed to the `Hdbscan` constructor
    ///
    /// # Examples
    /// ```
    ///use hdbscan::{Center, Hdbscan};
    ///
    /// let data: Vec<Vec<f32>> = vec![
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
    ///let labels = clusterer.cluster().unwrap();
    ///let centroids = clusterer.calc_centers(Center::Centroid, &labels).unwrap();
    ///assert_eq!(2, centroids.len());
    ///assert!(centroids.contains(&vec![3.8, 4.0]) && centroids.contains(&vec![1.12, 1.34]));
    /// ```
    pub fn calc_centers(
        &self,
        center: Center,
        labels: &[i32]
    ) -> Result<Vec<Vec<T>>, HdbscanError> {

        assert_eq!(labels.len(), self.data.len());
        Ok(center.calc_centers(self.data, labels))
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
        let (data, k, dist_metric) = (
            self.data, self.hp.min_samples, self.hp.dist_metric);
        
        match (&self.hp.nn_algo, self.n_samples) {
            (NnAlgorithm::Auto, usize::MIN..=BRUTE_FORCE_N_SAMPLES_LIMIT) => {
                KdTree::calc_core_distances(data, k, dist_metric)
            }
            (NnAlgorithm::Auto, _) => {
                KdTree::calc_core_distances(data, k, dist_metric)
            }
            (NnAlgorithm::BruteForce, _) => {
                BruteForce::calc_core_distances(data, k, dist_metric)
            }
            (NnAlgorithm::KdTree, _) => {
                KdTree::calc_core_distances(data, k, dist_metric)
            }
        }
    }

    fn prims_min_spanning_tree(&self, core_distances: &[T]) -> Vec<MSTEdge<T>> {
        let mut in_tree = vec![false; self.n_samples];
        let mut distances = vec![T::infinity(); self.n_samples];
        distances[0] = T::zero();

        let mut mst = Vec::with_capacity(self.n_samples);

        let mut left_node_id = 0;
        let mut right_node_id = 0;

        for _ in 1..self.n_samples {
            in_tree[left_node_id] = true;
            let mut current_min_dist = T::infinity();

            for i in 0..self.n_samples {
                if in_tree[i] {
                    continue;
                }
                let mrd = self.calc_mutual_reachability_dist(left_node_id, i, core_distances);
                if mrd < distances[i] {
                    distances[i] = mrd;
                }
                if distances[i] < current_min_dist {
                    right_node_id = i;
                    current_min_dist = distances[i];
                }
            }
            mst.push(MSTEdge { left_node_id, right_node_id, distance: current_min_dist });
            left_node_id = right_node_id;
        }
        self.sort_mst_by_dist(&mut mst);
        mst
    }

    fn calc_mutual_reachability_dist(&self, a: usize, b: usize, core_distances: &[T]) -> T {
        let core_dist_a = core_distances[a];
        let core_dist_b = core_distances[b];
        let dist_a_b = self.hp.dist_metric.calc_dist(&self.data[a], &self.data[b]);

        core_dist_a.max(core_dist_b).max(dist_a_b)
    }

    fn sort_mst_by_dist(&self, min_spanning_tree: &mut [MSTEdge<T>]) {
        min_spanning_tree.sort_by(|a, b|
            a.distance.partial_cmp(&b.distance).expect("Invalid floats"));
    }

    fn make_single_linkage_tree(&self, min_spanning_tree: &[MSTEdge<T>]) -> Vec<SLTNode<T>> {
        let mut single_linkage_tree: Vec<SLTNode<T>> = Vec::with_capacity(self.n_samples - 1);

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

    fn condense_tree(&self, single_linkage_tree: &[SLTNode<T>]) -> CondensedTree<T> {
        let top_node = (self.n_samples - 1) * 2;
        let node_ids = self.find_slt_children_breadth_first(single_linkage_tree, top_node);

        let mut new_node_ids = vec![0_usize; top_node + 1];
        new_node_ids[top_node] = self.n_samples;
        let mut next_parent_id = self.n_samples + 1;

        let mut visited = vec![false; node_ids.len()];
        let mut condensed_tree = Vec::new();

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

            match (is_left_a_cluster, is_right_a_cluster) {
                (true, true) => {
                    for (child_id, child_size) in [left_child_id, right_child_id].iter()
                        .zip([left_child_size, right_child_size]) {
                        new_node_ids[*child_id] = next_parent_id;
                        next_parent_id += 1;
                        condensed_tree.push(CondensedNode {
                            node_id: new_node_ids[*child_id],
                            parent_node_id: new_node_ids[node_id],
                            lambda_birth,
                            size: child_size
                        });
                    }
                }
                (false, false) => {
                    let new_node_id = new_node_ids[node_id];
                    self.add_children_to_tree(
                        left_child_id, new_node_id, &single_linkage_tree,
                        &mut condensed_tree, &mut visited, lambda_birth);
                    self.add_children_to_tree(
                        right_child_id, new_node_id, &single_linkage_tree,
                        &mut condensed_tree, &mut visited, lambda_birth);

                }
                (false, true) => {
                    new_node_ids[right_child_id] = new_node_ids[node_id];
                    self.add_children_to_tree(
                        left_child_id, new_node_ids[node_id], &single_linkage_tree,
                        &mut condensed_tree, &mut visited, lambda_birth);

                }
                (true, false) => {
                    new_node_ids[left_child_id] = new_node_ids[node_id];
                    self.add_children_to_tree(
                        right_child_id, new_node_ids[node_id], &single_linkage_tree,
                        &mut condensed_tree, &mut visited, lambda_birth);

                }
            }
        }
        condensed_tree
    }

    fn find_slt_children_breadth_first(
        &self,
        single_linkage_tree: &[SLTNode<T>],
        root: usize
    ) -> Vec<usize> {

        let mut process_queue = VecDeque::from([root]);
        let mut child_nodes = Vec::new();

        while !process_queue.is_empty() {
            let mut current_node_id = match process_queue.pop_front() {
                Some(node_id) => node_id,
                None => break,
            };
            child_nodes.push(current_node_id);
            if self.is_individual_sample(&current_node_id) {
                continue;
            }
            current_node_id -= self.n_samples;
            let current_node = &single_linkage_tree[current_node_id];
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

    fn extract_cluster_size(&self, node_id: usize, single_linkage_tree: &[SLTNode<T>]) -> usize {
        if self.is_individual_sample(&node_id) {
            1
        } else {
            single_linkage_tree[node_id - self.n_samples].size
        }
    }

    fn is_cluster_big_enough(&self, cluster_size: usize) -> bool {
        cluster_size >= self.hp.min_cluster_size
    }

    fn add_children_to_tree(
        &self,
        node_id: usize,
        new_node_id: usize,
        single_linkage_tree: &[SLTNode<T>],
        condensed_tree: &mut CondensedTree<T>,
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

    fn extract_winning_clusters(&self, condensed_tree: &CondensedTree<T>) -> Vec<usize> {
        let n_clusters = condensed_tree.len() - self.n_samples + 1;
        let stabilities = self.calc_all_stabilities(n_clusters, &condensed_tree);
        let mut selected_clusters: HashMap<usize, bool> =
            stabilities.keys().map(|id| (id.clone(), false)).collect();

        for (cluster_id, stability) in stabilities.iter().rev() {
            let combined_child_stability =
                self.get_immediate_child_clusters(*cluster_id, &condensed_tree)
                    .iter()
                    .map(|node| {
                        stabilities.get(&node.node_id)
                            .unwrap_or(&RefCell::new(T::zero())).borrow().clone()
                    })
                    .fold(T::zero(), std::ops::Add::add);

            if *stability.borrow() > combined_child_stability
                && !self.is_cluster_too_big(cluster_id, condensed_tree) {
                *selected_clusters
                    .get_mut(&cluster_id)
                    .expect("Couldn't retrieve stability") = true;

                // If child clusters were already marked as winning clusters reverse
                self.find_child_clusters(&cluster_id, &condensed_tree).iter().for_each(|node_id| {
                    let is_child_selected = selected_clusters.get(node_id);
                    if let Some(true) = is_child_selected {
                        *selected_clusters
                            .get_mut(node_id)
                            .expect("Couldn't retrieve stability") = false;
                    }
                });
            } else {
                stabilities
                    .get(&cluster_id)
                    .expect("Couldn't retrieve stability")
                    .replace(combined_child_stability);
            }
        }

        selected_clusters.into_iter()
            .filter(|(_id, should_keep)| *should_keep)
            .map(|(id, _should_keep)| id)
            .collect()
    }

    fn calc_all_stabilities(
        &self,
        n_clusters: usize,
        condensed_tree: &CondensedTree<T>
    ) -> BTreeMap<usize, RefCell<T>> {

        (0..n_clusters).into_iter()
            .filter(|&n| { 
                if !self.hp.allow_single_cluster && n == 0 { false } else { true } 
            })
            .map(|n| self.n_samples + n)
            .map(|cluster_id| (
                cluster_id, RefCell::new(self.calc_stability(cluster_id, &condensed_tree))
            ))
            .collect()
    }

    fn calc_stability(&self, cluster_id: usize, condensed_tree: &CondensedTree<T>) -> T {
        let lambda_birth = self.extract_lambda_birth(cluster_id, &condensed_tree);
        condensed_tree.iter()
            .filter(|node| node.parent_node_id == cluster_id)
            .map(|node| (node.lambda_birth - lambda_birth) * T::from(node.size).unwrap_or(T::one()))
            .fold(T::zero(), std::ops::Add::add)
    }

    fn extract_lambda_birth(&self, cluster_id: usize, condensed_tree: &CondensedTree<T>) -> T {
        if self.is_top_cluster(&cluster_id) {
            T::zero()
        } else {
            condensed_tree.iter()
                .find(|node| node.node_id == cluster_id)
                .map(|node| node.lambda_birth)
                .unwrap_or(T::zero())
        }
    }

    fn is_top_cluster(&self, cluster_id: &usize) -> bool {
        cluster_id == &self.n_samples
    }

    fn get_immediate_child_clusters<'b>(
        &'b self,
        cluster_id: usize,
        condensed_tree: &'b CondensedTree<T>
    ) -> Vec<&CondensedNode<T>> {
        condensed_tree.iter()
            .filter(|node| node.parent_node_id == cluster_id)
            .filter(|node| self.is_cluster(&node.node_id))
            .collect()
    }

    fn is_cluster_too_big(&self, cluster_id: &usize, condensed_tree: &CondensedTree<T>) -> bool {
        self.get_cluster_size(cluster_id, &condensed_tree) > self.hp.max_cluster_size
    }

    fn get_cluster_size(&self, cluster_id: &usize, condensed_tree: &CondensedTree<T>) -> usize {
        if self.hp.allow_single_cluster && self.is_top_cluster(cluster_id) {
            condensed_tree.iter()
                .filter(|node| self.is_cluster(&node.node_id))
                .filter(|node| &node.parent_node_id == cluster_id)
                .map(|node| node.size)
                .sum()

        } else {
            // All other clusters are in the tree with sizes
            condensed_tree.iter()
                .find(|node| &node.node_id == cluster_id)
                .map(|node| node.size)
                .unwrap_or(1usize)  // The cluster has to be in the tree
        }
    }

    fn find_child_clusters(
        &self,
        root_node_id: &usize,
        condensed_tree: &CondensedTree<T>
    ) -> Vec<usize> {

        let mut process_queue = VecDeque::from([root_node_id]);
        let mut child_clusters= Vec::new();

        while !process_queue.is_empty() {
            let current_node_id = match process_queue.pop_front() {
                Some(node_id) => node_id,
                None => break,
            };

            for node in condensed_tree {
                if self.is_individual_sample(&node.node_id) {
                    continue;
                }
                if node.parent_node_id == *current_node_id {
                    child_clusters.push(node.node_id);
                    process_queue.push_back(&node.node_id);
                }
            }
        }
        child_clusters
    }

    fn label_data(
        &self,
        winning_clusters: &Vec<usize>,
        condensed_tree: &CondensedTree<T>
    ) -> Vec<i32> {

        // Assume all data points are noise by default then label the ones in clusters
        let mut current_cluster_id = 0;
        let mut labels = vec![-1; self.n_samples];

        for cluster_id in winning_clusters {
            let node_size = self.get_cluster_size(cluster_id, &condensed_tree);
            self.find_child_samples(*cluster_id, node_size, &condensed_tree).into_iter()
                .for_each(|id| labels[id] = current_cluster_id);
            current_cluster_id += 1;
        }
        labels
    }

    fn find_child_samples(
        &self,
        root_node_id: usize,
        node_size: usize,
        condensed_tree: &CondensedTree<T>
    ) -> Vec<usize> {

        let mut process_queue = VecDeque::from([root_node_id]);
        let mut child_nodes = Vec::with_capacity(node_size);

        while !process_queue.is_empty() {
            let current_node_id = match process_queue.pop_front() {
                Some(node_id) => node_id,
                None => break,
            };
            for node in condensed_tree {
                if node.parent_node_id == current_node_id {
                    if self.is_individual_sample(&node.node_id) {
                        if self.hp.allow_single_cluster
                            && self.is_top_cluster(&current_node_id) {
                            continue;
                        }
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
    use std::collections::HashSet;
    use super::*;

    #[test]
    fn cluster() {
        let data = cluster_test_data();
        let clusterer = Hdbscan::default(&data);
        let result = clusterer.cluster().unwrap();
        // First five points form one cluster
        assert_eq!(1, result[..5].iter().collect::<HashSet<_>>().len());
        // Next five points are a second cluster
        assert_eq!(1, result[5..10].iter().collect::<HashSet<_>>().len());
        // The final point is noise
        assert_eq!(-1, result[10]);
    }

    #[test]
    fn builder_cluster() {
        let data: Vec<Vec<f32>> = vec![
            vec![1.3, 1.1],
            vec![1.3, 1.2],
            vec![1.2, 1.2],
            vec![1.0, 1.1],
            vec![0.9, 1.0],
            vec![0.9, 1.0],
            vec![3.7, 4.0],
        ];
        let hyper_params = HdbscanHyperParams::builder()
            .min_cluster_size(3)
            .min_samples(2)
            .dist_metric(DistanceMetric::Manhattan)
            .nn_algorithm(NnAlgorithm::BruteForce)
            .build();
        let clusterer = Hdbscan::new(&data, hyper_params);
        let result = clusterer.cluster().unwrap();
        // First three points form one cluster
        assert_eq!(1, result[..3].iter().collect::<HashSet<_>>().len());
        // Next three points are a second cluster
        assert_eq!(1, result[3..6].iter().collect::<HashSet<_>>().len());
        // The final point is noise
        assert_eq!(-1, result[6]);
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

    #[test]
    fn calc_centers() {
        let data = cluster_test_data();
        let clusterer = Hdbscan::default(&data);
        let labels = clusterer.cluster().unwrap();
        let centroids = clusterer.calc_centers(Center::Centroid, &labels).unwrap();
        assert_eq!(2, centroids.len());
        assert!(centroids.contains(&vec![3.8, 4.0]) && centroids.contains(&vec![1.12, 1.34]));
    }

    fn cluster_test_data() -> Vec<Vec<f32>> {
        vec![
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
        ]
    }
}
