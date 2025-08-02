#[cfg(feature = "parallel")]
use crate::core_distances::parallel::CoreDistanceCalculatorPar;
#[cfg(feature = "serial")]
use crate::core_distances::serial::CoreDistanceCalculator;
use crate::data_wrappers::{CondensedNode, MSTEdge, SLTNode};
use crate::union_find::UnionFind;
use crate::validation::DataValidator;
use crate::{distance, Center, DistanceMetric, HdbscanError, HdbscanHyperParams};
use num_traits::Float;
use std::collections::{HashMap, VecDeque};
use std::ops::Range;

type CondensedTree<T> = Vec<CondensedNode<T>>;

/// The HDBSCAN clustering algorithm in Rust. Generic over floating point numeric types.
#[derive(Debug, Clone, PartialEq)]
pub struct Hdbscan<'a, T> {
    data: &'a [Vec<T>],
    n_samples: usize,
    hp: HdbscanHyperParams,
}

impl<'a, T: Float + Send + Sync> Hdbscan<'a, T> {
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
    pub fn new(data: &'a [Vec<T>], hyper_params: HdbscanHyperParams) -> Self {
        let n_samples = data.len();
        Hdbscan {
            data,
            n_samples,
            hp: hyper_params,
        }
    }

    #[deprecated(
        since = "0.8.1",
        note = "Please use `default_hyper_params` constructor instead"
    )]
    pub fn default(data: &'a [Vec<T>]) -> Hdbscan<'a, T> {
        let hyper_params = HdbscanHyperParams::default();
        Hdbscan::new(data, hyper_params)
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
    ///let clusterer = Hdbscan::default_hyper_params(&data);
    /// ```
    pub fn default_hyper_params(data: &'a [Vec<T>]) -> Hdbscan<'a, T> {
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
    ///let clusterer = Hdbscan::default_hyper_params(&data);
    ///let labels = clusterer.cluster().unwrap();
    /// //First five points form one cluster
    ///assert_eq!(1, labels[..5].iter().collect::<HashSet<_>>().len());
    /// // Next five points are a second cluster
    ///assert_eq!(1, labels[5..10].iter().collect::<HashSet<_>>().len());
    /// // The final point is noise
    ///assert_eq!(-1, labels[10]);
    /// ```
    #[cfg(feature = "serial")]
    pub fn cluster(&self) -> Result<Vec<i32>, HdbscanError> {
        let validator = DataValidator::new(self.data, &self.hp);
        validator.validate_input_data()?;
        let calculator = CoreDistanceCalculator::new(self.data, &self.hp);
        let core_distances = calculator.calc_core_distances();
        let min_spanning_tree = self.prims_min_spanning_tree(&core_distances);
        let single_linkage_tree = self.make_single_linkage_tree(&min_spanning_tree);
        let condensed_tree = self.condense_tree(&single_linkage_tree);
        let winning_clusters = self.extract_winning_clusters(&condensed_tree);
        let labelled_data = self.label_data(&winning_clusters, &condensed_tree);
        Ok(labelled_data)
    }

    /// Performs clustering on the list of vectors passed to the constructor, with some parallelism.
    /// Not recommended for small or low dimension datasets.
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
    ///let clusterer = Hdbscan::default_hyper_params(&data);
    ///let labels = clusterer.cluster_par().unwrap();
    /// //First five points form one cluster
    ///assert_eq!(1, labels[..5].iter().collect::<HashSet<_>>().len());
    /// // Next five points are a second cluster
    ///assert_eq!(1, labels[5..10].iter().collect::<HashSet<_>>().len());
    /// // The final point is noise
    ///assert_eq!(-1, labels[10]);
    /// ```
    #[cfg(feature = "parallel")]
    pub fn cluster_par(&self) -> Result<Vec<i32>, HdbscanError> {
        let validator = DataValidator::new(self.data, &self.hp);
        validator.validate_input_data()?;
        let calculator = CoreDistanceCalculatorPar::new(self.data, &self.hp);
        let core_distances = calculator.calc_core_distances();
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
    /// * `center` - the type of center to calculate.
    /// * `labels` - a reference to the labels calculated by a call to `Hdbscan::cluster`.
    ///
    /// # Returns
    /// * A vector of the cluster centers, of shape num clusters by num dimensions/features. The
    ///   index of the centroid is the cluster label. For example, the centroid cluster of label 0
    ///   will be the first centroid in the vector of centroids.
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
    ///let clusterer = Hdbscan::default_hyper_params(&data);
    ///let labels = clusterer.cluster().unwrap();
    ///let centroids = clusterer.calc_centers(Center::Centroid, &labels).unwrap();
    ///assert_eq!(2, centroids.len());
    ///assert!(centroids.contains(&vec![3.8, 4.0]) && centroids.contains(&vec![1.12, 1.34]));
    /// ```
    pub fn calc_centers(
        &self,
        center: Center,
        labels: &[i32],
    ) -> Result<Vec<Vec<T>>, HdbscanError> {
        if labels.len() != self.data.len() {
            return Err(HdbscanError::WrongDimension(String::from(
                "The length of the labels must equal the length of the original clustering data.",
            )));
        }
        if self.hp.dist_metric != DistanceMetric::Haversine && center == Center::GeoCentroid {
            // TODO: Implement a more appropriate error variant when doing a major version bump
            return Err(HdbscanError::WrongDimension(String::from(
                "Geographical centroids can only be used with geographical coordinates.",
            )));
        }
        if self.hp.dist_metric == DistanceMetric::Precalculated {
            // TODO: Implement a more appropriate error variant when doing a major version bump
            return Err(HdbscanError::WrongDimension(String::from(
                "Centroids cannot be calculated when using precalculated distances.",
            )));
        }

        Ok(center.calc_centers(
            self.data,
            labels,
            distance::get_dist_func(&self.hp.dist_metric),
        ))
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
            mst.push(MSTEdge {
                left_node_id,
                right_node_id,
                distance: current_min_dist,
            });
            left_node_id = right_node_id;
        }
        self.sort_mst_by_dist(&mut mst);
        mst
    }

    fn calc_mutual_reachability_dist(&self, a: usize, b: usize, core_distances: &[T]) -> T {
        let core_dist_a = core_distances[a];
        let core_dist_b = core_distances[b];
        let dist_a_b = if self.hp.dist_metric == DistanceMetric::Precalculated {
            self.data[a][b]
        } else {
            self.hp.dist_metric.calc_dist(&self.data[a], &self.data[b])
        };

        core_dist_a.max(core_dist_b).max(dist_a_b)
    }

    fn sort_mst_by_dist(&self, min_spanning_tree: &mut [MSTEdge<T>]) {
        min_spanning_tree
            .sort_by(|a, b| a.distance.partial_cmp(&b.distance).expect("Invalid floats"));
    }

    fn make_single_linkage_tree(&self, min_spanning_tree: &[MSTEdge<T>]) -> Vec<SLTNode<T>> {
        let mut single_linkage_tree: Vec<SLTNode<T>> = Vec::with_capacity(self.n_samples - 1);

        let mut union_find = UnionFind::new(self.n_samples);

        for mst_edge in min_spanning_tree.iter().take(self.n_samples - 1) {
            let left_node = mst_edge.left_node_id;
            let right_node = mst_edge.right_node_id;
            let distance = mst_edge.distance;

            let left_child = union_find.find(left_node);
            let right_child = union_find.find(right_node);
            let size = union_find.size_of(left_child) + union_find.size_of(right_child);

            single_linkage_tree.push(SLTNode {
                left_child,
                right_child,
                distance,
                size,
            });

            union_find.union(left_child, right_child);
        }

        single_linkage_tree
    }

    fn condense_tree(&self, single_linkage_tree: &[SLTNode<T>]) -> CondensedTree<T> {
        let top_node = (self.n_samples - 1) * 2;
        let node_ids = self.find_single_linkage_children(single_linkage_tree, top_node);

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
                    for (child_id, child_size) in [left_child_id, right_child_id]
                        .iter()
                        .zip([left_child_size, right_child_size])
                    {
                        new_node_ids[*child_id] = next_parent_id;
                        next_parent_id += 1;
                        condensed_tree.push(CondensedNode {
                            node_id: new_node_ids[*child_id],
                            parent_node_id: new_node_ids[node_id],
                            lambda_birth,
                            size: child_size,
                        });
                    }
                }
                (false, false) => {
                    let new_node_id = new_node_ids[node_id];
                    self.add_children_to_tree(
                        left_child_id,
                        new_node_id,
                        single_linkage_tree,
                        &mut condensed_tree,
                        &mut visited,
                        lambda_birth,
                    );
                    self.add_children_to_tree(
                        right_child_id,
                        new_node_id,
                        single_linkage_tree,
                        &mut condensed_tree,
                        &mut visited,
                        lambda_birth,
                    );
                }
                (false, true) => {
                    new_node_ids[right_child_id] = new_node_ids[node_id];
                    self.add_children_to_tree(
                        left_child_id,
                        new_node_ids[node_id],
                        single_linkage_tree,
                        &mut condensed_tree,
                        &mut visited,
                        lambda_birth,
                    );
                }
                (true, false) => {
                    new_node_ids[left_child_id] = new_node_ids[node_id];
                    self.add_children_to_tree(
                        right_child_id,
                        new_node_ids[node_id],
                        single_linkage_tree,
                        &mut condensed_tree,
                        &mut visited,
                        lambda_birth,
                    );
                }
            }
        }
        condensed_tree
    }

    fn find_single_linkage_children(
        &self,
        single_linkage_tree: &[SLTNode<T>],
        root: usize,
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
        !self.is_individual_sample(node_id)
    }

    fn calc_lambda(&self, dist: T) -> T {
        if dist > T::zero() {
            T::one() / dist
        } else {
            T::infinity()
        }
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
        visited: &mut [bool],
        lambda_birth: T,
    ) {
        for child_id in self.find_single_linkage_children(single_linkage_tree, node_id) {
            if self.is_individual_sample(&child_id) {
                condensed_tree.push(CondensedNode {
                    node_id: child_id,
                    parent_node_id: new_node_id,
                    lambda_birth,
                    size: 1,
                })
            }
            visited[child_id] = true
        }
    }

    fn extract_winning_clusters(&self, condensed_tree: &CondensedTree<T>) -> Vec<usize> {
        let (lower, upper) = self.get_cluster_id_bounds(condensed_tree);
        let n_clusters = upper - lower;

        let mut stabilities = self.calc_all_stabilities(lower..upper, condensed_tree);
        let mut clusters: HashMap<usize, bool> =
            stabilities.keys().map(|id| (*id, false)).collect();

        for cluster_id in (lower..upper).rev() {
            let stability = stabilities
                .get(&cluster_id)
                .expect("Couldn't retrieve stability");
            let combined_child_stability = self
                .get_immediate_child_clusters(cluster_id, condensed_tree)
                .iter()
                .map(|node| *stabilities.get(&node.node_id).unwrap_or(&T::zero()))
                .fold(T::zero(), std::ops::Add::add);

            if stability > &combined_child_stability
                && !self.is_cluster_too_big(&cluster_id, condensed_tree)
            {
                clusters.insert(cluster_id, true);

                // If child clusters were already marked as winning clusters reverse
                self.find_child_clusters(&cluster_id, condensed_tree)
                    .iter()
                    .for_each(|node_id| {
                        let is_child_selected = clusters.get(node_id);
                        if let Some(true) = is_child_selected {
                            clusters.insert(*node_id, false);
                        }
                    });
            } else {
                stabilities.insert(cluster_id, combined_child_stability);
            }
        }

        let mut selected_cluster_ids = clusters
            .into_iter()
            .filter(|(_id, should_keep)| *should_keep)
            .map(|(id, _should_keep)| id)
            .collect();

        if self.hp.epsilon != 0.0 && n_clusters > 0 {
            selected_cluster_ids =
                self.check_cluster_epsilons(selected_cluster_ids, condensed_tree);
        }

        selected_cluster_ids.sort();
        selected_cluster_ids
    }

    fn get_cluster_id_bounds(&self, condensed_tree: &CondensedTree<T>) -> (usize, usize) {
        if self.hp.allow_single_cluster {
            let n_clusters = condensed_tree.len() - self.n_samples + 1;
            (self.n_samples, self.n_samples + n_clusters)
        } else {
            let lower = self.n_samples + 1;
            let n_clusters = condensed_tree.len() - self.n_samples;
            (lower, lower + n_clusters)
        }
    }

    fn calc_all_stabilities(
        &self,
        cluster_id_range: Range<usize>,
        condensed_tree: &CondensedTree<T>,
    ) -> HashMap<usize, T> {
        cluster_id_range
            .map(|cluster_id| (cluster_id, self.calc_stability(cluster_id, condensed_tree)))
            .collect()
    }

    fn calc_stability(&self, cluster_id: usize, condensed_tree: &CondensedTree<T>) -> T {
        let lambda_birth = self.extract_lambda_birth(cluster_id, condensed_tree);
        condensed_tree
            .iter()
            .filter(|node| node.parent_node_id == cluster_id)
            .map(|node| (node.lambda_birth - lambda_birth) * T::from(node.size).unwrap_or(T::one()))
            .fold(T::zero(), std::ops::Add::add)
    }

    fn extract_lambda_birth(&self, cluster_id: usize, condensed_tree: &CondensedTree<T>) -> T {
        if self.is_top_cluster(&cluster_id) {
            T::zero()
        } else {
            condensed_tree
                .iter()
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
        condensed_tree: &'b CondensedTree<T>,
    ) -> Vec<&'b CondensedNode<T>> {
        condensed_tree
            .iter()
            .filter(|node| node.parent_node_id == cluster_id)
            .filter(|node| self.is_cluster(&node.node_id))
            .collect()
    }

    fn is_cluster_too_big(&self, cluster_id: &usize, condensed_tree: &CondensedTree<T>) -> bool {
        self.get_cluster_size(cluster_id, condensed_tree) > self.hp.max_cluster_size
    }

    fn get_cluster_size(&self, cluster_id: &usize, condensed_tree: &CondensedTree<T>) -> usize {
        if self.hp.allow_single_cluster && self.is_top_cluster(cluster_id) {
            condensed_tree
                .iter()
                .filter(|node| self.is_cluster(&node.node_id))
                .filter(|node| &node.parent_node_id == cluster_id)
                .map(|node| node.size)
                .sum()
        } else {
            // All other clusters are in the tree with sizes
            condensed_tree
                .iter()
                .find(|node| &node.node_id == cluster_id)
                .map(|node| node.size)
                .unwrap_or(1usize) // The cluster has to be in the tree
        }
    }

    fn find_child_clusters(
        &self,
        root_node_id: &usize,
        condensed_tree: &CondensedTree<T>,
    ) -> Vec<usize> {
        let mut process_queue = VecDeque::from([root_node_id]);
        let mut child_clusters = Vec::new();

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

    fn check_cluster_epsilons(
        &self,
        winning_clusters: Vec<usize>,
        condensed_tree: &CondensedTree<T>,
    ) -> Vec<usize> {
        let epsilon = T::from(self.hp.epsilon).expect("Couldn't convert f64 epsilon to T");
        let mut processed: Vec<usize> = Vec::new();
        let mut winning_epsilon_clusters = Vec::new();

        for cluster_id in winning_clusters.iter() {
            let cluster_epsilon = self.calc_cluster_epsilon(*cluster_id, condensed_tree, epsilon);

            if cluster_epsilon < epsilon {
                if processed.contains(cluster_id) {
                    continue;
                }
                let winning_cluster_id =
                    self.find_higher_node_sufficient_epsilon(*cluster_id, condensed_tree, epsilon);
                winning_epsilon_clusters.push(winning_cluster_id);

                for sub_node in self.find_child_clusters(&winning_cluster_id, condensed_tree) {
                    if sub_node != winning_cluster_id {
                        processed.push(sub_node)
                    }
                }
            } else {
                winning_epsilon_clusters.push(*cluster_id);
            }
        }
        winning_epsilon_clusters
    }

    fn find_higher_node_sufficient_epsilon(
        &self,
        starting_cluster_id: usize,
        condensed_tree: &CondensedTree<T>,
        epsilon: T,
    ) -> usize {
        let mut current_id = starting_cluster_id;
        let winning_cluster_id;
        loop {
            let parent_id = condensed_tree
                .iter()
                .find(|node| node.node_id == current_id)
                .map(|node| node.parent_node_id)
                // If the node isn't in the tree there must be only a single root cluster as
                // this isn't stored explicitly in the tree. Its id is always max node id + 1
                .unwrap_or(self.n_samples);
            if self.is_top_cluster(&parent_id) {
                if self.hp.allow_single_cluster {
                    winning_cluster_id = parent_id;
                } else {
                    winning_cluster_id = current_id;
                }
                break;
            }

            let parent_epsilon = self.calc_cluster_epsilon(parent_id, condensed_tree, epsilon);
            if parent_epsilon > epsilon {
                winning_cluster_id = parent_id;
                break;
            }
            current_id = parent_id;
        }
        winning_cluster_id
    }

    fn calc_cluster_epsilon(
        &self,
        cluster_id: usize,
        condensed_tree: &CondensedTree<T>,
        epsilon: T,
    ) -> T {
        let cluster_lambda = condensed_tree
            .iter()
            .find(|node| node.node_id == cluster_id)
            .map(|node| node.lambda_birth);
        match cluster_lambda {
            Some(lambda) => T::one() / lambda,
            // Should be unreachable, but set to a value that will skip the cluster
            None => epsilon - T::one(),
        }
    }

    fn label_data(
        &self,
        winning_clusters: &[usize],
        condensed_tree: &CondensedTree<T>,
    ) -> Vec<i32> {
        // Assume all data points are noise by default then label the ones in clusters
        let mut labels = vec![-1; self.n_samples];
        let n_clusters = winning_clusters.len();

        for (current_cluster_id, cluster_id) in winning_clusters.iter().enumerate() {
            let node_size = self.get_cluster_size(cluster_id, condensed_tree);
            self.find_child_samples(*cluster_id, node_size, n_clusters, condensed_tree)
                .into_iter()
                .for_each(|id| labels[id] = current_cluster_id as i32);
        }
        labels
    }

    fn find_child_samples(
        &self,
        root_node_id: usize,
        node_size: usize,
        n_clusters: usize,
        condensed_tree: &CondensedTree<T>,
    ) -> Vec<usize> {
        let mut process_queue = VecDeque::from([root_node_id]);
        let mut child_nodes = Vec::with_capacity(node_size);

        while !process_queue.is_empty() {
            let current_node_id = match process_queue.pop_front() {
                Some(node_id) => node_id,
                None => break,
            };
            for node in condensed_tree {
                // Skip nodes that aren't the child of this one
                if node.parent_node_id != current_node_id {
                    continue;
                }
                // If node is a cluster, then its children need processing
                if self.is_cluster(&node.node_id) {
                    process_queue.push_back(node.node_id);
                    continue;
                }
                // Finally, handle individual data points
                if n_clusters == 1 && self.hp.allow_single_cluster {
                    let lambda_threshold = self.get_lambda_threshold(root_node_id, condensed_tree);
                    let node_lambda = self.extract_lambda_birth(node.node_id, condensed_tree);
                    if node_lambda >= lambda_threshold {
                        child_nodes.push(node.node_id)
                    }
                } else if self.hp.allow_single_cluster && self.is_top_cluster(&current_node_id) {
                    continue;
                } else {
                    child_nodes.push(node.node_id);
                }
            }
        }
        child_nodes
    }

    fn get_lambda_threshold(&self, root_node_id: usize, condensed_tree: &CondensedTree<T>) -> T {
        if self.hp.epsilon == 0.0 {
            condensed_tree
                .iter()
                .filter(|node| node.parent_node_id == root_node_id)
                .map(|node| node.lambda_birth)
                .fold(None, |max, lambda| match max {
                    None => Some(lambda),
                    Some(max_lambda) => Some(if lambda > max_lambda {
                        lambda
                    } else {
                        max_lambda
                    }),
                })
                .unwrap_or(T::zero())
        } else {
            T::from(1.0 / self.hp.epsilon).unwrap()
        }
    }
}
