use crate::{distance, DistanceMetric};
use num_traits::Float;

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

pub(crate) trait CoreDistance {
    fn calc_core_distances<T: Float>(
        data: &[Vec<T>],
        k: usize,
        dist_metric: DistanceMetric,
    ) -> Vec<T>;
}

pub(crate) struct BruteForce;

impl CoreDistance for BruteForce {
    fn calc_core_distances<T: Float>(
        data: &[Vec<T>],
        k: usize,
        dist_metric: DistanceMetric,
    ) -> Vec<T> {
        let dist_matrix = calc_pairwise_distances(data, distance::get_dist_func(&dist_metric));
        get_core_distances_from_matrix(&dist_matrix, k)
    }
}

fn calc_pairwise_distances<T, F>(data: &[Vec<T>], dist_func: F) -> Vec<Vec<T>>
where
    T: Float,
    F: Fn(&[T], &[T]) -> T,
{
    let n_samples = data.len();
    let mut dist_matrix = vec![vec![T::nan(); n_samples]; n_samples];

    for i in 0..n_samples {
        for j in 0..n_samples {
            let a = &data[i];
            let b = &data[j];
            dist_matrix[i][j] = dist_func(a, b);
        }
    }
    dist_matrix
}

pub(crate) fn get_core_distances_from_matrix<T: Float>(dist_matrix: &[Vec<T>], k: usize) -> Vec<T> {
    let n_samples = dist_matrix.len();
    let mut core_distances = Vec::with_capacity(n_samples);

    for distances in dist_matrix.iter().take(n_samples) {
        let mut distances = distances.clone();
        distances.sort_by(|a, b| a.partial_cmp(b).expect("Invalid float"));
        core_distances.push(distances[k - 1]);
    }

    core_distances
}

pub(crate) struct KdTree;

impl CoreDistance for KdTree {
    fn calc_core_distances<T: Float>(
        data: &[Vec<T>],
        k: usize,
        dist_metric: DistanceMetric,
    ) -> Vec<T> {
        let mut tree: kdtree::KdTree<T, usize, &Vec<T>> = kdtree::KdTree::new(data[0].len());
        data.iter()
            .enumerate()
            .for_each(|(n, datapoint)| tree.add(datapoint, n).expect("Failed to add to KdTree"));

        let dist_func = distance::get_dist_func(&dist_metric);
        data.iter()
            .map(|datapoint| {
                let result = tree
                    .nearest(datapoint, k, &dist_func)
                    .expect("Failed to find neighbours");
                result
                    .into_iter()
                    .map(|(dist, _idx)| dist)
                    .last()
                    .expect("Failed to find neighbours")
            })
            .collect()
    }
}
