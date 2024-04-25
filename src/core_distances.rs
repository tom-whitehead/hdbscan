use num_traits::Float;
use crate::{distance, DistanceMetric};

/// The nearest neighbour algorithm options
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
        data: &Vec<Vec<T>>, k: usize, dist_metric: DistanceMetric) -> Vec<T>;
}

pub(crate) struct BruteForce {}

impl CoreDistance for BruteForce {
    fn calc_core_distances<T: Float>(
        data: &Vec<Vec<T>>, 
        k: usize, 
        dist_metric: DistanceMetric
    ) -> Vec<T> {
        
        let n_samples = data.len();
        let dist_matrix = calc_pairwise_distances(&data, distance::get_dist_func(&dist_metric));
        let mut core_distances = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let mut distances = dist_matrix[i].to_owned();
            distances.sort_by(|a, b| a.partial_cmp(&b).unwrap());
            core_distances.push(distances[k - 1]);
        }

        core_distances
    }
}

fn calc_pairwise_distances<T, F>(data: &Vec<Vec<T>>, dist_func: F) -> Vec<Vec<T>>
where
    T: Float,
    F: Fn(&[T], &[T]) -> T
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

pub(crate) struct KdTree {}

impl CoreDistance for KdTree {
    fn calc_core_distances<T: Float>(
        data: &Vec<Vec<T>>,
        k: usize,
        dist_metric: DistanceMetric
    ) -> Vec<T> {
        
        let mut tree: kdtree::KdTree<T, usize, &Vec<T>> = kdtree::KdTree::new(data[0].len());
        data.iter().enumerate()
            .for_each(|(n, datapoint)| tree.add(datapoint, n).unwrap());

        let dist_func = distance::get_dist_func(&dist_metric);
        data.iter()
            .map(|datapoint| {
                let result = tree.nearest(datapoint, k, &dist_func).unwrap();
                result.into_iter()
                    .map(|(dist, _idx)| dist)
                    .last()
                    .unwrap()
            })
            .collect()
    }
}
