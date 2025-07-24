use super::{NnAlgorithm, BRUTE_FORCE_N_SAMPLES_LIMIT};
use crate::distance::{get_dist_func, DistanceMetric};
use crate::HdbscanHyperParams;
use num_traits::Float;

pub(crate) struct CoreDistanceCalculator<'a, T> {
    data: &'a [Vec<T>],
    nn_algo: NnAlgorithm,
    dist_metric: DistanceMetric,
    k: usize,
}

impl<'a, T: Float> CoreDistanceCalculator<'a, T> {
    pub(crate) fn new(data: &'a [Vec<T>], hp: &'a HdbscanHyperParams) -> Self {
        Self {
            data,
            nn_algo: hp.nn_algo.clone(),
            dist_metric: hp.dist_metric,
            k: hp.min_samples,
        }
    }

    pub(crate) fn calc_core_distances(&self) -> Vec<T> {
        let n_samples = self.data.len();
        match (&self.nn_algo, n_samples, &self.dist_metric) {
            (_, _, DistanceMetric::Precalculated) => {
                get_core_distances_from_matrix(&self.data, self.k)
            }
            (NnAlgorithm::Auto, usize::MIN..=BRUTE_FORCE_N_SAMPLES_LIMIT, _) => {
                BruteForce::calc_core_distances(&self.data, self.k, self.dist_metric)
            }
            (NnAlgorithm::Auto, _, _) => {
                KdTree::calc_core_distances(&self.data, self.k, self.dist_metric)
            }
            (NnAlgorithm::BruteForce, _, _) => {
                BruteForce::calc_core_distances(&self.data, self.k, self.dist_metric)
            }
            (NnAlgorithm::KdTree, _, _) => {
                KdTree::calc_core_distances(&self.data, self.k, self.dist_metric)
            }
        }
    }
}

pub(crate) struct BruteForce;

impl BruteForce {
    fn calc_core_distances<T: Float>(
        data: &[Vec<T>],
        k: usize,
        dist_metric: DistanceMetric,
    ) -> Vec<T> {
        let dist_matrix = calc_pairwise_distances(data, get_dist_func(&dist_metric));
        get_core_distances_from_matrix(&dist_matrix, k)
    }
}

fn calc_pairwise_distances<T, F>(data: &[Vec<T>], dist_func: F) -> Vec<Vec<T>>
where
    T: Float,
    F: Fn(&[T], &[T]) -> T,
{
    (0..data.len())
        .into_iter()
        .map(|i| {
            (0..data.len())
                .into_iter()
                .map(|j| dist_func(&data[i], &data[j]))
                .collect()
        })
        .collect()
}

pub(crate) fn get_core_distances_from_matrix<T: Float>(dist_matrix: &[Vec<T>], k: usize) -> Vec<T> {
    dist_matrix
        .iter()
        .map(|distances| {
            let mut dist = distances.clone();
            dist.sort_by(|a, b| a.partial_cmp(b).expect("Invalid float"));
            dist[k - 1]
        })
        .collect()
}

pub(crate) struct KdTree;

impl KdTree {
    fn calc_core_distances<T: Float>(
        data: &[Vec<T>],
        k: usize,
        dist_metric: DistanceMetric,
    ) -> Vec<T> {
        let mut tree: kdtree::KdTree<T, usize, &Vec<T>> = kdtree::KdTree::new(data[0].len());
        data.iter()
            .enumerate()
            .for_each(|(n, datapoint)| tree.add(datapoint, n).expect("Failed to add to KdTree"));

        let dist_func = get_dist_func(&dist_metric);
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
