use num_traits::Float;
use std::collections::HashSet;

/// Possible methodologies for calculating the center of clusters
pub enum Center {
    /// The elementwise mean of all data points in a cluster.
    /// The output is not guaranteed to be an observed data point.
    Centroid,
}

pub type ClusterCentroids<T> = Vec<(i32, Vec<T>)>;

impl Center {
    pub(crate) fn calc_centers<T: Float>(
        &self,
        data: &[Vec<T>],
        labels: &[i32],
    ) -> ClusterCentroids<T> {
        match self {
            Center::Centroid => self.calc_centroids(data, labels),
        }
    }

    fn calc_centroids<T: Float>(&self, data: &[Vec<T>], labels: &[i32]) -> ClusterCentroids<T> {
        // All points weighted equally
        let weights = vec![T::one(); data.len()];
        Center::calc_weighted_centroids(data, labels, &weights)
    }

    fn calc_weighted_centroids<T: Float>(
        data: &[Vec<T>],
        labels: &[i32],
        weights: &[T],
    ) -> ClusterCentroids<T> {
        let n_dims = data[0].len();
        let unique_labels: HashSet<_> = labels.iter().filter(|&&label| label != -1).collect();

        let mut centroids = Vec::with_capacity(unique_labels.len());
        for &cluster_id in unique_labels.iter() {
            let mut count = T::zero();
            let mut element_wise_sum = vec![T::zero(); n_dims];
            for n in 0..data.len() {
                if *cluster_id == labels[n] {
                    count = count + weights[n];
                    element_wise_sum = data[n]
                        .iter()
                        .zip(element_wise_sum.iter())
                        .map(|(&element, &sum)| (element * weights[n]) + sum)
                        .collect();
                }
            }
            let centroid = element_wise_sum.iter().map(|&sum| sum / count).collect();
            centroids.push((*cluster_id, centroid));
        }
        centroids
    }
}
