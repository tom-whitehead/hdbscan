use std::collections::HashSet;
use num_traits::Float;

/// Possible methodologies for calculating the center of clusters
pub enum Center {
    /// The elementwise mean of all data points in a cluster.
    /// The output is not guaranteed to be an observed data point.
    Centroid,
}

impl Center {
    pub(crate) fn calc_centers<T: Float>(
        &self, data: &Vec<Vec<T>>, labels: &Vec<i32>) -> Vec<Vec<T>> {
        assert_eq!(data.len(), labels.len());
        match self {
            Center::Centroid  => self.calc_centroids(data, labels),
        }
    }

    fn calc_centroids<T: Float>(&self, data: &Vec<Vec<T>>, labels: &Vec<i32>) ->  Vec<Vec<T>> {
        // All points weighted equally
        let weights = vec![T::one(); data.len()];
        Self::calc_weighted_centroids(data, labels, &weights)
    }

    fn calc_weighted_centroids<T: Float>(
        data: &Vec<Vec<T>>, labels: &Vec<i32>, weights: &Vec<T>) ->  Vec<Vec<T>> {
        let n_dims = data[0].len();
        let n_clusters = labels
            .iter()
            .filter(|&&label| label != -1)
            .collect::<HashSet<_>>()
            .len();

        let mut centroids = Vec::with_capacity(n_clusters);
        for cluster_id in 0..n_clusters as i32 {
            let mut count = T::zero();
            let mut element_wise_mean = vec![T::zero(); n_dims];
            for n in 0..data.len() {
                if cluster_id == labels[n] {
                    count = count + T::one();
                    element_wise_mean = data[n].iter().zip(element_wise_mean.iter())
                        .map(|(&element, &sum)| (element * weights[n]) + sum)
                        .collect();
                }
            }
            for element in element_wise_mean.iter_mut() {
                *element = *element / count;
            }
            centroids.push(element_wise_mean);
        }
        centroids
    }
}
