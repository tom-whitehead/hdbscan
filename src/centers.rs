use num_traits::Float;
use std::cmp::Ordering;
use std::collections::HashSet;

/// Possible methodologies for calculating the center of clusters
#[derive(Debug, PartialEq)]
pub enum Center {
    /// The elementwise mean of all data points in a cluster.
    /// The output is not guaranteed to be an observed data point.
    Centroid,
    /// Calculates the geographical centroid for lat/lon coordinates.
    /// Assumes input coordinates are in degrees (latitude, longitude).
    /// Output coordinates are also in degrees.
    GeoCentroid,
    /// The point in a cluster with the minimum distance to all other points. Computationally more
    /// expensive than centroids as requires calculation of pairwise distances (using the selected
    /// distance metric). The output will be an observed data point in the cluster.
    Medoid,
}

impl Center {
    pub(crate) fn calc_centers<T: Float, F: Fn(&[T], &[T]) -> T>(
        &self,
        data: &[Vec<T>],
        labels: &[i32],
        dist_func: F,
    ) -> Vec<Vec<T>> {
        match self {
            Center::Centroid => self.calc_centroids(data, labels),
            Center::GeoCentroid => self.calc_geo_centroids(data, labels),
            Center::Medoid => self.calc_medoids(data, labels, dist_func),
        }
    }

    fn calc_centroids<T: Float>(&self, data: &[Vec<T>], labels: &[i32]) -> Vec<Vec<T>> {
        // All points weighted equally for now
        let weights = vec![T::one(); data.len()];
        Center::calc_weighted_centroids(data, labels, &weights)
    }

    fn calc_weighted_centroids<T: Float>(
        data: &[Vec<T>],
        labels: &[i32],
        weights: &[T],
    ) -> Vec<Vec<T>> {
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
                    element_wise_mean = data[n]
                        .iter()
                        .zip(element_wise_mean.iter())
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

    /// Calculates the geographical centeroid for each cluster.
    ///
    /// This method is specifically designed for geographical data where each point
    /// is represented by latitude and longitude coordinates.
    ///
    /// # Arguments
    ///
    /// * `data` - A slice of vectors, where each vector contains [latitude, longitude] in degrees.
    /// * `labels` - A slice of cluster labels corresponding to each data point.
    ///
    /// # Returns
    ///
    /// A vector of cluster centers, where each center is a vector of [latitude, longitude] in degrees.
    ///
    /// # Notes
    ///
    /// - Assumes input coordinates are in degrees.
    /// - Output coordinates are in degrees.
    /// - Points with label -1 are considered noise and are ignored in calculations.
    /// - Uses a spherical approximation of the Earth for calculations.
    fn calc_geo_centroids<T: Float>(&self, data: &[Vec<T>], labels: &[i32]) -> Vec<Vec<T>> {
        let n_clusters = labels
            .iter()
            .filter(|&&label| label != -1)
            .collect::<HashSet<_>>()
            .len();
        let mut centers = vec![vec![T::zero(), T::zero()]; n_clusters];
        let mut counts = vec![T::zero(); n_clusters];

        for (point, &label) in data.iter().zip(labels.iter()) {
            if label != -1 {
                let cluster_index = label as usize;
                centers[cluster_index][0] = centers[cluster_index][0] + point[0].to_radians();
                centers[cluster_index][1] = centers[cluster_index][1] + point[1].to_radians();
                counts[cluster_index] = counts[cluster_index] + T::one();
            }
        }

        // Calculate final geo centroid for each cluster
        for (center, &count) in centers.iter_mut().zip(counts.iter()) {
            if count > T::zero() {
                let avg_lat = center[0] / count;
                let avg_lon = center[1] / count;

                let x = avg_lon.cos() * avg_lat.cos();
                let y = avg_lon.sin() * avg_lat.cos();
                let z = avg_lat.sin();

                let lon = y.atan2(x);
                let hyp = (x * x + y * y).sqrt();
                let lat = z.atan2(hyp);

                // Convert back to degrees
                center[0] = lat.to_degrees();
                center[1] = lon.to_degrees();
            }
        }

        centers
    }

    fn calc_medoids<T: Float, F: Fn(&[T], &[T]) -> T>(
        &self,
        data: &[Vec<T>],
        labels: &[i32],
        dist_func: F,
    ) -> Vec<Vec<T>> {
        let n_clusters = labels
            .iter()
            .filter(|&&label| label != -1)
            .collect::<HashSet<_>>()
            .len();
        let mut medoids = Vec::with_capacity(n_clusters);

        for cluster_id in 0..n_clusters as i32 {
            let cluster_data = data
                .iter()
                .zip(labels.iter())
                .filter(|(_datapoint, &label)| label == cluster_id)
                .map(|(datapoint, _label)| datapoint)
                .collect::<Vec<&Vec<_>>>();

            let n_samples = cluster_data.len();
            let medoid_idx = (0..n_samples)
                .map(|i| {
                    (0..n_samples)
                        .map(|j| dist_func(cluster_data[i], cluster_data[j]))
                        .fold(T::zero(), std::ops::Add::add)
                })
                .enumerate()
                .min_by(|(_idx_a, sum_a), (_idx_b, sum_b)| {
                    sum_a.partial_cmp(sum_b).unwrap_or(Ordering::Equal)
                })
                .map(|(idx, _sum)| idx)
                .unwrap_or(0);

            medoids.push(cluster_data[medoid_idx].clone())
        }

        medoids
    }
}
