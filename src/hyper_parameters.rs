use crate::distance::DistanceMetric;
use crate::core_distances::NnAlgorithm;

// Defaults for parameters
const MIN_CLUSTER_SIZE_DEFAULT: usize = 5;
const MAX_CLUSTER_SIZE_DEFAULT: usize = usize::MAX; // Set to a value that will never be triggered
const ALLOW_SINGLE_CLUSTER_DEFAULT: bool = false;
const DISTANCE_METRIC_DEFAULT: DistanceMetric = DistanceMetric::Euclidean;
const NN_ALGORITHM_DEFAULT: NnAlgorithm = NnAlgorithm::Auto;

// Valid minimums/left bounds of parameters
const MIN_CLUSTER_SIZE_MINIMUM: usize = 2;
const MAX_CLUSTER_SIZE_MINIMUM: usize = 2;
const MIN_SAMPLES_MINIMUM: usize = 1;


/// A wrapper around the various hyper parameters used in HDBSCAN clustering.
/// Only use if you want to tune hyper parameters. Otherwise use `Hdbscan::default()` to 
/// instantiate the model with default hyper parameters.
pub struct HdbscanHyperParams {
    pub(crate) min_cluster_size: usize,
    pub(crate) max_cluster_size: usize,
    pub(crate) allow_single_cluster: bool,
    pub(crate) min_samples: usize,
    pub(crate) dist_metric: DistanceMetric,
    pub(crate) nn_algo: NnAlgorithm,
}

/// Builder object to set custom hyper parameters.
pub struct HyperParamBuilder {
    min_cluster_size: Option<usize>,
    max_cluster_size: Option<usize>,
    allow_single_cluster: Option<bool>,
    min_samples: Option<usize>,
    dist_metric: Option<DistanceMetric>,
    nn_algo: Option<NnAlgorithm>,
}

impl HdbscanHyperParams {
    pub(crate) fn default() -> Self {
        Self::builder().build()
    }

    /// Enters the builder pattern, allowing custom hyper parameters to be set using
    /// various setter methods.
    /// 
    /// # Returns 
    /// * the hyper parameter configuration builder
    pub fn builder() -> HyperParamBuilder {
        HyperParamBuilder {
            min_cluster_size: None,
            max_cluster_size: None,
            allow_single_cluster: None,
            min_samples: None,
            dist_metric: None,
            nn_algo: None,
        }
    }
}

impl HyperParamBuilder {

    /// Sets the minimum cluster size - the minimum number of samples for a group of
    /// data points to be considered a cluster. If a grouping of data points has fewer
    /// members than this, then they will be considered noise.
    /// This should be considered the main hyper parameter for changing the results of clustering.
    /// Defaults to 5.
    ///
    /// # Parameters
    /// * min_cluster_size - the minimum cluster size
    ///
    /// # Returns
    /// * the hyper parameter configuration builder
    pub fn min_cluster_size(mut self, min_cluster_size: usize) -> HyperParamBuilder {
        let valid_min_cluster_size = HyperParamBuilder::validate_input_left_bound(
            min_cluster_size, MIN_CLUSTER_SIZE_MINIMUM, "min_cluster_size");
        self.min_cluster_size = Some(valid_min_cluster_size);
        self
    }

    /// Sets the maximum cluster size - the maximum number of samples for a group of
    /// data points to be considered a cluster. If a grouping of data points has more
    /// members than this. By default, this value is not considered in clustering.
    ///
    /// # Parameters
    /// * max_cluster_size - the maximum cluster size
    ///
    /// # Returns
    /// * the hyper parameter configuration builder
    pub fn max_cluster_size(mut self, max_cluster_size: usize) -> HyperParamBuilder {
        let valid_max_cluster_size = HyperParamBuilder::validate_input_left_bound(
            max_cluster_size, MAX_CLUSTER_SIZE_MINIMUM, "max_cluster_size");
        self.max_cluster_size = Some(valid_max_cluster_size);
        self
    }

    /// Sets whether to allow one single cluster (i.e. the root or top cluster). Only set
    /// this to true if you feel there being one cluster is correct for your dataset.
    /// Defaults to false.
    ///
    /// # Parameters
    /// * allow_single_cluster - whether to allow a single cluster
    ///
    /// # Returns
    /// * the hyper parameter configuration builder
    pub fn allow_single_cluster(mut self, allow_single_cluster: bool) -> HyperParamBuilder {
        self.allow_single_cluster = Some(allow_single_cluster);
        self
    }

    /// Sets min samples. HDBSCAN calculates the core distances between points as a first step
    /// in clustering. The core distance is the distance to the Kth neighbour using a nearest
    /// neighbours algorithm, where k = min_samples. Defaults to min_cluster_size.
    ///
    /// # Parameters
    /// * min_cluster_size - the number of neighbourhood points considered in distances
    ///
    /// # Returns
    /// * the hyper parameter configuration builder
    pub fn min_samples(mut self, min_samples: usize) -> HyperParamBuilder {
        let valid_min_samples = HyperParamBuilder::validate_input_left_bound(
            min_samples, MIN_SAMPLES_MINIMUM, "min_samples");
        self.min_samples = Some(valid_min_samples);
        self
    }

    /// Sets the distance metric. HDBSCAN uses this metric to calculate the distance between 
    /// data points. Defaults to Euclidean. Options are defined by the DistanceMetric enum.
    ///
    /// # Parameters
    /// * dist_metric - the distance metric
    ///
    /// # Returns
    /// * the hyper parameter configuration builder
    pub fn dist_metric(mut self, dist_metric: DistanceMetric) -> HyperParamBuilder {
        self.dist_metric = Some(dist_metric);
        self
    }
    
    /// Sets the nearest neighbour algorithm. Internally, HDBSCAN calculates a density measure
    /// called core distances, which is defined as the distance of a data point to it's kth 
    /// (min_samples-th) neighbour. 
    /// The primary reason for changing this parameter is performance. For example, using BruteForce 
    /// involves computing a distance matrix between all data points. This works fine on small 
    /// datasets, however scales poorly to larger ones.
    /// Defaults to Auto, whereby the nearest neighbour algorithm will be chosen internally based
    /// on size and dimensionality of the input data. 
    /// 
    /// # Returns
    /// * the hyper parameter configuration builder
    pub fn nn_algorithm(mut self, nn_algorithm: NnAlgorithm) -> HyperParamBuilder {
        self.nn_algo = Some(nn_algorithm);
        self
    }
    
    /// Finishes the building of the hyper parameter configuration. A call to this method is
    /// required to exist the builder pattern and complete the construction of the hyper parameters.
    ///
    /// # Returns
    /// * The completed HDBSCAN hyper parameter configuration.
    pub fn build(self) -> HdbscanHyperParams {
        let min_cluster_size = self.min_cluster_size.unwrap_or(MIN_CLUSTER_SIZE_DEFAULT);
        HdbscanHyperParams {
            min_cluster_size,
            max_cluster_size: self.max_cluster_size.unwrap_or(MAX_CLUSTER_SIZE_DEFAULT),
            allow_single_cluster: self.allow_single_cluster.unwrap_or(ALLOW_SINGLE_CLUSTER_DEFAULT),
            min_samples: self.min_samples.unwrap_or(min_cluster_size),
            dist_metric: self.dist_metric.unwrap_or(DISTANCE_METRIC_DEFAULT),
            nn_algo: self.nn_algo.unwrap_or(NN_ALGORITHM_DEFAULT),
        }
    }

    fn validate_input_left_bound(input_param: usize, left_bound: usize, param: &str) -> usize {
        if input_param < left_bound {
            println!(
                "HDBSCAN_WARNING: {param} ({input_param}) cannot be lower \
                than {left_bound}. Set to {left_bound}."
            );
            left_bound
        } else {
            input_param
        }
    }

}
