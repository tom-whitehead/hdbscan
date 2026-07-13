#![cfg(feature = "parallel")]
use hdbscan::{Hdbscan, HdbscanError, HdbscanResult};

mod common;

macro_rules! define_parallel_test {
    ($test_fn:ident) => {
        #[test]
        fn $test_fn() {
            fn cluster_fn(hdb: &Hdbscan<f32>) -> Result<Vec<i32>, HdbscanError> {
                hdb.cluster_par()
            }

            common::$test_fn(cluster_fn);
        }
    };
}

macro_rules! define_parallel_detailed_test {
    ($test_fn:ident) => {
        #[test]
        fn $test_fn() {
            fn cluster_fn(hdb: &Hdbscan<f32>) -> Result<HdbscanResult<f32>, HdbscanError> {
                hdb.cluster_detailed_par()
            }

            common::$test_fn(cluster_fn);
        }
    };
}

define_parallel_test!(test_cluster);
define_parallel_test!(test_builder_cluster);
define_parallel_test!(test_single_cluster);
define_parallel_test!(test_single_cluster_epsilon_search);
define_parallel_test!(test_single_root_cluster_only_epsilon_search);
define_parallel_test!(test_empty_data);
define_parallel_test!(test_mismatched_dimensions);
define_parallel_test!(test_non_finite_coordinate);
define_parallel_test!(test_calc_centroids);
define_parallel_test!(test_calc_medoids);
define_parallel_test!(test_nyc_landmarks_haversine);
define_parallel_test!(test_geo_cluster_across_180th_meridian);
define_parallel_test!(test_cylindrical_hsv_colours);
define_parallel_test!(test_precomputed_distances);
define_parallel_test!(test_leaf_basic_clusters);
define_parallel_test!(test_leaf_matches_eom_simple);
define_parallel_test!(test_leaf_with_epsilon);
define_parallel_test!(test_leaf_finer_grained);
define_parallel_test!(test_leaf_allow_single_cluster);
define_parallel_test!(test_leaf_allow_single_cluster_epsilon);
define_parallel_test!(test_leaf_manhattan_distance);
define_parallel_test!(test_leaf_precomputed_distances);
define_parallel_test!(test_default_is_eom);

#[test]
fn test_detailed_labels_match_cluster() {
    let data = common::cluster_test_data();
    let clusterer = Hdbscan::default_hyper_params(&data);
    let labels = clusterer.cluster_par().unwrap();
    let result = clusterer.cluster_detailed_par().unwrap();
    assert_eq!(labels, result.labels);
}

define_parallel_detailed_test!(test_condensed_tree_not_empty);
define_parallel_detailed_test!(test_condensed_tree_has_all_points);
define_parallel_detailed_test!(test_condensed_tree_valid_parents);
define_parallel_detailed_test!(test_probabilities_range);
define_parallel_detailed_test!(test_probabilities_noise_zero);
define_parallel_detailed_test!(test_probabilities_clustered_nonzero);
define_parallel_detailed_test!(test_probabilities_length);
define_parallel_detailed_test!(test_outlier_scores_range);
define_parallel_detailed_test!(test_outlier_scores_length);
define_parallel_detailed_test!(test_outlier_scores_distant_point_high);
define_parallel_detailed_test!(test_leaf_detailed_probabilities_and_outlier_scores);
