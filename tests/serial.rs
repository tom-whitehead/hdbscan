#![cfg(feature = "serial")]
use hdbscan::{Hdbscan, HdbscanError, HdbscanResult};

mod common;

macro_rules! define_serial_test {
    ($test_fn:ident) => {
        #[test]
        fn $test_fn() {
            fn cluster_fn(hdb: &Hdbscan<f32>) -> Result<Vec<i32>, HdbscanError> {
                hdb.cluster()
            }

            common::$test_fn(cluster_fn);
        }
    };
}

macro_rules! define_serial_detailed_test {
    ($test_fn:ident) => {
        #[test]
        fn $test_fn() {
            fn cluster_fn(hdb: &Hdbscan<f32>) -> Result<HdbscanResult<f32>, HdbscanError> {
                hdb.cluster_detailed()
            }

            common::$test_fn(cluster_fn);
        }
    };
}

define_serial_test!(test_cluster);
define_serial_test!(test_builder_cluster);
define_serial_test!(test_single_cluster);
define_serial_test!(test_single_cluster_epsilon_search);
define_serial_test!(test_single_root_cluster_only_epsilon_search);
define_serial_test!(test_empty_data);
define_serial_test!(test_mismatched_dimensions);
define_serial_test!(test_non_finite_coordinate);
define_serial_test!(test_calc_centroids);
define_serial_test!(test_calc_medoids);
define_serial_test!(test_nyc_landmarks_haversine);
define_serial_test!(test_geo_cluster_across_180th_meridian);
define_serial_test!(test_cylindrical_hsv_colours);
define_serial_test!(test_precomputed_distances);
define_serial_test!(test_leaf_basic_clusters);
define_serial_test!(test_leaf_matches_eom_simple);
define_serial_test!(test_leaf_with_epsilon);
define_serial_test!(test_leaf_finer_grained);
define_serial_test!(test_leaf_allow_single_cluster);
define_serial_test!(test_leaf_allow_single_cluster_epsilon);
define_serial_test!(test_leaf_manhattan_distance);
define_serial_test!(test_leaf_precomputed_distances);
define_serial_test!(test_default_is_eom);

#[test]
fn test_detailed_labels_match_cluster() {
    let data = common::cluster_test_data();
    let clusterer = Hdbscan::default_hyper_params(&data);
    let labels = clusterer.cluster().unwrap();
    let result = clusterer.cluster_detailed().unwrap();
    assert_eq!(labels, result.labels);
}

define_serial_detailed_test!(test_condensed_tree_not_empty);
define_serial_detailed_test!(test_condensed_tree_has_all_points);
define_serial_detailed_test!(test_condensed_tree_valid_parents);
define_serial_detailed_test!(test_probabilities_range);
define_serial_detailed_test!(test_probabilities_noise_zero);
define_serial_detailed_test!(test_probabilities_clustered_nonzero);
define_serial_detailed_test!(test_probabilities_length);
define_serial_detailed_test!(test_outlier_scores_range);
define_serial_detailed_test!(test_outlier_scores_length);
define_serial_detailed_test!(test_outlier_scores_distant_point_high);
define_serial_detailed_test!(test_leaf_detailed_probabilities_and_outlier_scores);

#[test]
fn test_duplicate_points_no_nan() {
    // Duplicate/near-duplicate points cause distance=0, lambda=infinity.
    // Probabilities and outlier scores must remain finite (no NaN from inf/inf).
    let data: Vec<Vec<f32>> = vec![
        // Cluster A: many exact duplicates
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        // Cluster B: many exact duplicates
        vec![10.0, 10.0],
        vec![10.0, 10.0],
        vec![10.0, 10.0],
        vec![10.0, 10.0],
        vec![10.0, 10.0],
        vec![10.1, 10.1],
    ];
    let clusterer = Hdbscan::default_hyper_params(&data);
    let result = clusterer.cluster_detailed().unwrap();

    for (i, &p) in result.probabilities.iter().enumerate() {
        assert!(!p.is_nan(), "Probability is NaN at index {}", i);
        assert!(
            p >= 0.0 && p <= 1.0,
            "Probability {} out of range at index {}",
            p,
            i
        );
    }
    for (i, &s) in result.outlier_scores.iter().enumerate() {
        assert!(!s.is_nan(), "Outlier score is NaN at index {}", i);
        assert!(
            s >= 0.0 && s <= 1.0,
            "Outlier score {} out of range at index {}",
            s,
            i
        );
    }
}
