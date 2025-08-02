#![cfg(feature = "serial")]
use hdbscan::{Hdbscan, HdbscanError};

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

