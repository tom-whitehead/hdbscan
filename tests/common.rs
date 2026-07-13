use hdbscan::{
    Center, ClusterSelectionMethod, DistanceMetric, Hdbscan, HdbscanError, HdbscanHyperParams,
    HdbscanResult, NnAlgorithm,
};
use num_traits::Float;
use std::collections::HashSet;

type ClusterFn = fn(&Hdbscan<f32>) -> Result<Vec<i32>, HdbscanError>;
pub(crate) type DetailedClusterFn = fn(&Hdbscan<f32>) -> Result<HdbscanResult<f32>, HdbscanError>;

pub(crate) fn test_cluster(cluster_fn: ClusterFn) {
    let data = cluster_test_data();
    let clusterer = Hdbscan::default_hyper_params(&data);
    let result = cluster_fn(&clusterer).unwrap();
    // First five points form one cluster
    assert_eq!(1, result[..5].iter().collect::<HashSet<_>>().len());
    // Next five points are a second cluster
    assert_eq!(1, result[5..10].iter().collect::<HashSet<_>>().len());
    // The final point is noise
    assert_eq!(-1, result[10]);
}

pub(crate) fn test_builder_cluster(cluster_fn: ClusterFn) {
    let data = vec![
        vec![1.3, 1.1],
        vec![1.3, 1.2],
        vec![1.2, 1.2],
        vec![1.0, 1.1],
        vec![0.9, 1.0],
        vec![0.9, 1.0],
        vec![3.7, 4.0],
    ];
    let hyper_params = HdbscanHyperParams::builder()
        .min_cluster_size(3)
        .min_samples(2)
        .dist_metric(DistanceMetric::Manhattan)
        .nn_algorithm(NnAlgorithm::BruteForce)
        .build();
    let clusterer = Hdbscan::new(&data, hyper_params);
    let result = cluster_fn(&clusterer).unwrap();
    // First three points form one cluster
    assert_eq!(1, result[..3].iter().collect::<HashSet<_>>().len());
    // Next three points are a second cluster
    assert_eq!(1, result[3..6].iter().collect::<HashSet<_>>().len());
    // The final point is noise
    assert_eq!(-1, result[6]);
}

pub(crate) fn test_single_cluster(cluster_fn: ClusterFn) {
    let data = vec![
        vec![1.1, 1.1],
        vec![1.2, 1.1],
        vec![1.3, 1.2],
        vec![1.1, 1.3],
        vec![1.2, 1.2],
        vec![3.0, 3.0],
    ];

    let hp = HdbscanHyperParams::builder()
        .nn_algorithm(NnAlgorithm::BruteForce)
        .allow_single_cluster(true)
        .min_cluster_size(4)
        .min_samples(4)
        .build();
    let clusterer = Hdbscan::new(&data, hp);
    let result = cluster_fn(&clusterer).unwrap();

    let unique_clusters: HashSet<_> = result.iter().filter(|&&x| x != -1).collect();
    assert_eq!(1, unique_clusters.len());

    let noise_points: Vec<_> = result.iter().filter(|&&x| x == -1).collect();
    assert_eq!(1, noise_points.len());
}

pub(crate) fn test_single_cluster_epsilon_search(cluster_fn: ClusterFn) {
    let data = vec![
        vec![1.1, 1.1],
        vec![1.2, 1.1],
        vec![1.3, 1.2],
        vec![2.1, 1.3],
        vec![2.2, 1.2],
        vec![2.0, 1.2],
        vec![3.0, 3.0],
    ];

    let hp = HdbscanHyperParams::builder().min_cluster_size(3).build();
    let clusterer = Hdbscan::new(&data, hp);
    let result = cluster_fn(&clusterer).unwrap();

    // Without allow_single_cluster and epsilon, there are two clusters
    let unique_clusters = result
        .iter()
        .filter(|&&label| label != -1)
        .collect::<HashSet<_>>();
    assert_eq!(2, unique_clusters.len());
    // One point is noise
    let n_noise = result.iter().filter(|&&label| label == -1).count();
    assert_eq!(1, n_noise);

    let hp = HdbscanHyperParams::builder()
        .allow_single_cluster(true)
        .min_cluster_size(3)
        .epsilon(1.2)
        .build();
    let clusterer = Hdbscan::new(&data, hp);
    let result = cluster_fn(&clusterer).unwrap();

    // With allow_single_cluster and epsilon, first size points are one merged cluster
    let unique_clusters = result
        .iter()
        .filter(|&&label| label != -1)
        .collect::<HashSet<_>>();
    assert_eq!(1, unique_clusters.len());
    // One point is still noise
    let n_noise = result.iter().filter(|&&label| label == -1).count();
    assert_eq!(1, n_noise);
}

pub(crate) fn test_single_root_cluster_only_epsilon_search(cluster_fn: ClusterFn) {
    // This used to cause a panic
    let data = vec![
        vec![1.1, 1.1],
        vec![1.2, 1.1],
        vec![1.3, 1.2],
        vec![3.0, 3.0],
    ];

    let hp = HdbscanHyperParams::builder()
        .allow_single_cluster(true)
        .min_cluster_size(3)
        .epsilon(1.2)
        .build();
    let clusterer = Hdbscan::new(&data, hp);
    let result = cluster_fn(&clusterer).unwrap();

    let unique_clusters = result
        .iter()
        .filter(|&&label| label != -1)
        .collect::<HashSet<_>>();
    assert_eq!(1, unique_clusters.len());
    let n_noise = result.iter().filter(|&&label| label == -1).count();
    assert_eq!(1, n_noise);
}

pub(crate) fn test_empty_data(cluster_fn: ClusterFn) {
    let data: Vec<Vec<f32>> = Vec::new();
    let clusterer = Hdbscan::default_hyper_params(&data);
    let result = cluster_fn(&clusterer);
    assert!(matches!(result, Err(HdbscanError::EmptyDataset)));
}

pub(crate) fn test_non_finite_coordinate(cluster_fn: ClusterFn) {
    let data = vec![vec![1.5, f32::infinity()]];
    let clusterer = Hdbscan::default_hyper_params(&data);
    let result = cluster_fn(&clusterer);
    assert!(matches!(result, Err(HdbscanError::NonFiniteCoordinate(..))));
}

pub(crate) fn test_mismatched_dimensions(cluster_fn: ClusterFn) {
    let data = vec![vec![1.5, 2.2], vec![1.0, 1.1], vec![1.2]];
    let clusterer = Hdbscan::default_hyper_params(&data);
    let result = cluster_fn(&clusterer);
    assert!(matches!(result, Err(HdbscanError::WrongDimension(..))));
}

pub(crate) fn test_calc_centroids(cluster_fn: ClusterFn) {
    let data = cluster_test_data();
    let clusterer = Hdbscan::default_hyper_params(&data);
    let labels = cluster_fn(&clusterer).unwrap();
    let centroids = clusterer.calc_centers(Center::Centroid, &labels).unwrap();
    assert_eq!(2, centroids.len());
    assert!(centroids.contains(&vec![3.8, 4.0]) && centroids.contains(&vec![1.12, 1.34]));
}

pub(crate) fn test_calc_medoids(cluster_fn: ClusterFn) {
    let data: Vec<Vec<f32>> = vec![
        vec![1.3, 1.2],
        vec![1.2, 1.3],
        vec![1.5, 1.5],
        vec![1.6, 1.7],
        vec![1.7, 1.6],
        vec![6.3, 6.2],
        vec![6.2, 6.3],
        vec![6.5, 6.5],
        vec![6.6, 6.7],
        vec![6.7, 6.6],
    ];
    let clusterer = Hdbscan::default_hyper_params(&data);
    let result = cluster_fn(&clusterer).unwrap();
    let centers = clusterer.calc_centers(Center::Medoid, &result).unwrap();

    let unique_clusters = result
        .iter()
        .filter(|&&label| label != -1)
        .collect::<HashSet<_>>();
    assert_eq!(centers.len(), unique_clusters.len());

    centers
        .iter()
        .for_each(|center| assert!(data.contains(center)));
    assert_eq!(vec![1.5, 1.5], centers[0]);
    assert_eq!(vec![6.5, 6.5], centers[1]);
}

pub(crate) fn cluster_test_data() -> Vec<Vec<f32>> {
    vec![
        vec![1.5, 2.2],
        vec![1.0, 1.1],
        vec![1.2, 1.4],
        vec![0.8, 1.0],
        vec![1.1, 1.0],
        vec![3.7, 4.0],
        vec![3.9, 3.9],
        vec![3.6, 4.1],
        vec![3.8, 3.9],
        vec![4.0, 4.1],
        vec![10.0, 10.0],
    ]
}

pub(crate) fn test_nyc_landmarks_haversine(cluster_fn: ClusterFn) {
    let data = vec![
        // Cluster 1: Statue of Liberty area
        vec![40.6892, -74.0445], // Statue of Liberty
        vec![40.7036, -74.0141], // Battery Park
        vec![40.7033, -74.0170], // Staten Island Ferry Terminal
        // Cluster 2: Central Park area
        vec![40.7812, -73.9665], // Metropolitan Museum of Art
        vec![40.7794, -73.9632], // Guggenheim Museum
        vec![40.7729, -73.9734], // Central Park Zoo
        // Cluster 3: Times Square area
        vec![40.7580, -73.9855], // Times Square
        vec![40.7614, -73.9776], // Rockefeller Center
        vec![40.7505, -73.9934], // Madison Square Garden
        // Outlier
        vec![40.6413, -74.0781], // Staten Island Mall (should be noise)
    ];

    let hyper_params = HdbscanHyperParams::builder()
        .min_cluster_size(2)
        .min_samples(1)
        .dist_metric(DistanceMetric::Haversine)
        // 500m to consider separate cluster
        .epsilon(500.0)
        .nn_algorithm(NnAlgorithm::BruteForce)
        .build();

    let clusterer = Hdbscan::new(&data, hyper_params);
    let result = cluster_fn(&clusterer).unwrap();

    // Check that we have 3 clusters and 1 noise point
    let unique_clusters: HashSet<_> = result.iter().filter(|&&x| x != -1).collect();
    assert_eq!(unique_clusters.len(), 3, "Should have 3 distinct clusters");
    assert_eq!(
        result.iter().filter(|&&x| x == -1).count(),
        1,
        "Should have 1 noise point"
    );

    // Check that points in each area are in the same cluster
    assert_eq!(result[0], result[1]);
    assert_eq!(result[1], result[2]);

    assert_eq!(result[3], result[4]);
    assert_eq!(result[4], result[5]);

    assert_eq!(result[6], result[7]);
    assert_eq!(result[7], result[8]);

    // Check that the last point is noise
    assert_eq!(result[9], -1);
}

pub(crate) fn test_geo_cluster_across_180th_meridian(cluster_fn: ClusterFn) {
    let data = vec![
        vec![-16.8410, 179.9813],  // Taveuni, Fiji
        vec![-16.7480, -179.9670], // Qamea, Fiji
        vec![51.5085, -0.1257],    // London - noise
    ];

    let hyper_params = HdbscanHyperParams::builder()
        .dist_metric(DistanceMetric::Haversine)
        .allow_single_cluster(true)
        .min_cluster_size(2)
        .min_samples(1)
        .build();

    let clusterer = Hdbscan::new(&data, hyper_params);
    let labels = cluster_fn(&clusterer).unwrap();

    // There is only one cluster
    assert_eq!(
        1,
        labels
            .iter()
            .filter(|&&x| x != -1)
            .collect::<HashSet<_>>()
            .len()
    );
    // The last point is noise
    assert_eq!(-1, labels[2]);

    let centroids = clusterer
        .calc_centers(Center::GeoCentroid, &labels)
        .unwrap();
    let cluster_longitude = centroids[0][1];

    // The cluster centroid is not impacted by the longitudes being either side
    // of the 180th meridian
    assert!(cluster_longitude > 179.0 || cluster_longitude < -179.0);
}

pub(crate) fn test_cylindrical_hsv_colours(cluster_fn: ClusterFn) {
    // HSV colours re-ordered to SHV
    let data = vec![
        // Blues
        vec![0.91, 3.80482, 0.62],
        vec![0.96, 4.13643, 0.86],
        vec![0.95, 3.56047, 0.85],
        // Greens
        vec![0.74, 1.91986, 0.39],
        vec![0.90, 1.69297, 0.82],
        vec![0.84, 2.14675, 0.72],
        // Red
        vec![0.60, 6.2657, 0.00],
    ];

    let hyper_params = HdbscanHyperParams::builder()
        .dist_metric(DistanceMetric::Cylindrical)
        .nn_algorithm(NnAlgorithm::BruteForce)
        .min_cluster_size(3)
        .min_samples(1)
        .build();

    let clusterer = Hdbscan::new(&data, hyper_params);
    let result = cluster_fn(&clusterer).unwrap();

    // Blues all form one cluster
    assert_eq!(1, result[..3].iter().collect::<HashSet<_>>().len());
    // Greens are a second cluster
    assert_eq!(1, result[3..6].iter().collect::<HashSet<_>>().len());
    // The final red point is noise
    assert_eq!(-1, result[6]);
}

pub(crate) fn test_precomputed_distances(cluster_fn: ClusterFn) {
    let dist_matrix = vec![
        vec![0.0, 0.1, 0.2, 0.3, 9.0],
        vec![0.1, 0.0, 0.1, 0.2, 9.0],
        vec![0.2, 0.1, 0.0, 0.1, 9.0],
        vec![0.3, 0.2, 0.1, 0.0, 9.0],
        vec![9.0, 9.0, 9.0, 9.0, 9.0],
    ];
    let hyper_params = HdbscanHyperParams::builder()
        .dist_metric(DistanceMetric::Precalculated)
        .allow_single_cluster(true)
        .min_cluster_size(2)
        .min_samples(1)
        .build();

    let clusterer = Hdbscan::new(&dist_matrix, hyper_params);
    let result = cluster_fn(&clusterer).unwrap();

    // Check that we have 1 cluster and 1 noise point
    let unique_clusters: HashSet<_> = result.iter().filter(|&&x| x != -1).collect();
    assert_eq!(unique_clusters.len(), 1, "Should have 1 distinct cluster");
    assert_eq!(result[result.len() - 1], -1, "Should have 0 noise points");
}

// ── Detailed clustering tests ───────────────────────────────────────────────

pub(crate) fn test_condensed_tree_not_empty(cluster_fn: DetailedClusterFn) {
    let data = cluster_test_data();
    let clusterer = Hdbscan::default_hyper_params(&data);
    let result = cluster_fn(&clusterer).unwrap();
    assert!(!result.condensed_tree.is_empty());
}

pub(crate) fn test_condensed_tree_has_all_points(cluster_fn: DetailedClusterFn) {
    let data = cluster_test_data();
    let n = data.len();
    let clusterer = Hdbscan::default_hyper_params(&data);
    let result = cluster_fn(&clusterer).unwrap();
    let point_ids: HashSet<usize> = result
        .condensed_tree
        .iter()
        .filter(|node| node.node_id < n)
        .map(|node| node.node_id)
        .collect();
    for i in 0..n {
        assert!(
            point_ids.contains(&i),
            "Point {} missing from condensed tree",
            i
        );
    }
}

pub(crate) fn test_condensed_tree_valid_parents(cluster_fn: DetailedClusterFn) {
    let data = cluster_test_data();
    let n = data.len();
    let clusterer = Hdbscan::default_hyper_params(&data);
    let result = cluster_fn(&clusterer).unwrap();
    for node in &result.condensed_tree {
        assert!(
            node.parent_node_id >= n,
            "Parent {} is less than n_samples {}",
            node.parent_node_id,
            n
        );
    }
}

pub(crate) fn test_probabilities_range(cluster_fn: DetailedClusterFn) {
    let data = cluster_test_data();
    let clusterer = Hdbscan::default_hyper_params(&data);
    let result = cluster_fn(&clusterer).unwrap();
    for (i, &p) in result.probabilities.iter().enumerate() {
        assert!(
            p >= 0.0 && p <= 1.0,
            "Probability {} out of range at index {}",
            p,
            i
        );
    }
}

pub(crate) fn test_probabilities_noise_zero(cluster_fn: DetailedClusterFn) {
    let data = cluster_test_data();
    let clusterer = Hdbscan::default_hyper_params(&data);
    let result = cluster_fn(&clusterer).unwrap();
    for (i, &label) in result.labels.iter().enumerate() {
        if label == -1 {
            assert_eq!(
                result.probabilities[i], 0.0,
                "Noise point {} should have probability 0",
                i
            );
        }
    }
}

pub(crate) fn test_probabilities_clustered_nonzero(cluster_fn: DetailedClusterFn) {
    let data = cluster_test_data();
    let clusterer = Hdbscan::default_hyper_params(&data);
    let result = cluster_fn(&clusterer).unwrap();
    for (i, &label) in result.labels.iter().enumerate() {
        if label != -1 {
            assert!(
                result.probabilities[i] > 0.0,
                "Clustered point {} should have probability > 0",
                i
            );
        }
    }
}

pub(crate) fn test_probabilities_length(cluster_fn: DetailedClusterFn) {
    let data = cluster_test_data();
    let clusterer = Hdbscan::default_hyper_params(&data);
    let result = cluster_fn(&clusterer).unwrap();
    assert_eq!(result.probabilities.len(), data.len());
}

pub(crate) fn test_outlier_scores_range(cluster_fn: DetailedClusterFn) {
    let data = cluster_test_data();
    let clusterer = Hdbscan::default_hyper_params(&data);
    let result = cluster_fn(&clusterer).unwrap();
    for (i, &s) in result.outlier_scores.iter().enumerate() {
        assert!(
            s >= 0.0 && s <= 1.0,
            "Outlier score {} out of range at index {}",
            s,
            i
        );
    }
}

pub(crate) fn test_outlier_scores_length(cluster_fn: DetailedClusterFn) {
    let data = cluster_test_data();
    let clusterer = Hdbscan::default_hyper_params(&data);
    let result = cluster_fn(&clusterer).unwrap();
    assert_eq!(result.outlier_scores.len(), data.len());
}

pub(crate) fn test_outlier_scores_distant_point_high(cluster_fn: DetailedClusterFn) {
    let data = cluster_test_data();
    let clusterer = Hdbscan::default_hyper_params(&data);
    let result = cluster_fn(&clusterer).unwrap();
    // Point at index 10 is (10, 10), far from both clusters
    let distant_score = result.outlier_scores[10];
    let avg_clustered: f32 = result
        .outlier_scores
        .iter()
        .enumerate()
        .filter(|(i, _)| result.labels[*i] != -1)
        .map(|(_, &s)| s)
        .sum::<f32>()
        / result.labels.iter().filter(|&&l| l != -1).count() as f32;
    assert!(
        distant_score > avg_clustered,
        "Distant point score {} should exceed average clustered score {}",
        distant_score,
        avg_clustered
    );
}

// ── Leaf cluster selection tests ────────────────────────────────────────────

pub(crate) fn test_leaf_basic_clusters(cluster_fn: ClusterFn) {
    let data = cluster_test_data();
    let hp = HdbscanHyperParams::builder()
        .cluster_selection_method(ClusterSelectionMethod::Leaf)
        .build();
    let clusterer = Hdbscan::new(&data, hp);
    let result = cluster_fn(&clusterer).unwrap();
    // First five points form one cluster
    assert_eq!(1, result[..5].iter().collect::<HashSet<_>>().len());
    // Next five points are a second cluster
    assert_eq!(1, result[5..10].iter().collect::<HashSet<_>>().len());
    // The final point is noise
    assert_eq!(-1, result[10]);
}

pub(crate) fn test_leaf_matches_eom_simple(cluster_fn: ClusterFn) {
    let data = cluster_test_data();

    let hp_eom = HdbscanHyperParams::builder()
        .cluster_selection_method(ClusterSelectionMethod::Eom)
        .build();
    let hp_leaf = HdbscanHyperParams::builder()
        .cluster_selection_method(ClusterSelectionMethod::Leaf)
        .build();

    let eom_result = cluster_fn(&Hdbscan::new(&data, hp_eom)).unwrap();
    let leaf_result = cluster_fn(&Hdbscan::new(&data, hp_leaf)).unwrap();

    let eom_clusters: HashSet<_> = eom_result.iter().filter(|&&l| l != -1).collect();
    let leaf_clusters: HashSet<_> = leaf_result.iter().filter(|&&l| l != -1).collect();
    assert_eq!(eom_clusters.len(), leaf_clusters.len());

    let eom_noise = eom_result.iter().filter(|&&l| l == -1).count();
    let leaf_noise = leaf_result.iter().filter(|&&l| l == -1).count();
    assert_eq!(eom_noise, leaf_noise);
}

pub(crate) fn test_leaf_with_epsilon(cluster_fn: ClusterFn) {
    let data = vec![
        // Sub-cluster A1
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        vec![0.2, 0.0],
        // Sub-cluster A2
        vec![1.0, 0.0],
        vec![1.1, 0.1],
        vec![1.2, 0.0],
        // Separate cluster B
        vec![10.0, 10.0],
        vec![10.1, 10.1],
        vec![10.2, 10.0],
    ];

    let hp_leaf = HdbscanHyperParams::builder()
        .min_cluster_size(3)
        .cluster_selection_method(ClusterSelectionMethod::Leaf)
        .build();
    let leaf_result = cluster_fn(&Hdbscan::new(&data, hp_leaf)).unwrap();
    let leaf_n_clusters = leaf_result
        .iter()
        .filter(|&&l| l != -1)
        .collect::<HashSet<_>>()
        .len();

    // With a large epsilon, sub-clusters should merge
    let hp_leaf_eps = HdbscanHyperParams::builder()
        .min_cluster_size(3)
        .cluster_selection_method(ClusterSelectionMethod::Leaf)
        .epsilon(2.0)
        .build();
    let eps_result = cluster_fn(&Hdbscan::new(&data, hp_leaf_eps)).unwrap();
    let eps_n_clusters = eps_result
        .iter()
        .filter(|&&l| l != -1)
        .collect::<HashSet<_>>()
        .len();

    assert!(
        eps_n_clusters <= leaf_n_clusters,
        "Epsilon merging should not increase cluster count: {} > {}",
        eps_n_clusters,
        leaf_n_clusters
    );
}

pub(crate) fn test_leaf_allow_single_cluster(cluster_fn: ClusterFn) {
    let data = vec![
        vec![1.1, 1.1],
        vec![1.2, 1.1],
        vec![1.3, 1.2],
        vec![1.1, 1.3],
        vec![1.2, 1.2],
        vec![3.0, 3.0],
    ];

    let hp = HdbscanHyperParams::builder()
        .allow_single_cluster(true)
        .min_cluster_size(4)
        .min_samples(4)
        .nn_algorithm(NnAlgorithm::BruteForce)
        .cluster_selection_method(ClusterSelectionMethod::Leaf)
        .build();
    let clusterer = Hdbscan::new(&data, hp);
    let result = cluster_fn(&clusterer).unwrap();

    // At least one cluster exists (not all noise)
    assert!(
        result.iter().any(|&l| l != -1),
        "Should have at least one cluster"
    );
    // Outlier at index 5 is noise
    assert_eq!(-1, result[5]);
}

pub(crate) fn test_leaf_allow_single_cluster_epsilon(cluster_fn: ClusterFn) {
    let data = vec![
        vec![1.1, 1.1],
        vec![1.2, 1.1],
        vec![1.3, 1.2],
        vec![2.1, 1.3],
        vec![2.2, 1.2],
        vec![2.0, 1.2],
        vec![3.0, 3.0],
    ];

    // Baseline: Leaf without epsilon
    let hp_baseline = HdbscanHyperParams::builder()
        .min_cluster_size(3)
        .cluster_selection_method(ClusterSelectionMethod::Leaf)
        .build();
    let baseline_result = cluster_fn(&Hdbscan::new(&data, hp_baseline)).unwrap();
    let baseline_n_clusters = baseline_result
        .iter()
        .filter(|&&l| l != -1)
        .collect::<HashSet<_>>()
        .len();

    // With allow_single_cluster + epsilon
    let hp = HdbscanHyperParams::builder()
        .allow_single_cluster(true)
        .min_cluster_size(3)
        .epsilon(1.2)
        .cluster_selection_method(ClusterSelectionMethod::Leaf)
        .build();
    let clusterer = Hdbscan::new(&data, hp);
    let result = cluster_fn(&clusterer).unwrap();

    let n_clusters = result
        .iter()
        .filter(|&&l| l != -1)
        .collect::<HashSet<_>>()
        .len();
    assert_eq!(1, n_clusters, "With epsilon, should have exactly 1 cluster");
    assert!(
        n_clusters <= baseline_n_clusters,
        "Cluster count with epsilon {} should not exceed baseline {}",
        n_clusters,
        baseline_n_clusters
    );
    let n_noise = result.iter().filter(|&&l| l == -1).count();
    assert_eq!(1, n_noise, "Should have exactly 1 noise point");
}

pub(crate) fn test_leaf_detailed_probabilities_and_outlier_scores(cluster_fn: DetailedClusterFn) {
    let data = cluster_test_data();
    let hp = HdbscanHyperParams::builder()
        .cluster_selection_method(ClusterSelectionMethod::Leaf)
        .build();
    let clusterer = Hdbscan::new(&data, hp);
    let result = cluster_fn(&clusterer).unwrap();

    // Lengths match data
    assert_eq!(result.labels.len(), data.len());
    assert_eq!(result.probabilities.len(), data.len());
    assert_eq!(result.outlier_scores.len(), data.len());

    // Probabilities in [0,1]; noise=0, clustered>0
    for (i, &p) in result.probabilities.iter().enumerate() {
        assert!(
            p >= 0.0 && p <= 1.0,
            "Probability {} out of range at {}",
            p,
            i
        );
        if result.labels[i] == -1 {
            assert_eq!(p, 0.0, "Noise point {} should have probability 0", i);
        } else {
            assert!(p > 0.0, "Clustered point {} should have probability > 0", i);
        }
    }

    // Outlier scores in [0,1]
    for (i, &s) in result.outlier_scores.iter().enumerate() {
        assert!(
            s >= 0.0 && s <= 1.0,
            "Outlier score {} out of range at {}",
            s,
            i
        );
    }

    // Distant point (index 10) has higher outlier score than average clustered point
    let distant_score = result.outlier_scores[10];
    let avg_clustered: f32 = result
        .outlier_scores
        .iter()
        .enumerate()
        .filter(|(i, _)| result.labels[*i] != -1)
        .map(|(_, &s)| s)
        .sum::<f32>()
        / result.labels.iter().filter(|&&l| l != -1).count() as f32;
    assert!(
        distant_score > avg_clustered,
        "Distant point score {} should exceed average clustered score {}",
        distant_score,
        avg_clustered
    );

    // Condensed tree not empty
    assert!(!result.condensed_tree.is_empty());
}

pub(crate) fn test_leaf_manhattan_distance(cluster_fn: ClusterFn) {
    let data = vec![
        vec![1.3, 1.1],
        vec![1.3, 1.2],
        vec![1.2, 1.2],
        vec![1.0, 1.1],
        vec![0.9, 1.0],
        vec![0.9, 1.0],
        vec![3.7, 4.0],
    ];
    let hp = HdbscanHyperParams::builder()
        .min_cluster_size(3)
        .min_samples(2)
        .dist_metric(DistanceMetric::Manhattan)
        .nn_algorithm(NnAlgorithm::BruteForce)
        .cluster_selection_method(ClusterSelectionMethod::Leaf)
        .build();
    let clusterer = Hdbscan::new(&data, hp);
    let result = cluster_fn(&clusterer).unwrap();

    // First three points form one cluster
    assert_eq!(1, result[..3].iter().collect::<HashSet<_>>().len());
    // Next three points are a second cluster
    assert_eq!(1, result[3..6].iter().collect::<HashSet<_>>().len());
    // The final point is noise
    assert_eq!(-1, result[6]);
}

pub(crate) fn test_leaf_precomputed_distances(cluster_fn: ClusterFn) {
    let dist_matrix = vec![
        vec![0.0, 0.1, 0.2, 0.3, 9.0],
        vec![0.1, 0.0, 0.1, 0.2, 9.0],
        vec![0.2, 0.1, 0.0, 0.1, 9.0],
        vec![0.3, 0.2, 0.1, 0.0, 9.0],
        vec![9.0, 9.0, 9.0, 9.0, 9.0],
    ];
    let hp = HdbscanHyperParams::builder()
        .dist_metric(DistanceMetric::Precalculated)
        .allow_single_cluster(true)
        .min_cluster_size(2)
        .min_samples(1)
        .cluster_selection_method(ClusterSelectionMethod::Leaf)
        .build();
    let clusterer = Hdbscan::new(&dist_matrix, hp);
    let result = cluster_fn(&clusterer).unwrap();

    // At least 1 cluster
    let unique_clusters: HashSet<_> = result.iter().filter(|&&x| x != -1).collect();
    assert!(
        !unique_clusters.is_empty(),
        "Should have at least 1 cluster"
    );
    // Last point is noise
    assert_eq!(result[result.len() - 1], -1, "Last point should be noise");
}

pub(crate) fn test_default_is_eom(cluster_fn: ClusterFn) {
    let data = vec![
        // Tight sub-cluster 1
        vec![0.0, 0.0],
        vec![0.05, 0.05],
        vec![0.1, 0.0],
        // Tight sub-cluster 2
        vec![2.0, 0.0],
        vec![2.05, 0.05],
        vec![2.1, 0.0],
        // Separate cluster
        vec![20.0, 20.0],
        vec![20.05, 20.05],
        vec![20.1, 20.0],
    ];

    // Default (no cluster_selection_method specified)
    let hp_default = HdbscanHyperParams::builder().min_cluster_size(3).build();
    let default_result = cluster_fn(&Hdbscan::new(&data, hp_default)).unwrap();

    // Explicit EOM
    let hp_eom = HdbscanHyperParams::builder()
        .min_cluster_size(3)
        .cluster_selection_method(ClusterSelectionMethod::Eom)
        .build();
    let eom_result = cluster_fn(&Hdbscan::new(&data, hp_eom)).unwrap();

    assert_eq!(
        default_result, eom_result,
        "Default cluster selection should produce identical results to explicit EOM"
    );
}

pub(crate) fn test_leaf_finer_grained(cluster_fn: ClusterFn) {
    let data = vec![
        // Tight sub-cluster 1
        vec![0.0, 0.0],
        vec![0.05, 0.05],
        vec![0.1, 0.0],
        // Tight sub-cluster 2
        vec![2.0, 0.0],
        vec![2.05, 0.05],
        vec![2.1, 0.0],
        // Separate cluster
        vec![20.0, 20.0],
        vec![20.05, 20.05],
        vec![20.1, 20.0],
    ];

    let hp_eom = HdbscanHyperParams::builder()
        .min_cluster_size(3)
        .cluster_selection_method(ClusterSelectionMethod::Eom)
        .build();
    let eom_result = cluster_fn(&Hdbscan::new(&data, hp_eom)).unwrap();
    let eom_n_clusters = eom_result
        .iter()
        .filter(|&&l| l != -1)
        .collect::<HashSet<_>>()
        .len();

    let hp_leaf = HdbscanHyperParams::builder()
        .min_cluster_size(3)
        .cluster_selection_method(ClusterSelectionMethod::Leaf)
        .build();
    let leaf_result = cluster_fn(&Hdbscan::new(&data, hp_leaf)).unwrap();
    let leaf_n_clusters = leaf_result
        .iter()
        .filter(|&&l| l != -1)
        .collect::<HashSet<_>>()
        .len();

    assert!(
        leaf_n_clusters >= eom_n_clusters,
        "Leaf should find at least as many clusters as EOM: {} < {}",
        leaf_n_clusters,
        eom_n_clusters
    );
}
