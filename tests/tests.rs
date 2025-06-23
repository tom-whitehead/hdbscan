use hdbscan::{Center, DistanceMetric, Hdbscan, HdbscanError, HdbscanHyperParams, NnAlgorithm};
use num_traits::Float;
use std::collections::HashSet;

#[test]
fn cluster() {
    let data = cluster_test_data();
    let clusterer = Hdbscan::default_hyper_params(&data);
    let result = clusterer.cluster().unwrap();
    // First five points form one cluster
    assert_eq!(1, result[..5].iter().collect::<HashSet<_>>().len());
    // Next five points are a second cluster
    assert_eq!(1, result[5..10].iter().collect::<HashSet<_>>().len());
    // The final point is noise
    assert_eq!(-1, result[10]);
}

#[test]
fn builder_cluster() {
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
    let result = clusterer.cluster().unwrap();
    // First three points form one cluster
    assert_eq!(1, result[..3].iter().collect::<HashSet<_>>().len());
    // Next three points are a second cluster
    assert_eq!(1, result[3..6].iter().collect::<HashSet<_>>().len());
    // The final point is noise
    assert_eq!(-1, result[6]);
}

#[test]
fn single_cluster() {
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
    let result = clusterer.cluster().unwrap();

    let unique_clusters: HashSet<_> = result.iter().filter(|&&x| x != -1).collect();
    assert_eq!(1, unique_clusters.len());

    let noise_points: Vec<_> = result.iter().filter(|&&x| x == -1).collect();
    assert_eq!(1, noise_points.len());
}

#[test]
fn single_cluster_epsilon_search() {
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
    let result = clusterer.cluster().unwrap();

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
    let result = clusterer.cluster().unwrap();

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

#[test]
fn single_root_cluster_only_epsilon_search() {
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
    let result = clusterer.cluster().unwrap();

    let unique_clusters = result
        .iter()
        .filter(|&&label| label != -1)
        .collect::<HashSet<_>>();
    assert_eq!(1, unique_clusters.len());
    let n_noise = result.iter().filter(|&&label| label == -1).count();
    assert_eq!(1, n_noise);
}

#[test]
fn empty_data() {
    let data: Vec<Vec<f32>> = Vec::new();
    let clusterer = Hdbscan::default_hyper_params(&data);
    let result = clusterer.cluster();
    assert!(matches!(result, Err(HdbscanError::EmptyDataset)));
}

#[test]
fn non_finite_coordinate() {
    let data = vec![vec![1.5, f32::infinity()]];
    let clusterer = Hdbscan::default_hyper_params(&data);
    let result = clusterer.cluster();
    assert!(matches!(result, Err(HdbscanError::NonFiniteCoordinate(..))));
}

#[test]
fn mismatched_dimensions() {
    let data = vec![vec![1.5, 2.2], vec![1.0, 1.1], vec![1.2]];
    let clusterer = Hdbscan::default_hyper_params(&data);
    let result = clusterer.cluster();
    assert!(matches!(result, Err(HdbscanError::WrongDimension(..))));
}

#[test]
fn calc_centroids() {
    let data = cluster_test_data();
    let clusterer = Hdbscan::default_hyper_params(&data);
    let labels = clusterer.cluster().unwrap();
    let centroids = clusterer.calc_centers(Center::Centroid, &labels).unwrap();
    assert_eq!(2, centroids.len());
    assert!(centroids.contains(&vec![3.8, 4.0]) && centroids.contains(&vec![1.12, 1.34]));
}

#[test]
fn calc_medoids() {
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
    let result = clusterer.cluster().unwrap();
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

fn cluster_test_data() -> Vec<Vec<f32>> {
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

#[test]
fn test_nyc_landmarks_haversine() {
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
    let result = clusterer.cluster().unwrap();

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

#[test]
fn geo_cluster_across_180th_meridian() {
    let data = vec![
        vec![-16.8410, 179.9813],  // Taveuni, Fiji
        vec![-16.7480, -179.9670], // Qamea, Fiji 
        vec![51.5085, -0.1257], // London - noise
    ];
    
    let hyper_params = HdbscanHyperParams::builder()
        .dist_metric(DistanceMetric::Haversine)
        .allow_single_cluster(true)
        .min_cluster_size(2)
        .min_samples(1)
        .build();
    
    let clusterer = Hdbscan::new(&data, hyper_params);
    let labels = clusterer.cluster().unwrap();
    
    // There is only one cluster
    assert_eq!(1, labels.iter().filter(|&&x| x != -1).collect::<HashSet<_>>().len());
    // The last point is noise
    assert_eq!(-1, labels[2]);
    
    let centroids = clusterer.calc_centers(Center::GeoCentroid, &labels).unwrap();
    let cluster_longitude = centroids[0][1];

    // The cluster centroid is not impacted by the longitudes being either side 
    // of the 180th meridian
    assert!(cluster_longitude > 179.0 || cluster_longitude < -179.0);
}

#[test]
fn test_cylindrical_hsv_colours() {
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
    let result = clusterer.cluster().unwrap();

    // Blues all form one cluster
    assert_eq!(1, result[..3].iter().collect::<HashSet<_>>().len());
    // Greens are a second cluster
    assert_eq!(1, result[3..6].iter().collect::<HashSet<_>>().len());
    // The final red point is noise
    assert_eq!(-1, result[6]);
}

#[test]
fn test_precomputed_distances() {
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
    let result = clusterer.cluster().unwrap();

    // Check that we have 1 cluster and 1 noise point
    let unique_clusters: HashSet<_> = result.iter().filter(|&&x| x != -1).collect();
    assert_eq!(unique_clusters.len(), 1, "Should have 1 distinct cluster");
    assert_eq!(result[result.len() - 1], -1, "Should have 0 noise points");
}
