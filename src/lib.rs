//! Hierarchical Density-Based Spatial Clustering of Applications with Noise ("HDBSCAN") clustering
//! algorithm in Rust. Generic over floating point numeric types.
//!
//! HDBSCAN is a powerful clustering algorithm that can be used to effectively find clusters in
//! real world data. The main benefits of HDBSCAN are that:
//!  1. It does not assume that all data points belong to a cluster, as many clustering algorithms
//!     do. I.e. a data set can contain "noise" points. This is important for modelling real world
//!     data, which is inherently noisy;
//!  2. It allows clusters of varying densities, unlike the plain DBSCAN algorithm which uses a
//!     static density threshold. The winning clusters are those that persist the longest at all
//!     densities. This is also crucial for modelling real world data; and
//!  3. It makes no assumptions about the number of clusters there have to be, unlike KMeans
//!     clustering. The algorithm will just select the clusters that are the most persistent
//!     at all densities.
//!
//! This implementation owes a debt to the Python scikit-learn implementation of this algorithm,
//! without which this algorithm would not have been possible. The "How HDBSCAN works" article
//! below is invaluable in understanding this algorithm better.
//!
//! # Examples
//!```
//!use std::collections::HashSet;
//!use hdbscan::Hdbscan;
//!
//!let data: Vec<Vec<f32>> = vec![
//!    vec![1.5, 2.2],
//!    vec![1.0, 1.1],
//!    vec![1.2, 1.4],
//!    vec![0.8, 1.0],
//!    vec![1.1, 1.0],
//!    vec![3.7, 4.0],
//!    vec![3.9, 3.9],
//!    vec![3.6, 4.1],
//!    vec![3.8, 3.9],
//!    vec![4.0, 4.1],
//!    vec![10.0, 10.0],
//!];
//!let clusterer = Hdbscan::default_hyper_params(&data);
//!let labels = clusterer.cluster().unwrap();
//!//First five points form one cluster
//!assert_eq!(1, labels[..5].iter().collect::<HashSet<_>>().len());
//!// Next five points are a second cluster
//!assert_eq!(1, labels[5..10].iter().collect::<HashSet<_>>().len());
//!// The final point is noise
//!assert_eq!(-1, labels[10]);
//!```
//!
//! # References
//! * [Campello, R.J.G.B.; Moulavi, D.; Sander, J. Density-based clustering based on hierarchical density estimates.](https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14)
//! * [How HDBSCAN Works](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html)

pub use crate::centers::Center;
pub use crate::core_distances::NnAlgorithm;
pub use crate::distance::DistanceMetric;
pub use crate::error::HdbscanError;
pub use crate::hdbscan::Hdbscan;
pub use crate::hyper_parameters::{HdbscanHyperParams, HyperParamBuilder};

mod centers;
mod core_distances;
mod data_wrappers;
mod distance;
mod error;
mod hdbscan;
mod hyper_parameters;
mod union_find;
mod validation;
