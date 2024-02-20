# Hierarchical Density-Based Spatial Clustering of Applications with Noise ("HDBSCAN")

HDBSCAN clustering algorithm in pure Rust. Generic over floating point numeric types.

HDBSCAN is a powerful clustering algorithm that can be used to effectively find clusters in real world data.
The main benefits of HDBSCAN are that:
 1. It does not assume that all data points belong to a cluster, as many clustering algorithms do. I.e. a data set
    can contain "noise" points. This is important for modelling real world data, which is inherently noisy;
 2. It allows clusters of varying densities, unlike the plain DBSCAN algorithm which uses a static density
    threshold. The winning clusters are those that persist the longest at a all densities. This is also crucial
    for modelling real world data; and
 3. It makes no assumptions about the number of clusters there have to be, unlike KMeans clustering. The algorithm
    will select just select the clusters that are the most persistent at all densities.

This implementation owes a debt to the Python scikit-learn implementation of this algorithm, without which this
algorithm would not have been possible. The "How HDBSCAN works" article below is invaluable in understanding this
algorithm better.

# Current state
Several variations of HDBSCAN are possible. Notably, a nearest neighbours algorithm is used to calculate the distance 
of a point to its Kth neighbour. This is a crucial input to calculate the density of points in the vector space. 
Currently, this implementation only supports the K-d Tree nearest neighbours algorithm to do this. While K-d Tree is 
the best candidate for most uses cases, in the future I hope to support other nearest neighbour algorithms to make this
implementation more flexible (as per the scikit-learn Python implementation).

Further, this implementation uses Prim's algorithm to find the minimum spanning tree of the points. Prim's algorithm 
will perform the best for dense vectors and therefore most uses cases. However, Kruskal's algorithm is another
possibility for this, that would perform better on sparse vectors.

I also hope to be able to offer more hyper parameters to tune the algorithm.

# Usage
### With default hyper parameters
```rust
use hdbscan::Hdbscan;

let data: Vec<Vec<f32>> = vec![
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
];
let clusterer = Hdbscan::default(&data);
let result = clusterer.cluster().unwrap();
assert_eq!(result, vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1]);
```

### With custom hyper parameters
```rust
let data: Vec<Vec<f32>> = vec![
    vec![1.3, 1.1],
    vec![1.3, 1.2],
    vec![1.0, 1.1],
    vec![1.2, 1.2],
    vec![0.9, 1.0],
    vec![0.9, 1.0],
    vec![3.7, 4.0],
    vec![3.9, 3.9],
];
let hyper_params = HdbscanHyperParams::builder()
    .min_cluster_size(3)
    .min_samples(2)
    .dist_metric(DistanceMetric::Manhattan)
    .build();
let clusterer = Hdbscan::new(&data, hyper_params);
let result = clusterer.cluster().unwrap();
assert_eq!(result, vec![0, 0, 1, 0, 1, 1, -1, -1]);
```

# License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the 
MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, 
or distributed except according to those terms.
