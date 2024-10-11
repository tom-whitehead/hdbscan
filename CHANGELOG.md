# Version 0.8.2 2024-10-11
## Changes
- Fix for a bug that occurred when `allow_single_cluster` was set to true. If you want to run clustering with 
  this hyper parameter set to true, then it is strongly recommended that you use this version or higher.
- Cluster centers for geographical clusters.

# Version 0.8.1 2024-09-18
## Changes
- Consistency in cluster labels between runs of the algorithm. If you ran clustering on the same data numerous times,
  previously the same cluster would likely have had a different label each time. Now there is more stability in the 
  labelling between runs.
- Renamed the `default` constructor to `default_hyper_params` and deprecated the former. This is to avoid a name clash
  with Rust's `Default` trait.

# Version 0.8.0 2024-09-12
## Changes
- Two new distance metrics - Haversine distance for clustering geographical data on the Earth's surface and Cylindrical
  distance for clustering cylindrical coordinates like HSV colours.
- General refactoring.

# Version 0.7.0 2024-08-15
## Changes
- Added the `epsilon` hyper parameter.  In HDBSCAN, each cluster has an epsilon value, which is the distance threshold 
  at which the cluster first appears, or the level that it splits from its parent cluster. Setting this epsilon 
  parameter creates a distance threshold that a cluster must have existed at to be selected. If a cluster does not meet 
  this threshold, it will be merged and the algorithm will search for a parent cluster that does meet the threshold.
- Derived `Debug` for all exported types. 

# Version 0.6.0 2024-04-25
## Changes
- Added the ability to specify the nearest neighbour algorithm in the `HyperParamBuilder` and also implemented a brute
  force nearest neighbour algorithm. Internally, HDBSCAN calculates a density measure called core distances, 
  which is defined as the distance of a data point to it's kth neighbour. Now it is possible to choose the nearest 
  neighbour algorithm using the `NnAlgorithm` enum, being any of `Auto`, `KdTree` or `BruteForce`. `Auto` will choose 
  the algorithm internally based on the nature of the data.

# Version 0.5.0 2024-04-01
## Changes
- Performance gain, which allows the algorithm to scale better to larger datasets as fewer operations are now 
  required to calculate the minimum spanning tree of the data points in a data set. An implication of this change
  is that the order in which the labels are applied to data points will change from run to run.

# Version 0.4.1 2024-03-22
## Changes
- Critical fix for a bug where clusters greater than one node down in the tree were not being deselected in 
  cases where the "grandparent" cluster was the most stable and should have been selected.
- Further bug fix for a panic that occurred when `allow_single_cluster == true`.

# Version 0.4.0 2024-03-12
## Changes
- Added support for calculating cluster centroids once clustering is done, with `Hdbscab::calc_centers` method

# Version 0.3.0 2024-02-20
## Changes
 - Added `max_cluster_size` hyper parameter, with support in the hyper parameter builder 
 - Improved read me documentation on current state of the algorithm