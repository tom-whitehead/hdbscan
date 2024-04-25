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