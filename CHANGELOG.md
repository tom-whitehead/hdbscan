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