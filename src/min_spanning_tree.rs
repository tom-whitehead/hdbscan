use crate::data_wrappers::MSTEdge;
use crate::DistanceMetric;
use num_traits::Float;

pub(crate) trait MinSpanningTree<'a, T> {
    fn compute(&self) -> Vec<MSTEdge<T>>;
}

#[derive(Clone, Debug)]
struct MinSpanningTreeCommon<'a, T> {
    data: &'a [Vec<T>],
    dist_metric: DistanceMetric,
    core_distances: &'a [T],
    n_samples: usize,
}

impl<'a, T: Float> MinSpanningTreeCommon<'a, T> {
    fn new(data: &'a [Vec<T>], dist_metric: DistanceMetric, core_distances: &'a [T]) -> Self {
        MinSpanningTreeCommon {
            data,
            dist_metric,
            core_distances,
            n_samples: data.len(),
        }
    }

    fn calc_mutual_reachability_dist(&self, a: usize, b: usize) -> T {
        let core_dist_a = self.core_distances[a];
        let core_dist_b = self.core_distances[b];
        let dist_a_b = if self.dist_metric == DistanceMetric::Precalculated {
            self.data[a][b]
        } else {
            self.dist_metric.calc_dist(&self.data[a], &self.data[b])
        };
        core_dist_a.max(core_dist_b).max(dist_a_b)
    }

    fn sort_mst_by_dist(&self, min_spanning_tree: &mut [MSTEdge<T>]) {
        min_spanning_tree
            .sort_by(|a, b| a.distance.partial_cmp(&b.distance).expect("Invalid floats"));
    }
}

#[cfg(feature = "serial")]
pub(crate) mod serial {
    use super::*;
    use crate::data_wrappers::MSTEdge;
    use num_traits::Float;

    #[derive(Clone, Debug)]
    pub(crate) struct PrimsMinSpanningTree<'a, T> {
        common: MinSpanningTreeCommon<'a, T>,
    }

    impl<'a, T: Float> PrimsMinSpanningTree<'a, T> {
        pub(crate) fn new(
            data: &'a [Vec<T>],
            dist_metric: DistanceMetric,
            core_distances: &'a [T],
        ) -> Self {
            let common = MinSpanningTreeCommon::new(data, dist_metric, core_distances);
            PrimsMinSpanningTree { common }
        }
    }

    impl<'a, T: Float> MinSpanningTree<'a, T> for PrimsMinSpanningTree<'a, T> {
        fn compute(&self) -> Vec<MSTEdge<T>> {
            let n_samples = self.common.n_samples;

            let mut in_tree = vec![false; n_samples];
            let mut distances = vec![T::infinity(); n_samples];
            distances[0] = T::zero();

            let mut mst = Vec::with_capacity(n_samples);

            let mut left_node_id = 0;
            let mut right_node_id = 0;

            for _ in 1..n_samples {
                in_tree[left_node_id] = true;
                let mut current_min_dist = T::infinity();

                for i in 0..n_samples {
                    if in_tree[i] {
                        continue;
                    }
                    let mrd = self.common.calc_mutual_reachability_dist(left_node_id, i);
                    if mrd < distances[i] {
                        distances[i] = mrd;
                    }
                    if distances[i] < current_min_dist {
                        right_node_id = i;
                        current_min_dist = distances[i];
                    }
                }
                mst.push(MSTEdge {
                    left_node_id,
                    right_node_id,
                    distance: current_min_dist,
                });
                left_node_id = right_node_id;
            }
            self.common.sort_mst_by_dist(&mut mst);
            mst
        }
    }
}

#[cfg(feature = "parallel")]
pub(crate) mod parallel {
    use super::*;
    use crate::data_wrappers::MSTEdge;
    use num_traits::Float;
    use rayon::prelude::*;

    #[derive(Clone, Debug)]
    pub(crate) struct PrimsMinSpanningTreePar<'a, T> {
        common: MinSpanningTreeCommon<'a, T>,
    }

    impl<'a, T: Float + Send + Sync> PrimsMinSpanningTreePar<'a, T> {
        pub(crate) fn new(
            data: &'a [Vec<T>],
            dist_metric: DistanceMetric,
            core_distances: &'a [T],
        ) -> Self {
            let common = MinSpanningTreeCommon::new(data, dist_metric, core_distances);
            PrimsMinSpanningTreePar { common }
        }
    }

    impl<'a, T: Float + Send + Sync> MinSpanningTree<'a, T> for PrimsMinSpanningTreePar<'a, T> {
        fn compute(&self) -> Vec<MSTEdge<T>> {
            let n_samples = self.common.n_samples;

            let mut in_tree = vec![false; n_samples];
            let mut distances = vec![T::infinity(); n_samples];
            distances[0] = T::zero();

            let mut mst = Vec::with_capacity(n_samples);

            let mut left_node_id = 0;

            for _ in 1..n_samples {
                in_tree[left_node_id] = true;

                let (min_idx, min_dist) = distances
                    .par_iter_mut()
                    .enumerate()
                    .filter_map(|(i, dist)| {
                        if in_tree[i] {
                            None
                        } else {
                            let mrd = self.common.calc_mutual_reachability_dist(left_node_id, i);
                            if mrd < *dist {
                                *dist = mrd;
                            }
                            Some((i, *dist))
                        }
                    })
                    .min_by(|(_, dist_a), (_, dist_b)| {
                        dist_a.partial_cmp(dist_b).expect("Invalid floats")
                    })
                    .expect("Malformed distance array");

                mst.push(MSTEdge {
                    left_node_id,
                    right_node_id: min_idx,
                    distance: min_dist,
                });
                left_node_id = min_idx;
            }

            self.common.sort_mst_by_dist(&mut mst);
            mst
        }
    }
}
