use num_traits::Float;
use crate::{distance, DistanceMetric};

pub(crate) enum NearestNeighbour {
  KdTree,
}

impl NearestNeighbour {
  pub(crate) fn calc_dist_to_kth_neighbours<'a, T: Float>(
    &self, data: &'a Vec<Vec<T>>, k: usize, dist_metric: &DistanceMetric) -> Vec<T> {
    
    let dist_func = distance::get_dist_func::<T>(&dist_metric);
    
    match self {
      NearestNeighbour::KdTree => calc_distances_kd_tree(&data, k, &dist_func)
    }
  }
}

fn calc_distances_kd_tree<T, F>(data: &Vec<Vec<T>>, k: usize, dist_func: F) -> Vec<T> 
where
    T: Float,
    F: Fn(&[T], &[T]) -> T
{
  let mut tree: kdtree::KdTree<T, usize, &Vec<T>> = kdtree::KdTree::new(data[0].len());
  data.iter().enumerate()
      .for_each(|(n, datapoint)| tree.add(datapoint, n).unwrap());

  data.iter()
      .map(|datapoint| {
        let result = tree.nearest(datapoint, k, &dist_func).unwrap();
        result.into_iter()
            .map(|(dist, _idx)| dist)
            .last()
            .unwrap()
      })
      .collect()
}
