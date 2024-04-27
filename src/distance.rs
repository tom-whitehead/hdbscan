use num_traits::Float;

/// Possible distance metrics that can be used in the HDBSCAN algorithm when
/// calculating the distances between data points.
#[derive(Copy, Clone)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Chebyshev,
}

impl DistanceMetric {
    pub(crate) fn calc_dist<T: Float>(&self, a: &[T], b: &[T]) -> T {
        match *self {
            Self::Euclidean => { euclidean_distance(a, b) }
            Self::Manhattan => { manhattan_distance(a, b) }
            Self::Chebyshev => { chebyshev_distance(a, b) }
        }
    }
}

pub(crate) fn get_dist_func<'a, T: Float>(metric: &DistanceMetric) -> impl Fn(&[T], &[T]) -> T {
    match metric {
        DistanceMetric::Euclidean => { euclidean_distance }
        DistanceMetric::Manhattan => { manhattan_distance }
        DistanceMetric::Chebyshev => { chebyshev_distance }
    }
}

pub(crate) fn euclidean_distance<T: Float>(a: &[T], b: &[T]) -> T {
    assert_inputs(&a, &b);
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x) - (*y)) * ((*x) - (*y)))
        .fold(T::zero(), std::ops::Add::add)
        .sqrt()
}

pub(crate) fn manhattan_distance<T: Float>(a: &[T], b: &[T]) -> T {
    assert_inputs(&a, &b);
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x) - (*y)).abs())
        .fold(T::zero(), std::ops::Add::add)
}

pub(crate) fn chebyshev_distance<T: Float>(a: &[T], b: &[T]) -> T {
    assert_inputs(&a, &b);
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x) - (*y)).abs())
        .max_by(|a, b| a.partial_cmp(b).expect("inputs were validated"))
        .expect("inputs were validated")
}

fn assert_inputs<T: Float>(a: &[T], b: &[T]) {
    assert_eq!(a.len(), b.len());
    assert!(!a.is_empty());
}
