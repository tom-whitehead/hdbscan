use num_traits::Float;

/// Possible distance metrics that can be used in the HDBSCAN algorithm when
/// calculating the distances between data points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Chebyshev,
    Haversine,
}

impl DistanceMetric {
    pub(crate) fn calc_dist<T: Float>(&self, a: &[T], b: &[T]) -> T {
        match *self {
            Self::Euclidean => euclidean_distance(a, b),
            Self::Haversine => haversine_distance(a, b),
            Self::Manhattan => manhattan_distance(a, b),
            Self::Chebyshev => chebyshev_distance(a, b),
        }
    }
}

pub(crate) fn get_dist_func<T: Float>(metric: &DistanceMetric) -> impl Fn(&[T], &[T]) -> T {
    match metric {
        DistanceMetric::Euclidean => euclidean_distance,
        DistanceMetric::Manhattan => manhattan_distance,
        DistanceMetric::Chebyshev => chebyshev_distance,
        DistanceMetric::Haversine => haversine_distance,
    }
}

pub(crate) fn haversine_distance<T: Float>(a: &[T], b: &[T]) -> T {
    assert_inputs(a, b);
    assert_eq!(a.len(), 2, "Need both lat and lon");

    let (lat1, lon1) = (a[0], a[1]);
    let (lat2, lon2) = (b[0], b[1]);

    // Earth's radius in meters
    let r = T::from(6371000.0).unwrap();

    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();

    let lat1 = lat1.to_radians();
    let lat2 = lat2.to_radians();

    let a = (dlat / T::from(2.0).unwrap()).sin().powi(2)
        + lat1.cos() * lat2.cos() * (dlon / T::from(2.0).unwrap()).sin().powi(2);
    let c = T::from(2.0).unwrap() * a.sqrt().asin();

    r * c
}

pub(crate) fn euclidean_distance<T: Float>(a: &[T], b: &[T]) -> T {
    assert_inputs(a, b);
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x) - (*y)) * ((*x) - (*y)))
        .fold(T::zero(), std::ops::Add::add)
        .sqrt()
}

pub(crate) fn manhattan_distance<T: Float>(a: &[T], b: &[T]) -> T {
    assert_inputs(a, b);
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x) - (*y)).abs())
        .fold(T::zero(), std::ops::Add::add)
}

pub(crate) fn chebyshev_distance<T: Float>(a: &[T], b: &[T]) -> T {
    assert_inputs(a, b);
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x) - (*y)).abs())
        .fold(T::zero(), T::max)
}

fn assert_inputs<T: Float>(a: &[T], b: &[T]) {
    assert_eq!(a.len(), b.len());
    assert!(!a.is_empty());
}
