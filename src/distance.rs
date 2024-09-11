use num_traits::Float;

/// Possible distance metrics that can be used in the HDBSCAN algorithm when
/// calculating the distances between data points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    /// Distance between two points as the maximum difference over any of their dimensions.
    /// Also known as L-infinity norm.
    Chebyshev,
    /// The distance between points in a cylindrical coordinate system where the input data is
    /// three-dimensional in the form (ρ , φ , z), or (radial distance, angular coordinate, height). 
    /// Degrees should be in radians and distances percents. If you're using HSV or HSL colour 
    /// systems, the coordinates will need to be re-ordered to SHV or SHL, respectively.
    Cylindrical,
    /// The length of the line between two points. Also known as L2 norm.
    Euclidean,
    /// The angular distance between two points on the surface of the Earth.
    /// Requires two-dimensional input data, where the first coordinate is latitude and
    /// the second is longitude.
    Haversine,
    /// The sum of all absolute differences between each dimension of two points.
    /// Also known as L1 norm or city block.
    Manhattan,
}

impl DistanceMetric {
    pub(crate) fn calc_dist<T: Float>(&self, a: &[T], b: &[T]) -> T {
        match *self {
            Self::Chebyshev => chebyshev_distance(a, b),
            Self::Cylindrical => cylindrical_distance(a, b),
            Self::Euclidean => euclidean_distance(a, b),
            Self::Haversine => haversine_distance(a, b),
            Self::Manhattan => manhattan_distance(a, b),
        }
    }
}

pub(crate) fn get_dist_func<T: Float>(metric: &DistanceMetric) -> impl Fn(&[T], &[T]) -> T {
    match metric {
        DistanceMetric::Chebyshev => chebyshev_distance,
        DistanceMetric::Cylindrical => cylindrical_distance,
        DistanceMetric::Euclidean => euclidean_distance,
        DistanceMetric::Haversine => haversine_distance,
        DistanceMetric::Manhattan => manhattan_distance,
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

pub(crate) fn cylindrical_distance<T: Float>(a: &[T], b: &[T]) -> T {
    assert_inputs(a, b);
    assert_eq!(a.len(), 3, "Cylindrical coordinates must have three dimensions (ρ, φ, z)");

    let (r1, theta1, z1) = (a[0], a[1], a[2]);
    let (r2, theta2, z2) = (b[0], b[1], b[2]);

    let theta_diff = theta2 - theta1;
    let radial_component = r1 * r1 + r2 * r2 - T::from(2.0).unwrap() * r1 * r2 * theta_diff.cos();
    let vertical_component = (z2 - z1) * (z2 - z1);

    (radial_component + vertical_component).sqrt()
}

fn assert_inputs<T: Float>(a: &[T], b: &[T]) {
    assert_eq!(a.len(), b.len(), "Two vectors need to have the same dimensions");
    assert!(!a.is_empty(), "The vectors cannot be empty");
}
