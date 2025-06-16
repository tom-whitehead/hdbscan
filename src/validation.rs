use crate::{DistanceMetric, HdbscanError, HdbscanHyperParams};
use num_traits::Float;
use std::f64::consts::PI;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DataValidator<'a, T> {
    data: &'a [Vec<T>],
    hp: &'a HdbscanHyperParams,
}

impl<'a, T: Float> DataValidator<'a, T> {
    pub(crate) fn new(data: &'a [Vec<T>], hp: &'a HdbscanHyperParams) -> Self {
        Self { data, hp }
    }

    pub(crate) fn validate_input_data(&self) -> Result<(), HdbscanError> {
        // TODO: Replace with checking for at least 2 points?
        if self.data.is_empty() {
            return Err(HdbscanError::EmptyDataset);
        }
        let dims_0th = self.data[0].len();
        for (n, datapoint) in self.data.iter().enumerate() {
            for element in datapoint {
                if element.is_infinite() {
                    return Err(HdbscanError::NonFiniteCoordinate(format!(
                        "{n}th vector contains non-finite element(s)"
                    )));
                }
            }
            let dims_nth = datapoint.len();
            if dims_nth != dims_0th {
                return Err(HdbscanError::WrongDimension(format!(
                    "Oth data point has {dims_0th} dimensions, but {n}th has {dims_nth}"
                )));
            }
        }
        if self.hp.dist_metric == DistanceMetric::Cylindrical {
            self.validate_cylindrical_coords()?
        }
        if self.hp.dist_metric == DistanceMetric::Haversine {
            self.validate_geographical_coords()?
        }
        if self.hp.dist_metric == DistanceMetric::Precalculated {
            self.validate_precomputed_distances()?
        }

        Ok(())
    }

    fn validate_cylindrical_coords(&self) -> Result<(), HdbscanError> {
        let n_dim = self.data[0].len();
        if n_dim != 3 {
            return Err(HdbscanError::WrongDimension(format!(
                "Cylindrical coordinates should have three dimensions (ρ, φ, z), not {n_dim}"
            )));
        }
        for datapoint in self.data {
            let (dim1, dim2, dim3) = (datapoint[0], datapoint[1], datapoint[2]);
            if dim1 < T::zero() || dim1 > T::one() {
                return Err(HdbscanError::WrongDimension(String::from(
                    "Dimension 1 of cylindrical coordinates should be a percent in range 0 to 1",
                )));
            }
            if dim2 < T::zero() || dim2 > T::from(PI * 2.0).unwrap() {
                return Err(HdbscanError::WrongDimension(String::from(
                    "Dimension 2 of cylindrical coordinates should be a radian in range 0 to 2π",
                )));
            }
            if dim3 < T::zero() || dim3 > T::one() {
                return Err(HdbscanError::WrongDimension(String::from(
                    "Dimension 3 of cylindrical coordinates should be a percent in range 0 to 1",
                )));
            }
        }
        Ok(())
    }

    fn validate_geographical_coords(&self) -> Result<(), HdbscanError> {
        let n_dim = self.data[0].len();
        if n_dim != 2 {
            return Err(HdbscanError::WrongDimension(format!(
                "Geographical coordinates should have two dimensions (lat, lon), not {n_dim}"
            )));
        }
        for datapoint in self.data {
            let (lat, lon) = (datapoint[0], datapoint[1]);
            if lat < T::from(-90.0).unwrap() || lat > T::from(90.0).unwrap() {
                return Err(HdbscanError::WrongDimension(String::from(
                    "Dimension 1 of geographical coordinates used in with Haversine distance \
                    should be a latitude in range -90 to 90",
                )));
            }
            if lon < T::from(-180.0).unwrap() || lon > T::from(180.0).unwrap() {
                return Err(HdbscanError::WrongDimension(String::from(
                    "Dimension 2 of geographical coordinates used in with Haversine distance \
                    should be a longitude in range -180 to 180",
                )));
            }
        }
        Ok(())
    }

    fn validate_precomputed_distances(&self) -> Result<(), HdbscanError> {
        if !self.is_symmetrical_matrix() {
            return Err(HdbscanError::WrongDimension(String::from(
                "Pre-calculated distances must be a symmetrical distance matrix",
            )));
        }
        Ok(())
    }

    fn is_symmetrical_matrix(&self) -> bool {
        let n = self.data.len();
        if self.data.iter().any(|row| row.len() != n) {
            return false;
        }
        for i in 0..n {
            for j in 0..n {
                if (self.data[i][j] - self.data[j][i]).abs() > T::epsilon() {
                    return false;
                }
            }
        }
        true
    }
}
