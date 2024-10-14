use thiserror::Error;

/// Possible errors that arise due to issues with HDBSCAN input data.
#[derive(Error, Debug)]
pub enum HdbscanError {
    #[error("Could not find node")]
    NodeNotFound,
    #[error("The dataset provided is empty")]
    EmptyDataset,
    #[error("Input vectors have mismatched dimensions {0}")]
    WrongDimension(String),
    #[error("Non finite coordinates: {0}")]
    NonFiniteCoordinate(String),
}
