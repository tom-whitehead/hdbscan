use std::error::Error;
use std::fmt::{Display, Formatter};

/// Possible errors that arise due to issues with HDBSCAN input data.
#[derive(Debug, Clone)]
pub enum HdbscanError {
    EmptyDataset,
    WrongDimension(String),
    NonFiniteCoordinate(String)
}

impl Error for HdbscanError {}

impl Display for HdbscanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            HdbscanError::EmptyDataset => String::from("The dataset provided is empty"),
            HdbscanError::WrongDimension(msg) => 
                format!("Input vectors have mismatched dimensions: {msg}"),
            HdbscanError::NonFiniteCoordinate(msg) => 
                format!("Non finite coordinate: {msg}"),
        };
        write!(f, "{message}")
    }
}
