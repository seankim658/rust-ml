//! # Dataset Module
//!
//! This module defines the simple structure for datasets.

use crate::base::error::{Error, ErrorKind};
use crate::base::MLResult;

use csv::ReaderBuilder;
use serde::Deserialize;
use std::fmt::Debug;
use std::fs::File;
use std::path::Path;

/// Loads the infamous UCI Iris dataset.
pub mod iris;

/// Struct for a datatset.
///
/// Fields:
/// - data: The features.
/// - target: The targets.
/// - columns: Vector of the feature column names (in order relative to data).
///
#[derive(Clone, Debug)]
pub struct Dataset<X, Y>
where
    X: Clone + Debug,
    Y: Clone + Debug,
{
    data: X,
    target: Y,
    columns: Vec<String>,
}

/// Constructor and some getters for the Dataset struct.
impl<X, Y> Dataset<X, Y>
where
    X: Clone + Debug,
    Y: Clone + Debug,
{
    /// Constructor.
    pub fn new(data: X, target: Y, columns: Vec<String>) -> Self {
        Dataset {
            data,
            target,
            columns,
        }
    }

    /// Gets the features.
    pub fn data(&self) -> &X {
        &self.data
    }

    /// Gets the targets.
    pub fn target(&self) -> &Y {
        &self.target
    }
}

// From functions for the dataset struct.
impl<X, Y> Dataset<X, Y>
where
    X: for<'de> Deserialize<'de> + Clone + Debug,
    Y: for<'de> Deserialize<'de> + Clone + Debug,
{
    /// Creates a Dataset struct from a CSV file. All
    /// features columns have to be of the same, numeric
    /// type. The taret column can be a categorical value
    /// but it will be automatically one hot encoded.
    ///
    /// Parameters
    /// - filepath: A Path reference.
    /// - target_column: The target column name.
    ///
    pub fn from_csv<P: AsRef<Path>>(file_path: P, target_column: &str) -> MLResult<Self> {

        let file = File::open(file_path.as_ref()).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        // Create the csv Reader from the file (assumes headers are available).
        let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

        // Isolate the header row.
        let headers = rdr.headers().map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        // Make sure target column exists in the file data.
        let target_index = headers.iter().position(|h| h == target_column)
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Target column not found in CSV file: {p}"))?;

    }
}
