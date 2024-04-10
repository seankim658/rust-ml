//! # Dataset Module
//!
//! This module defines the simple structure for datasets.

use crate::base::error::{Error, ErrorKind};
use crate::base::MLResult;
use crate::linalg::Matrix;
use crate::linalg::Vector;

use csv::ReaderBuilder;
use num::Float;
use std::fmt::Debug;
use std::fs::File;
use std::path::Path;
use std::str::FromStr;

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
    data_columns: Vector<String>,
    target_column: String,
}

/// Constructor and some getters for the Dataset struct.
impl<X, Y> Dataset<X, Y>
where
    X: Clone + Debug,
    Y: Clone + Debug,
{
    /// Constructor.
    pub fn new(data: X, target: Y, data_columns: Vector<String>, target_column: String) -> Self {
        Dataset {
            data,
            target,
            data_columns,
            target_column,
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
impl<X> Dataset<Matrix<X>, Vector<X>>
where
    X: Float + Debug + FromStr,
{
    /// Creates a Dataset struct from a CSV file. All
    /// features columns have to be of the same, numeric
    /// type. The taret column can be a categorical value
    /// but it will be automatically label encoded.
    ///
    /// Parameters
    /// - filepath: A Path reference.
    /// - target_column: The target column name.
    ///
    pub fn from_csv<P: AsRef<Path>>(file_path: P, target_column: &str) -> MLResult<Self> {
        let file =
            File::open(file_path.as_ref()).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        // Create the csv Reader from the file (assumes headers are available).
        let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

        // Isolate the header row.
        let headers = rdr
            .headers()
            .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;

        // Make sure target column exists in the file data.
        let target_index = headers
            .iter()
            .position(|h| h == target_column)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidData,
                    format!("Target column {} not found in CSV file.", target_column),
                )
            })?;

        let mut data_rows = Vec::new();
        let mut target_values = Vec::new();
        // Build the data rows 2d vector and the label vector.
        for record_result in rdr.records() {
            let record = record_result.map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
            let mut record_features = Vec::new();
            for (index, feature) in record.iter().enumerate() {
                if index == target_index {
                    let record_target = X::from_str(feature).map_err(|_| {
                        Error::new(
                            ErrorKind::InvalidData,
                            format!("Failed to parse target value {}", feature),
                        )
                    })?;
                    target_values.push(record_target);
                } else {
                    let feature_value = X::from_str(feature).map_err(|_| {
                        Error::new(ErrorKind::InvalidData, format!("Failed to parse value {} in column {}", feature, index))
                    })?;
                    record_features.push(feature_value);
                }
            }
            data_rows.push(record_features);
        }

        // Convert the accumulated data into a matrix for the dataset struct.
        let flattened_data: Vec<X> = data_rows.into_iter().flatten().collect();
        let data = Matrix::new(
            data_rows.len(),
            data_rows[0].len(),
            flattened_data
        );

        Ok(Dataset {
            data,
            target: Vector::new(target_values),
            data_columns: headers.iter().map(|s| s.to_string()).collect(),
            target_column: String::from(target_column),
        })
    }
}
