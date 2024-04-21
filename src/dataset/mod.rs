//! # Dataset Module
//!
//! Handles the basic Dataset concept used for the tools in this crate.
//!
//! ## Examples
//!
//! ```
//! use rust_ml::dataset::Dataset;
//! use rust_ml::linalg::{Matrix, BaseMatrix, Vector};
//!
//! let dataset = Dataset::new(
//!     Matrix::new(2, 2, Vector::new(vec![1.0, 2.0, 3.0, 4.0])),
//!     Vector::new(vec![1.0, 2.0, 3.0]),
//!     Vector::new(vec!["feature_1".to_string(), "feature_2".to_string()]),
//!     "label".to_string(),
//! );
//!
//! assert_eq!(2, dataset.data().rows());
//! assert_eq!(2, dataset.data().cols());
//! assert_eq!(
//!     &Vector::new(vec![
//!         "feature_1".to_string(),
//!         "feature_2".to_string()
//!     ]),
//!     dataset.data_columns(),
//! );
//! assert_eq!("label", dataset.target_column());
//! ```

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

/// Module for UCI Iris dataset.
pub mod iris;

/// Struct for a datatset.
#[derive(Clone, Debug)]
pub struct Dataset<X, Y>
where
    X: Clone + Debug,
    Y: Clone + Debug,
{
    /// The feature matrix.
    data: X,
    /// The label vector.
    target: Y,
    /// The data column headers (not including target column header).
    data_columns: Vector<String>,
    /// The target (label) column header.
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

    /// Returns a reference to the features value.
    pub fn data(&self) -> &X {
        &self.data
    }

    /// Returns a reference to the targets value.
    pub fn target(&self) -> &Y {
        &self.target
    }

    /// Returns a reference to the data_columns value.
    pub fn data_columns(&self) -> &Vector<String> {
        &self.data_columns
    }

    /// Returns a reference to the target_column value.
    pub fn target_column(&self) -> &str {
        &self.target_column
    }
}

impl<X, Y> Dataset<Matrix<X>, Vector<Y>>
where
    X: Float + Debug + FromStr,
    Y: Debug + Clone + FromStr,
{
    /// Creates a Dataset struct from a CSV file. All
    /// features columns have to be of the same, numeric
    /// type. The taret column can be a categorical value
    /// but it will be automatically label encoded.
    ///
    /// Parameters:
    /// - filepath: A Path reference.
    /// - target_column: The target column name.
    ///
    /// Returns:
    /// - The loaded dataset in an MLResult instance.
    ///
    pub fn from_csv<P: AsRef<Path>>(file_path: P, target_column: &str) -> MLResult<Self> {
        let file =
            File::open(file_path.as_ref()).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        // Create the csv Reader from the file (assumes headers are available).
        let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

        // Isolate the header row.
        let headers = rdr
            .headers()
            .map_err(|e| Error::new(ErrorKind::InvalidData, e))?
            .clone();

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
                    let record_target = Y::from_str(feature).map_err(|_| {
                        Error::new(
                            ErrorKind::InvalidData,
                            format!("Failed to parse target value {}", feature),
                        )
                    })?;
                    target_values.push(record_target);
                } else {
                    let feature_value = X::from_str(feature).map_err(|_| {
                        Error::new(
                            ErrorKind::InvalidData,
                            format!("Failed to parse value {} in column {}", feature, index),
                        )
                    })?;
                    record_features.push(feature_value);
                }
            }
            data_rows.push(record_features);
        }
        let row_dim = data_rows.len();
        let col_dim = data_rows[0].len();

        // Convert the accumulated data into a matrix for the dataset struct.
        let flattened_data: Vec<X> = data_rows.into_iter().flatten().collect();
        let data = Matrix::new(row_dim, col_dim, flattened_data);

        Ok(Dataset::new(
            data,
            Vector::new(target_values),
            headers
                .iter()
                .filter(|&h| h != target_column)
                .map(|s| s.to_string())
                .collect(),
            String::from(target_column),
        ))
    }
}
