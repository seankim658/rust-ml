//! # Dataset Module
//!
//! Handles the basic Dataset concepts used for the tools in this crate.
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
/// Module for Pokemon stats dataset.
pub mod pokemon;

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

    /// Returns a reference to the data_columns vector.
    pub fn data_columns(&self) -> &Vector<String> {
        &self.data_columns
    }

    /// Returns a reference to the target_column name.
    pub fn target_column(&self) -> &str {
        &self.target_column
    }
}

impl<X, Y> Dataset<Matrix<X>, Vector<Y>>
where
    X: Float + Debug + FromStr,
    Y: Debug + Clone + FromStr,
{
    /// Creates a Dataset struct from a CSV file. All features columns have to be of
    /// the same, numeric type. The taret column can be a categorical value.
    ///
    /// #### Parameters:
    /// - filepath: A Path reference.
    /// - target_column: The target column name.
    ///
    /// #### Returns:
    /// - The loaded dataset in an MLResult instance.
    ///
    pub fn from_csv<P: AsRef<Path>>(file_path: P, target_column: &str) -> MLResult<Self> {
        let file = File::open(file_path).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        // Create the csv reader from the file (assumes headers are available).
        let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

        let (headers, target_index) = process_headers(&mut rdr, target_column)?;

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
            Vector::new(
                headers
                    .iter()
                    .filter(|&h| h != target_column)
                    .map(|s| s.to_string())
                    .collect::<Vec<String>>(),
            ),
            String::from(target_column),
        ))
    }
}

/// Can represent a numeric or categorical data value.
#[derive(Debug, Clone, PartialEq)]
pub enum MixedDataValue {
    /// Numeric data values are f64s.
    Numeric(f64),
    /// Categorical data values are Strings.
    Categorical(String),
}

/// Struct for a mixed value dataset. This struct can
/// support the loading of datasets with mixed data
/// values. If your dataset contains categorical values
/// it must be read in as a MixedDataset before using
/// an encoder to coerce it into a standard Datset.
#[derive(Debug, Clone)]
pub struct MixedDataset<Y>
where
    Y: Clone + Debug,
{
    /// The 2 dimensional feature vector.
    data: Vec<Vec<MixedDataValue>>,
    /// The label vector.
    target: Y,
    /// The data column headers (not including target column header).
    data_columns: Vector<String>,
    /// The target (label) column header.
    target_column: String,
}

/// Constructor and some getters for the MixedDataset struct.
impl<Y> MixedDataset<Y>
where
    Y: Clone + Debug,
{
    /// Constructor.
    pub fn new(
        data: Vec<Vec<MixedDataValue>>,
        target: Y,
        data_columns: Vector<String>,
        target_column: String,
    ) -> Self {
        MixedDataset {
            data,
            target,
            data_columns,
            target_column,
        }
    }

    /// Returns a reference to the 2D feature vector.
    pub fn data(&self) -> &Vec<Vec<MixedDataValue>> {
        &self.data
    }

    /// Returns a reference to the target vector.
    pub fn target(&self) -> &Y {
        &self.target
    }

    /// Returns a reference to the data_columns vector.
    pub fn data_columns(&self) -> &Vector<String> {
        &self.data_columns
    }

    /// Returns a reference to the target_column name.
    pub fn target_column(&self) -> &str {
        &self.target_column
    }
}

impl<Y> MixedDataset<Vector<Y>>
where
    Y: Debug + Clone + FromStr,
{
    /// Creates a MixedDataset struct from a CSV file. Unlike the `from_csv` method on the
    /// Dataset struct, this method supports data with categorical features, but you have
    /// to specify the numeric columns.
    ///
    /// #### Parameters:
    /// - filepath: A Path reference.
    /// - target_column: The target column name.
    /// - numeric_columns: The columns that contain numeric values, other columns will be assumed categorical.
    ///
    /// #### Returns:
    /// - The loaded dataset in an MLResult instance.
    ///
    pub fn from_csv<P: AsRef<Path>>(
        file_path: P,
        target_column: &str,
        numeric_columns: &[&str],
    ) -> MLResult<Self> {
        let file = File::open(file_path).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        // Create the csv reader from the file (assumes headers are available).
        let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

        let (headers, target_index) = process_headers(&mut rdr, target_column)?;

        // Collect indices for columns specified as numeric.
        let numeric_idxs: Vec<usize> = headers
            .iter()
            .enumerate()
            .filter_map(|(idx, name)| {
                if numeric_columns.contains(&name) {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();

        let mut data_rows = Vec::new();
        let mut target_values = Vec::new();
        // Build the data rows 2d vector and the label vector.
        for record_result in rdr.records() {
            let record = record_result.map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
            let mut record_features = Vec::new();
            for (index, feature) in record.iter().enumerate() {
                let data_value = if numeric_idxs.contains(&index) {
                    MixedDataValue::Numeric(feature.parse::<f64>().map_err(|e| {
                        Error::new(
                            ErrorKind::InvalidData,
                            format!(
                                "Failed to parse value {} in column {}.\n{}",
                                feature, index, e
                            ),
                        )
                    })?)
                } else {
                    MixedDataValue::Categorical(feature.to_string())
                };

                if index == target_index {
                    let record_target = Y::from_str(feature).map_err(|_| {
                        Error::new(
                            ErrorKind::InvalidData,
                            format!("Failed to parse target value {}", feature),
                        )
                    })?;
                    target_values.push(record_target);
                } else {
                    record_features.push(data_value);
                }
            }
            data_rows.push(record_features);
        }
        Ok(MixedDataset::new(
            data_rows,
            Vector::new(target_values),
            Vector::new(
                headers
                    .iter()
                    .filter(|&h| h != target_column)
                    .map(|s| s.to_string())
                    .collect::<Vec<String>>(),
            ),
            String::from(target_column),
        ))
    }
}

/// Helper function that processes the headers in the CSV file and makes sure
/// the user passed target column exists.
///
/// #### Parameters:
/// - rdr: The CSV Reader.
/// - target_column: The target column name.
///
/// #### Returns:
/// - A Result wrapped tuple containing the isolated header row and the target column
/// index or an Error.
///
fn process_headers<R: std::io::Read>(
    rdr: &mut csv::Reader<R>,
    target_column: &str,
) -> Result<(csv::StringRecord, usize), Error> {
    // Isolate header row.
    let headers = rdr
        .headers()
        .map_err(|e| Error::new(ErrorKind::InvalidData, e))?
        .clone();

    // Make sure the target column exists in the file column headers.
    let target_index = headers
        .iter()
        .position(|h| h == target_column)
        .ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidData,
                format!("Target column {} not found in CSV file.", target_column),
            )
        })?;

    Ok((headers, target_index))
}
