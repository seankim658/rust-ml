//! # Iris Dataset
//!
//! Loads the infamous UCI Iris dataset for examples and testing.
//!
//! ## Dataset
//!
//! The dataset consists of 5 features as follows (which are all numeric): 
//! - Id 
//! - SepalLength 
//! - SepalWidth 
//! - PetalLength 
//! - PetalWidth 
//!
//! Dataset consists of 150 rows (not including the header row) and
//! all feature columns are of type f64, `Matrix<f64>`.
//!
//! The target is the `Species` column, `Vector<String>`.
//!
//! ## Examples
//! 
//! ```
//! use rust_ml::dataset::iris;
//! use rust_ml::linalg::BaseMatrix;
//!
//! let iris_dataset = iris::load();
//! 
//! assert_eq!(150, iris_dataset.data().rows());
//! assert_eq!(5, iris_dataset.data().cols());
//! ```

use crate::linalg::{Matrix, Vector};
use super::Dataset;

/// Loads the default Iris dataset.
///
/// ## Panics
///
/// If filepath is incorrect.
/// 
pub fn load() -> Dataset<Matrix<f64>, Vector<String>> {
    Dataset::from_csv("./src/dataset/data/iris.csv", "Species").unwrap()
}
