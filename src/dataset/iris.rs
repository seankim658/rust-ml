//! # Iris Dataset
//!
//! Loads the infamous UCI Iris dataset for examples and testing.
//!
//! ## Dataset
//!
//! The dataset consists of 5 features as follows: 
//! - Id 
//! - SepalLength 
//! - SepalWidth 
//! - PetalLength 
//! - PetalWidth 
//!
//! Dataset consists of 150 rows (not including the header row) and
//! all feature columns are of type f64, `Matrix<f64>`.
//!
//! The target is the `Species` column, `Vector<usize>`.
//!
//! ## Examples

use crate::linalg::{Matrix, Vector};
use super::Dataset;

pub fn load() -> Dataset<Matrix<f64>, Vec<usize>> {

}
