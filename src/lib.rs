//! # Rust-ML
//!
//! A simple Rust machine learning library.
//!
//! ## Structure
//!
//! Currently supported machine learning techniques:
//!
//! - Linear Regression
//!
//! ## Examples
//!
//!

/// Re-exports of commonnly used rulinalg (`https://github.com/AtheMathmo/rulinalg`) linear
/// algebra tools.
///
/// rulinalg::matrix : 
/// - Axes: Enum for column or row indication.
/// - Matrix: Struct for the matrix. 
/// - MatrixSlice: Struct to provide a slice into a matrix.
/// - MatrixSliceMut: Struct to provide a mutable slice into a matrix.
/// - BaseMatrix: Trait for immutable matrix structs.
/// - BaseMatrixMut: Trait for mutable matrix structs.
///
/// rulinalg::vector tools:
/// - Vector: Struct for vectors.
pub mod linalg {
    pub use rulinalg::matrix::{Axes, Matrix, MatrixSlice, MatrixSliceMut, BaseMatrix, BaseMatrixMut};
    pub use rulinalg::vector::Vector;
}

/// Module for base items used throughout the crate.
pub mod base {

    /// Module to define errors used in this crate.
    pub mod error;

    /// Type alias for the use of the Result type in this crate.
    pub type MLResult<T> = Result<T, error::Error>;
}

/// Module for the basic dataset structure.
pub mod dataset;

/// Module for some data preprocessing functionality.
///
/// Encoders:
/// - Label encoder.
///
/// Scalers:
///     - MinMax scaler.
pub mod preprocessing {
    pub mod encoders;
    pub mod scalers;
}

