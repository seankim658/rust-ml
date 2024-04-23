//! # Rust_ML Crate
//!
//! A simple Rust machine learning library.
//!
//! ## Features
//! 
//! Datasets:
//! - Iris dataset.
//!
//! Encoders:
//! - Label encoder.
//!
//! Scalers:
//! - MinMax scaler.
//!
//! ## Examples
//!
//!

/// Re-exports of commonnly used rulinalg (`https://github.com/AtheMathmo/rulinalg`) linear
/// algebra tools and data types.
///
/// Re-exports: 
/// - Axes: Enum for column or row indication.
/// - Matrix: Struct for the matrix. 
/// - MatrixSlice: Struct to provide a slice into a matrix.
/// - MatrixSliceMut: Struct to provide a mutable slice into a matrix.
/// - Column: Struct 
/// - BaseMatrix: Trait for immutable matrix structs.
/// - BaseMatrixMut: Trait for mutable matrix structs.
/// - Vector: Struct for vectors.
/// - ColumnMut: Struct for a mutable column of a matrix.
pub mod linalg {
    pub use rulinalg::matrix::{Axes, Matrix, MatrixSlice, MatrixSliceMut, Column, BaseMatrix, BaseMatrixMut};
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
pub mod preprocessing; 
