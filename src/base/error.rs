//! # Error
//!
//! This module implements the basic error types for the library.
//!
//! ## Structure
//!
//! ## Examples

use std::error;
use std::fmt;

/// Enum for our defined error kinds.
#[derive(Debug)]
pub enum ErrorKind {
    /// Parameters passed or used are invalid. 
    InvalidParameters,
    /// Error with the input data.
    InvalidData,
    /// Model is in an invalid state.
    InvalidState,
    /// Trying to perform an invalid action on an unfitted model.
    UntrainedModel,
    /// Linear algebra module error.
    LinAlgError
}

/// Struct for an error.
///
/// Fields:
/// - kind: The ErrorKind enum value for more context.
/// - error: Thread safe wrapper for Rust errors.
///
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    error: Box<dyn error::Error + Send + Sync>,
}

/// Creates the Error methods.
impl Error {

    /// Constructor.
    ///
    /// Parameters
    /// - kind: The ErrorKind enum.
    /// - error: Generic that implements Into Box.
    ///
    /// Returns
    /// - New error struct.
    ///
    pub fn new<E>(kind: ErrorKind, error: E) -> Error
        where E: Into<Box<dyn error::Error + Send + Sync>>
    {
        Error {
            kind,
            error: error.into()
        }
    }

    /// Method to get the error kind variant.
    ///
    /// Parameters
    /// - &self: Reference to self (of type Error).
    ///
    /// Returns
    /// - The ErrorKind variant.
    ///
    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }
}

/// Implements the Display trait for printing and formatting, just passes on
/// the formatting to the wrapped error.
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.error.fmt(f)
    }
}
