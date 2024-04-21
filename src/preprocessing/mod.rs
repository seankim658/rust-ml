//! # Preprocessing Module
//!
//! The base module for all the preprocessing functionality in the crate.
//!
//! ## Features
//!
//! Encoders:
//! - Label Encoder
//!
//! Scalers:
//! - MinMax Scaler

use crate::base::MLResult;

pub mod encoders;
pub mod scalers;

/// Trait for a preprocessor.
pub trait Preprocessor<I> {

    /// Associated type for the output type.
    type O;

    /// Function to scale the data. 
    fn transform(&mut self, inputs: &I) -> MLResult<Self::O>;
}

/// Trait for the preprocessor fitters.
pub trait PreprocessorFitter<I, O: Preprocessor<I>> {

    /// Fit the preprocessor to the dataset.
    fn fit(self, inputs: &I) -> MLResult<O>;

    /// Get the fit status for the preprocessor fitter.
    fn fit_status(&self) -> &FitStatus;

}

/// Enum for the fit status.
#[derive(Clone, Debug, PartialEq)]
pub enum FitStatus {
    /// The fitter has not been fit.
    NotFit,
    /// The fitter has been fit.
    Fit
}

impl Default for FitStatus {
    /// Sets the FitStatus enum to the default value of NotFit.
    fn default() -> Self {
       FitStatus::NotFit 
    }
}
