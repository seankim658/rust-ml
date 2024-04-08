//! # Scaler Module
//!

pub mod minmaxscaler;

use crate::base::MLResult;

/// Trait for the encompassing scaler types.
pub trait Scaler<T> {

    /// Function to scale the data.
    fn transform(&mut self, inputs: T) -> MLResult<T>;

}

/// Trait for the scaler fitters.
pub trait ScalerFitter<U, T: Scaler<U>> {

    /// Compute the min and max to be used for later scaling. 
    fn fit(self, inputs: &U) -> MLResult<T>;

    /// Get the status for the fitter, whether it has been
    /// fit or not. 
    fn fit_status(self) -> FitStatus;
}

/// Enum for the fit status.
///
/// Variants:
/// - NotFit: The fitter has not been fit yet.
/// - Fit: The fitter has been fit.
///
#[derive(Debug)]
pub enum FitStatus {
    NotFit,
    Fit
}
