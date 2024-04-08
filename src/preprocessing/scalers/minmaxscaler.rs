//! # MinMax Scaler
//!
//! This module creates the implementation for a basic MinMax scaler.
//!
//! ## Structure
//!
//! ## Examples
//!

use crate::base::error::{Error, ErrorKind};
use crate::base::MLResult;
use crate::linalg::{Matrix, BaseMatrix, BaseMatrixMut, Vector};
use super::{Scaler, ScalerFitter, FitStatus};
use num::Float;

/// Struct for a MinMax scaler.
///
/// Fields:
/// - fitter: The struct for the MinMax fitter.
/// - scale_factors: Coefficients used to adjust the range of the features in the 
///  dataset. Follows the formula a = (scaled_max - scaled_min) / (max - min) where
///  scaled_max and scaled_min are the target maximum and minimum values for the scaled
///  data and max and min are the actual maximum and minimum values found in the data
///  for the specified feature. 
/// - 
#[derive(Debug)]
pub struct MinMaxScaler<T: Float> {
    fitter: MinMaxFitter<T>,
    scale_factors: Vector<T>,
    const_factors: Vector<T>,
}

/// Implement the Scaler trait for the MinMaxScaler.
impl<T: Float> Scaler<Matrix<T>> for MinMaxScaler<T> {

    fn transform(&mut self, inputs: Matrix<T>) -> MLResult<Matrix<T>> {
        
    }

}

/// Struct for the fitter for the MinMax Scaler.
///
/// Fields:
/// - scaled_min: 
/// - scaled_max:
/// - fit: The fit status of the fitter.
///
#[derive(Debug)]
pub struct MinMaxFitter<T: Float> {
    scaled_min: T,
    scaled_max: T,
    fit: FitStatus
}

/// Implement the Default trait for the MinMaxFitter. 
/// 
/// Defaults:
/// - scaled_min: zero
/// - scaled_max: one
/// - fit: NotFit
/// 
impl<T: Float> Default for MinMaxFitter<T> {
    fn default() -> Self {
        MinMaxFitter {
            scaled_min: T::zero(),
            scaled_max: T::one(),
            fit: FitStatus::NotFit,
        }
    }
}

/// Implement the ScalerFitter trait for the MinMaxFitter.
impl<T: Float> ScalerFitter<Matrix<T>, MinMaxScaler<T>> for MinMaxFitter<T> {

    fn fit(self, inputs: &Matrix<T>) -> MLResult<MinMaxScaler<T>> {
        let num_features = inputs.cols();
    }

    fn fit_status(self) -> FitStatus {
        self.fit
    }

}
