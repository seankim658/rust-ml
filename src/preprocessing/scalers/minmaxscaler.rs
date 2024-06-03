//! # MinMax Scaler
//!
//! This module creates the implementation for a basic MinMax scaler.
//!
//! ## Examples
//! ```
//! use rust_ml::dataset::iris;
//! use rust_ml::preprocessing::scalers::minmaxscaler::MinMaxFitter;
//! use rust_ml::preprocessing::{FitStatus, Preprocessor, PreprocessorFitter};
//!
//! let iris_dataset = iris::load();
//!
//! let minmax_fitter = MinMaxFitter::default();
//! let mut minmax_scaler = minmax_fitter.fit(&iris_dataset).unwrap();
//! let transformed_dataset = minmax_scaler.transform(&iris_dataset).unwrap();
//!
//! assert_eq!(minmax_scaler.fitter().fit_status(), &FitStatus::Fit);
//! ```

use crate::base::error::{Error, ErrorKind};
use crate::base::MLResult;
use crate::dataset::Dataset;
use crate::linalg::{BaseMatrix, Matrix, Vector};
use crate::preprocessing::{FitStatus, Preprocessor, PreprocessorFitter};
use std::fmt::Debug;

/// Struct for a MinMax scaler.
#[derive(Debug)]
pub struct MinMaxScaler<Y> {
    /// The struct for the MinMax fitter.
    fitter: MinMaxFitter<Y>,
}

impl<Y> MinMaxScaler<Y> {
    /// Returns a reference to the fitter.
    pub fn fitter(&self) -> &MinMaxFitter<Y> {
        &self.fitter
    }
}

impl<Y> Preprocessor<Dataset<Matrix<f64>, Vector<Y>>> for MinMaxScaler<Y>
where
    Y: Clone + Debug,
{
    type O = Dataset<Matrix<f64>, Vector<Y>>;

    /// Scales the features into the scaled min and max range and returns
    /// a new Dataset struct.
    ///
    /// #### Parameters:
    /// - input: Reference to the Dataset to scale.
    ///
    /// #### Returns:
    /// - MLResults wrapped scaled Dataset.
    ///
    fn transform(&mut self, input: &Dataset<Matrix<f64>, Vector<Y>>) -> MLResult<Self::O> {
        let fitter = self.fitter();
        let num_features = fitter.num_features();
        if num_features != &input.data_columns().size() {
            return Err(Error::new(
                ErrorKind::InvalidState,
                format!(
                    "Fitter's number of features ({}) does not match dataset's number of features ({})",
                    num_features,
                    input.data_columns().size()
                ),
            ));
        }
        let num_rows = input.data().rows();
        let mut scaled_data = Vec::with_capacity(input.data().data().len());

        for row in input.data().row_iter() {
            for (idx, &value) in row.iter().enumerate() {
                let scaled_value =
                    value * fitter.scale_factors()[idx] + fitter.constant_factors()[idx];
                scaled_data.push(scaled_value);
            }
        }

        let scaled_matrix = Matrix::new(num_rows, *num_features, scaled_data);
        Ok(Dataset::new(
            scaled_matrix,
            input.target().clone(),
            input.data_columns().clone(),
            input.target_column().to_string(),
        ))
    }
}

/// Struct for the fitter for the MinMax Scaler.
#[derive(Debug)]
pub struct MinMaxFitter<Y> {
    /// The number of features in the dataset.
    num_featues: usize,
    /// The range minimum to scale by.
    scaled_min: f64,
    /// The range maximum to scale by.
    scaled_max: f64,
    /// The minimum value for each feature.
    min_values: Vec<f64>,
    /// The maximum value for each feature.
    max_values: Vec<f64>,
    /// Scale factor for each feature. Used to adjust the range of
    /// the original data to the scaled range. Calculated with the
    /// formula a = (scaled_max - scaled_min) / (max - min) where
    /// scaled_max and scaled_min are the target maximum and minimum
    /// values for the scaled data and max and min are the actual
    /// maximum and minimum values found in the data for the specified
    /// feature.
    scale_factors: Vec<f64>,
    /// The constant factor used to shift the scaled data to start
    /// from the scaled minimum value. Calculated with the formula
    /// b = scaled_min - min * scale_factor.
    constant_factors: Vec<f64>,
    /// Indicates whether the fitter has been fit.
    fit: FitStatus,
    phantom: std::marker::PhantomData<Y>,
}

impl<Y> MinMaxFitter<Y> {
    /// Create a new instance of of the MinMaxFitter with
    /// a custom scaled min and scaled max value.
    ///
    /// #### Parameters
    /// - min: The scaled minimum.
    /// - max: The scaled maximum.
    ///
    pub fn new(min: f64, max: f64) -> Self {
        MinMaxFitter {
            num_featues: 0,
            scaled_min: min,
            scaled_max: max,
            min_values: Vec::new(),
            max_values: Vec::new(),
            scale_factors: Vec::new(),
            constant_factors: Vec::new(),
            fit: FitStatus::NotFit,
            phantom: std::marker::PhantomData,
        }
    }

    /// Returns the number of features in the dataset.
    pub fn num_features(&self) -> &usize {
        &self.num_featues
    }

    /// Returns a tuple of references to the scaled_min and scaled_max.
    pub fn min_max(&self) -> (&f64, &f64) {
        (&self.scaled_min, &self.scaled_max)
    }

    /// Returns a reference to the min_values vector.
    pub fn min_values(&self) -> &Vec<f64> {
        &self.min_values
    }

    /// Returns a reference to the max_values vector.
    pub fn max_values(&self) -> &Vec<f64> {
        &self.max_values
    }

    /// Returns a reference to the scale_factors vector.
    pub fn scale_factors(&self) -> &Vec<f64> {
        &self.scale_factors
    }

    /// Returns a reference to the constant_factors vector.
    pub fn constant_factors(&self) -> &Vec<f64> {
        &self.constant_factors
    }
}

impl<Y> Default for MinMaxFitter<Y> {
    /// Implement the Default trait for the MinMaxFitter.
    fn default() -> Self {
        MinMaxFitter {
            num_featues: usize::default(),
            scaled_min: 0.0,
            scaled_max: 1.0,
            min_values: Vec::default(),
            max_values: Vec::default(),
            scale_factors: Vec::default(),
            constant_factors: Vec::default(),
            fit: FitStatus::NotFit,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<Y> PreprocessorFitter<Dataset<Matrix<f64>, Vector<Y>>, MinMaxScaler<Y>> for MinMaxFitter<Y>
where
    Y: Clone + Debug,
{
    /// Fits the min max scaler on a given dataset.
    ///
    /// #### Parameters:
    /// - input: Reference to the Dataset to fit on.
    ///
    /// #### Returns:
    /// - MLResult wrapped MinMaxScaler.
    ///
    fn fit(mut self, input: &Dataset<Matrix<f64>, Vector<Y>>) -> MLResult<MinMaxScaler<Y>> {
        let num_features = input.data_columns().size();
        self.num_featues = num_features;
        let mut min_values = vec![f64::MAX; num_features];
        let mut max_values = vec![f64::MIN; num_features];
        let mut scale_factors = vec![0.0; num_features];
        let mut constant_factors = vec![0.0; num_features];

        for row in input.data().row_iter() {
            for (idx, &value) in row.iter().enumerate() {
                if value < min_values[idx] {
                    min_values[idx] = value;
                }
                if value > max_values[idx] {
                    max_values[idx] = value;
                }
            }
        }

        self.fit = FitStatus::Fit;
        self.min_values = min_values.clone();
        self.max_values = max_values.clone();

        for i in 0..num_features {
            let scaled_difference = self.scaled_max - self.scaled_min;
            let scale_factor = (scaled_difference) / (max_values[i] - min_values[i]);
            scale_factors[i] = scale_factor;
            let constant_factor = self.scaled_min - (min_values[i] * scale_factor);
            constant_factors[i] = constant_factor;
        }

        self.scale_factors = scale_factors.clone();
        self.constant_factors = constant_factors.clone();

        Ok(MinMaxScaler { fitter: self })
    }

    /// Get the fit status for the preprocessor fitter.
    fn fit_status(&self) -> &FitStatus {
        &self.fit
    }
}
