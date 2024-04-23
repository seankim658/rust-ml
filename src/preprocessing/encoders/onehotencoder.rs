//! # One Hot Encoder Module
//!
//! This module defines the one hot encoder. The one hot encoder
//! encodes categorical features as a one-hot numeric array. The
//! encoder will automatically determine the categories from the
//! training data.
//!
//! ## Examples
//! ```
//! ```

use super::super::{FitStatus, Preprocessor, PreprocessorFitter};
use crate::base::error::{Error, ErrorKind};
use crate::base::MLResult;
use crate::linalg::{BaseMatrix, Column, Matrix};
use crate::dataset::Dataset;

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

#[derive(Clone, Debug)]
pub struct OneHotEncoder {
    fitter: OneHotEncoderFitter,
}

/// Struct that defines the parameters that are passed into the OHE fitter.
pub struct OheParams<'a, T, X, Y> 
where
    X: Clone + Debug,
    Y: Clone + Debug,
{
    pub dataset: Dataset<X, Y>,
    /// The column name of the column to be one hot encoded.
    pub column_name: &'a str,
    /// The column to be one hot encoded.
    pub column: &'a Column<'a, T>,
}

#[derive(Clone, Debug)]
pub struct OneHotEncoderFitter {
    /// Holds the categories found in the column.
    category_map: HashMap<String, usize>,
    /// Indicates whether the fitter has been fit.
    fit: FitStatus,
}

impl<'a, T, X, Y> PreprocessorFitter<OheParams<'a, T, X, Y>, OneHotEncoder> for OneHotEncoderFitter
where
    T: Eq + Hash + Clone + Debug,
    X: Clone + Debug,
    Y: Clone + Debug,
{
    /// Fits the one hot encoder on a given dataset column.
    fn fit(mut self, params: &OheParams<'a, T, X, Y>) -> MLResult<OneHotEncoder> {
        let mut category_map = HashMap::new();
        let mut next_idx = 0;

        // Use row_iter to iterate over the elements in the column.
        for value in params.column.row_iter() {
            category_map.entry(value).or_insert_with(|| {
                let index = next_idx;
                next_idx += 1;
                index
            })
        }

        self.category_map = category_map;
        self.fit = FitStatus::Fit;

        Ok(OneHotEncoder { fitter: self })
    }

    fn fit_status(&self) -> &FitStatus {
        &self.fit_status
    }
}
