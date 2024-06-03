//! # One Hot Encoder Module
//!
//! This module defines the one hot encoder. The one hot encoder
//! encodes all categorical features in a `MixedDataset`. The
//! encoder will automatically determine the categories from the
//! data.
//!
//! ## Examples
//! ```
//! use rust_ml::dataset::{pokemon, MixedDataset};
//! use rust_ml::linalg::{BaseMatrix, Vector};
//! use rust_ml::preprocessing::encoders::onehotencoder::OneHotEncoderFitter;
//! use rust_ml::preprocessing::{FitStatus, Preprocessor, PreprocessorFitter};
//!
//! let pokemon_dataset: MixedDataset<Vector<String>> = pokemon::load();
//!
//! let ohe_fitter = OneHotEncoderFitter::default();
//! let mut ohe = ohe_fitter.fit(&pokemon_dataset).unwrap();
//!
//! let pokemon_ohe_dataset = ohe.transform(&pokemon_dataset).unwrap();
//! assert_eq!(pokemon_ohe_dataset.data().rows(), 800);
//! assert_eq!(pokemon_ohe_dataset.data().cols(), 46);
//! ```

use super::super::{FitStatus, Preprocessor, PreprocessorFitter};
use crate::base::MLResult;
use crate::dataset::{Dataset, MixedDataValue, MixedDataset};
use crate::linalg::{Matrix, Vector};

use std::collections::HashMap;
use std::fmt::Debug;

/// Struct for the One Hot Encoder.
#[derive(Clone, Debug)]
pub struct OneHotEncoder<Y> {
    /// The fitter.
    fitter: OneHotEncoderFitter<Y>,
}

impl<Y> OneHotEncoder<Y> {
    /// Returns a reference to the fitter struct.
    pub fn fitter(&self) -> &OneHotEncoderFitter<Y> {
        &self.fitter
    }
}

impl<Y> Preprocessor<MixedDataset<Vector<Y>>> for OneHotEncoder<Y>
where
    Y: Clone + Debug,
{
    type O = Dataset<Matrix<f64>, Vector<Y>>;

    /// One hot encodes the categorical columns and returns a new Dataset struct.
    /// 
    /// #### Parameters:
    /// - input: Reference to the MixedDataset to encode.
    ///
    /// #### Returns:
    /// - MLResult wrapped Dataset struct.
    ///
    fn transform(&mut self, input: &MixedDataset<Vector<Y>>) -> MLResult<Self::O> {
        let mut transformed_data = Vec::new();
        let mut new_column_names = Vec::new();

        // Add the new one hot encoded categorical column names defined
        // during the fitting process.
        for col_name in input.data_columns().iter() {
            if let Some(map) = self.fitter.category_map.get(col_name) {
                // Make sure one hot encoded column names are in the right order.
                let mut category_with_indices: Vec<(&String, &usize)> = map.iter().collect();
                category_with_indices.sort_by_key(|&(_, &index)| index);
                for (category, _) in category_with_indices {
                    new_column_names.push(format!("{}_{}", col_name, category));
                }
            } else {
                new_column_names.push(col_name.clone());
            }
        }

        // Handle the data transformation.
        for row in input.data() {
            let mut new_row = Vec::new();
            for (col_index, value) in row.iter().enumerate() {
                let col_name = &input.data_columns()[col_index];
                match value {
                    // For categorical values, look up the encoding map for the
                    // column and initialize the zero-filled vector of the
                    // appropriate length. Then set the corresponding index
                    // to 1 for the one hot encoded binary value.
                    MixedDataValue::Categorical(val) => {
                        if let Some(map) = self.fitter.category_map.get(col_name) {
                            let mut encoded = vec![0.0; map.len()];
                            if let Some(&index) = map.get(val) {
                                encoded[index] = 1.0;
                            }
                            new_row.extend(encoded);
                        }
                    }
                    // For numerical values, dereference the number value and add
                    // it to the row as is.
                    MixedDataValue::Numeric(num) => {
                        new_row.push(*num);
                    }
                }
            }
            transformed_data.push(new_row);
        }

        // Create data Matrix.
        let row_dimension = transformed_data.len();
        let column_dimension = transformed_data[0].len();
        let flattened_data: Vec<f64> = transformed_data.into_iter().flatten().collect();
        let data = Matrix::new(row_dimension, column_dimension, flattened_data);

        Ok(Dataset::new(
            data,
            Vector::new(input.target().clone()),
            Vector::new(new_column_names),
            input.target_column().to_string().clone(),
        ))
    }
}

/// Struct for the one hot encoder fitter.
#[derive(Clone, Debug)]
pub struct OneHotEncoderFitter<Y> {
    /// Holds the categories found in the columns to be encoded.
    category_map: HashMap<String, HashMap<String, usize>>,
    /// Indicates whether the fitter has been fit.
    fit: FitStatus,
    phantom: std::marker::PhantomData<Y>,
}

impl<Y> OneHotEncoderFitter<Y>
where
    Y: Clone + Debug,
{
    /// Returns a reference to the category map.
    pub fn category_map(&self) -> &HashMap<String, HashMap<String, usize>> {
        &self.category_map
    }
}

impl<Y> Default for OneHotEncoderFitter<Y> {
    /// Creates an initial, default One Hot Encoder fitter.
    fn default() -> Self {
        Self {
            category_map: HashMap::default(),
            fit: FitStatus::default(),
            phantom: std::marker::PhantomData,
        }
    }
}

impl<Y> PreprocessorFitter<MixedDataset<Vector<Y>>, OneHotEncoder<Y>> for OneHotEncoderFitter<Y>
where
    Y: Clone + Debug,
{
    /// Fits the one hot encoder on a given dataset's categorical columns.
    ///
    /// #### Parameters:
    /// - input: Reference to the MixedDataset to encode the categorical columns for.
    ///
    /// #### Returns:
    /// - MLResult wrapped OneHotEncoder.
    ///
    fn fit(mut self, input: &MixedDataset<Vector<Y>>) -> MLResult<OneHotEncoder<Y>> {
        self.category_map.clear();
        let mut category_map = HashMap::new();

        for (col_index, col_name) in input.data_columns().iter().enumerate() {
            // Initialize a hashmap for current column that will store
            // mapping from categorical value to their indices.
            let mut map = HashMap::new();

            for row in input.data() {
                // On each row, match on the column value to check if it is categorical.
                if let MixedDataValue::Categorical(value) = &row[col_index] {
                    // If categorical, capture value as a category in the current column map.
                    let index = map.len();
                    map.entry(value.clone()).or_insert_with(|| index);
                }
            }
            // Insert the column map into the fitter category map.
            if !map.is_empty() {
                category_map.insert(col_name.clone(), map);
            }
        }
        self.fit = FitStatus::Fit;
        self.category_map = category_map;
        Ok(OneHotEncoder { fitter: self })
    }

    /// Get the fit status for the preprocessor fitter.
    fn fit_status(&self) -> &FitStatus {
        &self.fit
    }
}
