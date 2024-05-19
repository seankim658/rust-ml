//! # One Hot Encoder Module
//!
//! This module defines the one hot encoder. The one hot encoder
//! encodes all categorical features in a `MixedDataset`. The
//! encoder will automatically determine the categories from the
//! data.
//!
//! ## Examples
//! ```
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

    fn transform(&mut self, inputs: &MixedDataset<Vector<Y>>) -> MLResult<Self::O> {
        let mut transformed_data = Vec::new();
        let mut new_column_names = Vec::new();

        // Add the new one hot encoded categorical column names defined
        // during the fitting process.
        for col_name in inputs.data_columns().iter() {
            if let Some(map) = self.fitter.category_map.get(col_name) {
                for category in map.keys() {
                    new_column_names.push(format!("{}_{}", col_name, category));
                }
            } else {
                new_column_names.push(col_name.clone());
            }
        }

        // Handle the data transformation.
        for row in inputs.data() {
            let mut new_row = Vec::new();
            for (col_index, value) in row.iter().enumerate() {
                let col_name = &inputs.data_columns()[col_index];
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
            Vector::new(inputs.target().clone()),
            Vector::new(new_column_names),
            inputs.target_column().to_string().clone(),
        ))
    }
}

#[derive(Clone, Debug)]
pub struct OneHotEncoderFitter<Y> {
    /// Holds the categories found in the columns to be encoded.
    pub category_map: HashMap<String, HashMap<String, usize>>,
    /// Indicates whether the fitter has been fit.
    fit: FitStatus,
    phantom: std::marker::PhantomData<Y>,
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
    /// Fits the one hot encoder on a given dataset column.
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

    fn fit_status(&self) -> &FitStatus {
        &self.fit
    }
}
