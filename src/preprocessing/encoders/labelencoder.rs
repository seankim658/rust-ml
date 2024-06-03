//! # Label Encoder Module
//!
//! This module defines a simple label encoder. The label encoder
//! has two capabilities, it can be used to normalize numerical
//! labels and to label encode non-numerical labels.
//!
//! ## Examples
//! ```
//! use rust_ml::dataset::iris;
//! use rust_ml::linalg::Vector;
//! use rust_ml::preprocessing::encoders::labelencoder::LabelEncoderFitter;
//! use rust_ml::preprocessing::{FitStatus, Preprocessor, PreprocessorFitter};
//! use std::collections::HashMap;
//!
//! let iris_dataset = iris::load();
//! let label_encoder_fitter = LabelEncoderFitter::<String, f64>::default();
//! let mut label_encoder = label_encoder_fitter.fit(iris_dataset.target()).unwrap();
//! let mapped_labels = label_encoder.transform(iris_dataset.target()).unwrap();
//!
//! let mut test_hashmap = HashMap::<String, f64>::new();
//! test_hashmap.insert("Iris-versicolor".to_string(), 1.0);
//! test_hashmap.insert("Iris-virginica".to_string(), 2.0);
//! test_hashmap.insert("Iris-setosa".to_string(), 0.0);
//!
//! assert_eq!(label_encoder.fitter().label_map(), &test_hashmap);
//! assert_eq!(label_encoder.fitter().fit_status(), &FitStatus::Fit);
//! ```

use super::super::{FitStatus, Preprocessor, PreprocessorFitter};
use crate::base::error::{Error, ErrorKind};
use crate::base::MLResult;
use crate::linalg::Vector;

use num::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

/// Struct for the Label Encoder.
#[derive(Clone, Debug)]
pub struct LabelEncoder<K, V>
where
    K: Clone + Debug,
    V: Float + Clone + Debug,
{
    /// The fitter.
    fitter: LabelEncoderFitter<K, V>,
}

impl<K, V> LabelEncoder<K, V>
where
    K: Clone + Debug + Eq + Hash,
    V: Float + Clone + Debug,
{
    /// Returns a reference to the fitter struct.
    pub fn fitter(&self) -> &LabelEncoderFitter<K, V> {
        &self.fitter
    }
}

impl<K, V> Preprocessor<Vector<K>> for LabelEncoder<K, V>
where
    K: Clone + Debug + Eq + Hash,
    V: Float + Clone + Debug,
{
    type O = Vector<V>;

    /// Transforms the Vector based on the fitted Label Encoder hash map.
    ///
    /// #### Parameters:
    /// - input: A reference to the label vector.
    ///
    /// #### Returns:
    /// - MLResult wrapped label encoded label vector.
    ///
    fn transform(&mut self, input: &Vector<K>) -> MLResult<Vector<V>> {
        let mut mapped_vec = Vec::with_capacity(input.size());
        for element in input {
            let mapped_value = self.fitter.label_map.get(&element);
            match mapped_value {
                Some(v) => mapped_vec.push(*v),
                None => {
                    return Err(Error::new(
                        ErrorKind::InvalidState,
                        "Label not found in encoder, invalid fitter state.",
                    ))
                }
            }
        }
        Ok(Vector::new(mapped_vec))
    }
}

/// Struct for the Label Encoder fitter.
#[derive(Clone, Debug)]
pub struct LabelEncoderFitter<K, V>
where
    K: Clone + Debug,
    V: Float + Clone + Debug,
{
    /// The label map.
    label_map: HashMap<K, V>,
    /// Indicates whether the fitter has been fit.
    fit: FitStatus,
}

impl<K, V> LabelEncoderFitter<K, V>
where
    K: Clone + Debug,
    V: Float + Clone + Debug,
{
    /// Returns a reference to the label map value.
    pub fn label_map(&self) -> &HashMap<K, V> {
        &self.label_map
    }
}

impl<K, V> Default for LabelEncoderFitter<K, V>
where
    K: Clone + Debug,
    V: Float + Clone + Debug,
{
    /// Creates an inital, default Label Encoder fitter.
    fn default() -> Self {
        Self {
            label_map: HashMap::default(),
            fit: FitStatus::default(),
        }
    }
}

impl<K, V> PreprocessorFitter<Vector<K>, LabelEncoder<K, V>> for LabelEncoderFitter<K, V>
where
    K: Clone + Debug + Eq + Hash,
    V: Float + Clone + Debug,
{
    /// Fits the label encoder fitter on the given vector.
    ///
    /// #### Parameters:
    /// - input: The categorical label vector to encode.
    ///
    /// #### Returns:
    /// - MLResult wrapped LabelEncoder.
    ///
    fn fit(mut self, input: &Vector<K>) -> MLResult<LabelEncoder<K, V>> {
        self.label_map.clear();
        let mut encoder_value: V = V::zero();

        for value in input {
            if !self.label_map.contains_key(value) {
                self.label_map.insert(value.clone(), encoder_value);
                encoder_value = encoder_value + V::one();
            }
        }
        self.fit = FitStatus::Fit;
        Ok(LabelEncoder { fitter: self })
    }

    /// Get the fit status for the preprocessor fitter.
    fn fit_status(&self) -> &FitStatus {
        &self.fit
    }
}
