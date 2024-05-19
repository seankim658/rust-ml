//! # Pokemon Dataset
//!
//! Loads the Pokemon stat dataset for examples and testing. Because
//! the Pokemon dataset contains categorical features, it must be 
//! read into a `MixedDataset` struct before it can be preprocessed to 
//! coerce it into a `Dataset` struct.
//!
//! ## Dataset
//!
//! The data consists of 800 rows (not including the header row) and
//! some feature columns contain categorical features. The dataset 
//! contains 720 Pokemon from generations 1-6 (due to regional 
//! variants, there are some pokemon with multiple rows).
//!
//! The default target is the `Legendary` column, `Vector<String>`.
//!
//! ## Examples 
//! ```
//! use rust_ml::dataset::pokemon;
//!
//! let pokemon_dataset = pokemon::load();
//!
//! assert_eq!(800, pokemon_dataset.data().len());
//! assert_eq!(12, pokemon_dataset.data()[0].len());
//! ```

use super::MixedDataset;
use crate::linalg::Vector;

/// Loads the default Pokemon dataset.
///
/// ## Panics
///
/// If filepath is incorrect.
///
pub fn load() -> MixedDataset<Vector<String>> {
    let numeric_columns = ["#", "Total", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Generation"];
    MixedDataset::from_csv("./src/dataset/data/pokemon.csv", "Legendary", &numeric_columns).unwrap()
}
