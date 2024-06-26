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
//! The dataset consists of 2 categorical features and 9 numerical features:
//! - Categorical features:
//!   - Type 1 (primary pokemon typing)
//!   - Type 2 (secondary pokemon typing)
//! - Numeric features:
//!   - \# (pokedex number)
//!   - Total (base stat total)
//!   - HP (health points stat)
//!   - Attack (physical attack stat)
//!   - Defense (physical defense stat)
//!   - Sp. Atk (special attack stat)
//!   - Sp. Def (special defense stat)
//!   - Speed (speed stat)
//!   - Generation (generation the pokemon was introduced)
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
//! assert_eq!(11, pokemon_dataset.data()[0].len());
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
