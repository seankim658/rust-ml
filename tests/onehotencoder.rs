use rust_ml::dataset::{pokemon, MixedDataset};
use rust_ml::linalg::{BaseMatrix, Vector};
use rust_ml::preprocessing::encoders::onehotencoder::OneHotEncoderFitter;
use rust_ml::preprocessing::{FitStatus, Preprocessor, PreprocessorFitter};

#[test]
fn onehotencoder_test() {
    let pokemon_dataset: MixedDataset<Vector<String>> = pokemon::load();

    let ohe_fitter = OneHotEncoderFitter::default();
    let mut ohe = ohe_fitter.fit(&pokemon_dataset).unwrap();

    let pokemon_ohe_dataset = ohe.transform(&pokemon_dataset).unwrap();
    assert_eq!(ohe.fitter().fit_status(), &FitStatus::Fit);
    assert_eq!(pokemon_ohe_dataset.data().rows(), 800);
    assert_eq!(pokemon_ohe_dataset.data().cols(), 46);
}
