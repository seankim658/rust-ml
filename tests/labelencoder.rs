use rust_ml::dataset::iris;
use rust_ml::linalg::Vector;
use rust_ml::preprocessing::encoders::labelencoder::LabelEncoderFitter;
use rust_ml::preprocessing::{FitStatus, Preprocessor, PreprocessorFitter};
use std::collections::HashMap;

#[test]
fn labelencoder_test() {
    let iris_dataset = iris::load();

    let label_encoder_fitter = LabelEncoderFitter::<String, f64>::default();
    let mut label_encoder = label_encoder_fitter.fit(iris_dataset.target()).unwrap();

    let mapped_labels = label_encoder.transform(iris_dataset.target()).unwrap();

    let mut test_hashmap = HashMap::<String, f64>::new();
    test_hashmap.insert("Iris-versicolor".to_string(), 1.0);
    test_hashmap.insert("Iris-virginica".to_string(), 2.0);
    test_hashmap.insert("Iris-setosa".to_string(), 0.0);
    let test_vec = Vector::new(vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    ]);

    assert_eq!(label_encoder.fitter().fit_status(), &FitStatus::Fit);
    assert_eq!(label_encoder.fitter().label_map(), &test_hashmap);
    assert_eq!(mapped_labels.size(), 150);
    assert_eq!(mapped_labels, test_vec);
}
