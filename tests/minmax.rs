use rust_ml::dataset::iris;
use rust_ml::preprocessing::scalers::minmaxscaler::MinMaxFitter;
use rust_ml::preprocessing::{FitStatus, Preprocessor, PreprocessorFitter};

#[test]
fn minmaxscaler_test() {
    let iris_dataset = iris::load();

    let minmax_fitter = MinMaxFitter::default();
    let mut minmax_scaler = minmax_fitter.fit(&iris_dataset).unwrap();
    let transformed_dataset = minmax_scaler.transform(&iris_dataset).unwrap();
    
    let min_values = vec![1.0, 4.3, 2.0, 1.0, 0.1];
    let max_values = vec![150.0, 7.9, 4.4, 6.9, 2.5];
    let first_row = &[0.0, 0.2222222222222221, 0.625, 0.06779661016949151, 0.04166666666666667];
    let transformed_first_row = &transformed_dataset.data().data()[0..5];

    assert_eq!(minmax_scaler.fitter().min_values(), &min_values);
    assert_eq!(minmax_scaler.fitter().max_values(), &max_values);
    assert_eq!(minmax_scaler.fitter().fit_status(), &FitStatus::Fit);
    assert_eq!(transformed_first_row, first_row);
}
