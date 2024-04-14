use rust_ml::dataset::iris;
use rust_ml::linalg::{BaseMatrix, Vector};

#[test]
fn iris_test() {
    let iris_dataset = iris::load();
    assert_eq!(150, iris_dataset.data().rows());
    assert_eq!(5, iris_dataset.data().cols());
    assert_eq!(
        &Vector::new(vec![
            "Id".to_string(),
            "SepalLengthCm".to_string(),
            "SepalWidthCm".to_string(),
            "PetalLengthCm".to_string(),
            "PetalWidthCm".to_string()
        ]),
        iris_dataset.data_columns()
    );
    assert_eq!("Species", iris_dataset.target_column());
}
