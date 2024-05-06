use rust_ml::dataset::pokemon;
use rust_ml::linalg::Vector;

#[test]
fn pokemon_test() {
    let pokemon_dataset = pokemon::load();
    assert_eq!(800, pokemon_dataset.data().len());
    assert_eq!(12, pokemon_dataset.data()[0].len());
    assert_eq!(
        &Vector::new(vec![
            "#".to_string(),
            "Name".to_string(),
            "Type 1".to_string(),
            "Type 2".to_string(),
            "Total".to_string(),
            "HP".to_string(),
            "Attack".to_string(),
            "Defense".to_string(),
            "Sp. Atk".to_string(),
            "Sp. Def".to_string(),
            "Speed".to_string(),
            "Generation".to_string()
        ]),
        pokemon_dataset.data_columns()
    );
    assert_eq!("Legendary", pokemon_dataset.target_column());
}
