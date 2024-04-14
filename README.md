# Rust ML 

*Work in progress.*

## Background 

Learning Rust and brushing up on basic machine learning concepts by building a simple machine learning library, (very) loosely based on the [scikit-learn](https://github.com/scikit-learn/scikit-learn) Python library. This project is still very much so in the **very** early stages and no real features have been fully implemented yet. Still working on setting up the foundation of the project.

## Dependencies

- [rulinalg](https://github.com/AtheMathmo/rulinalg) is used for some basic linear algebra concepts.
- [num](https://github.com/rust-num/num) is used for the `Float` trait.
- [csv](https://github.com/BurntSushi/rust-csv) is used for CSV handling.
- [serde](https://github.com/serde-rs/serde) is used for deserializing Rust data structures.

## Current Progress

Really nothing has been completed fully yet except for some of the foundational concepts used throughout the library. Documentation is being written as things are developed. 

## In Progress

Currently working on the implementation of the `Dataset` module. The module lays out the foundation for the Dataset type and some basic functionality, such as reading from a CSV file. Using the included Iris dataset, going to implement some preprocessing features such as a label encoder. 
