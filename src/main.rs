use std::fs;
use hdbscan::Hdbscan;

fn main() {

    let contents = fs::read_to_string("test_data.csv").expect("Unable to read file");
    let data = contents.lines().into_iter()
        .map(|s| s.split(",").map(|num| num.parse::<f64>().unwrap()).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    let clusterer = Hdbscan::default(&data);
    let result = clusterer.cluster();
    if let Ok(labels) = result {
        for label in labels {
            println!("{label}");
        }
    }
}