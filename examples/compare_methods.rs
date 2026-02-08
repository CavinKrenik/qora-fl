//! Compare different aggregation methods under attack

use ndarray::array;
use qora_fl::{AggregationMethod, ByzantineAggregator};

fn main() {
    println!("Comparing Aggregation Methods\n");
    println!("Scenario: 7 honest clients (value=1.0), 3 Byzantine (value=100.0)\n");

    // Setup
    let honest = vec![array![[1.0]]; 7];
    let byzantine = vec![array![[100.0]]; 3];

    let mut updates = honest.clone();
    updates.extend(byzantine);

    // Test each method
    let methods: Vec<(&str, AggregationMethod)> = vec![
        ("FedAvg (no defense)", AggregationMethod::FedAvg),
        ("Trimmed Mean (Qora)", AggregationMethod::TrimmedMean),
        ("Median", AggregationMethod::Median),
        ("Krum (Qora, f=3)", AggregationMethod::Krum(3)),
    ];

    for (name, method) in methods {
        let mut agg = ByzantineAggregator::new(method, 0.3);
        let result = agg.aggregate(&updates, None).unwrap();
        let value = result[[0, 0]];

        let status = if (value - 1.0).abs() < 0.5 {
            "ROBUST"
        } else {
            "CORRUPTED"
        };

        println!("{:<25} Result: {:.2}  {}", name, value, status);
    }

    println!("\nTrimmed Mean, Median, and Krum successfully defend against 30% attack!");
}
