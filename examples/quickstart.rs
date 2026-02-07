//! Quickstart example showing Byzantine tolerance

use ndarray::array;
use qora_fl::{AggregationMethod, ByzantineAggregator};

fn main() {
    println!("Qora-FL Quickstart Demo\n");

    // Create aggregator with 30% trim fraction (handles 30% attackers)
    let mut agg = ByzantineAggregator::new(AggregationMethod::TrimmedMean, 0.3);

    println!("Simulating 10 clients (7 honest, 3 Byzantine)...\n");

    // 7 honest clients with similar updates
    let mut updates = vec![array![[1.0, 2.0, 3.0]]; 7];

    // 3 Byzantine attackers with extreme values
    updates.push(array![[100.0, 200.0, 300.0]]);
    updates.push(array![[100.0, 200.0, 300.0]]);
    updates.push(array![[100.0, 200.0, 300.0]]);

    // Aggregate with Byzantine tolerance
    let result = agg.aggregate(&updates, None).unwrap();

    println!("Aggregation complete!");
    println!("   Result: {:?}", result);
    println!("   Expected (close to honest mean): [1.0, 2.0, 3.0]");
    println!("\nByzantine clients successfully ignored!");
}
