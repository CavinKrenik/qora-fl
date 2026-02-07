use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use qora_fl::{fedavg, median, trimmed_mean};

fn bench_aggregation(c: &mut Criterion) {
    let mut group = c.benchmark_group("aggregation");

    for &n_clients in &[10, 50, 100] {
        for &n_params in &[1_000usize, 100_000, 1_000_000] {
            let updates: Vec<Array2<f32>> = (0..n_clients)
                .map(|i| {
                    Array2::from_shape_fn((1, n_params), |(_, j)| {
                        ((i * n_params + j) as f32).sin()
                    })
                })
                .collect();

            let id = format!("{}c_{}p", n_clients, n_params);

            group.bench_with_input(
                BenchmarkId::new("trimmed_mean", &id),
                &updates,
                |b, updates| b.iter(|| trimmed_mean(updates, 0.2).unwrap()),
            );

            group.bench_with_input(
                BenchmarkId::new("median", &id),
                &updates,
                |b, updates| b.iter(|| median(updates).unwrap()),
            );

            group.bench_with_input(
                BenchmarkId::new("fedavg", &id),
                &updates,
                |b, updates| b.iter(|| fedavg(updates, None).unwrap()),
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_aggregation);
criterion_main!(benches);
