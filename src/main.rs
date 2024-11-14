mod parallel_primitives;
mod sort;
mod utils;

use crate::sort::{
    parallel_quicksort, rayon_parallel_quicksort, sequential_quicksort, simple_parallel_quicksort,
};
use crate::utils::Random;
use rayon::prelude::ParallelSliceMut;
use std::time::{Duration, Instant};

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build_global()
        .unwrap();

    bench_sort("sequential", sequential_quicksort);
    bench_sort(
        "parallel (with polylog span and handmade primitives)",
        parallel_quicksort,
    );
    bench_sort("parallel (rayon primitives)", rayon_parallel_quicksort);
    bench_sort(
        "parallel (simple but with big span)",
        simple_parallel_quicksort,
    );
}

const BENCH_ITERATIONS: u32 = 5;

fn bench_sort(name: &str, mut sorter: impl FnMut(&mut [i32])) {
    println!("Benchmarking {}", name);

    let mut random = Random::new(3);
    let total: Duration = (1..=BENCH_ITERATIONS)
        .map(|iteration_num| {
            let mut arr = random.next_vec(100_000_000);
            let mut expected_arr = arr.clone();
            expected_arr.par_sort();

            let start_time = Instant::now();
            sorter(&mut arr);
            let elapsed = start_time.elapsed();

            println!("Iteration {}: {} ms", iteration_num, elapsed.as_millis());

            assert_eq!(expected_arr, arr);

            elapsed
        })
        .sum();
    let avg = total / BENCH_ITERATIONS;

    println!("Avg time: {} ms", avg.as_millis());
    println!()
}
