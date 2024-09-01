use std::time::Instant;

use neural::matrix::Matrix;

fn main() {
    let sizes = vec![8, 64, 128, 256, 512, 1024];//, 2048];

    for n in sizes {
        let mut a = Matrix::new(n, n);
        let mut b = Matrix::new(n, n);
        let mut c = Matrix::new(n, n);
        a.randomize();
        b.randomize();

        let start = Instant::now();
        a.product_into(&b, &mut c);
        let end = start.elapsed().as_secs_f64();

        println!(
            "Size: {n}x{n} Perf: {:.2} GFLOP/S",
            (2 * n.pow(3)) as f64 / end / 1000000000.0
        );
    }
}

// 21:46 30.08.2024:
//   Size: 8x8 Perf: 0.84 GFLOP/S
//   Size: 64x64 Perf: 1.64 GFLOP/S
//   Size: 128x128 Perf: 1.36 GFLOP/S
//   Size: 256x256 Perf: 1.30 GFLOP/S
//   Size: 512x512 Perf: 1.27 GFLOP/S
//   Size: 1024x1024 Perf: 1.21 GFLOP/S
//   Size: 2048x2048 Perf: 0.58 GFLOP/S

// 22:48 30.08.2024:
//   Size: 8x8 Perf: 0.62 GFLOP/S
//   Size: 64x64 Perf: 3.44 GFLOP/S
//   Size: 128x128 Perf: 2.51 GFLOP/S
//   Size: 256x256 Perf: 2.22 GFLOP/S
//   Size: 512x512 Perf: 2.28 GFLOP/S
//   Size: 1024x1024 Perf: 2.30 GFLOP/S
//   Size: 2048x2048 Perf: 2.02 GFLOP/S
