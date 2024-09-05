use neural::matrix::{init, Matrix};

fn main() {
    init();

    let mut a = Matrix::new(4096, 4096);
    a.randomize();
    let mut sum = 0;
    for _ in 0..10 {
        let start = std::time::Instant::now();
        a.multiply_scalar(0.9);
        sum += start.elapsed().as_micros()
    }

    println!("[4096, 4096] Average: {}µs", sum / 10);
}
