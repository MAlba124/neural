use neural::matrix::Matrix;

fn main() {
    let mut a = Matrix::new(4096, 4096);
    let mut sum = 0;
    for _ in 0..10 {
        a.randomize();
        let start = std::time::Instant::now();
        a.multiply_scalar(0.5);
        sum += start.elapsed().as_micros()
    }

    println!("[4096, 4096] Average: {}µs", sum / 10);
}
