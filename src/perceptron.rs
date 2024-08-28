use rand::Rng;

fn f(x: f32) -> f32 {
    0.3 * x + 0.2
}

struct Point {
    x: f32,
    y: f32,
    bias: f32,
    label: i32,
}

impl Point {
    pub fn new() -> Self {
        let x = rand::thread_rng().gen_range(-1.0..=1.0);
        let y = rand::thread_rng().gen_range(-1.0..=1.0);
        let line_y = f(x);
        Self {
            x, y,
            label: if y > line_y { 1 } else { -1 },
            bias: 1.0,
        }
    }
}

// Activation function
fn sign(y: f32) -> i32 {
    if y < 0.0 {
        return -1;
    }
    return 1;
}

#[derive(Debug)]
struct Perceptron {
    weights: [f32; 3],
}

impl Perceptron {
    const LEARNING_RATE: f32 = 0.000001;

    pub fn new() -> Self {
        Self {
            weights: [
                rand::thread_rng().gen_range(-1.0..=1.0),
                rand::thread_rng().gen_range(-1.0..=1.0),
                rand::thread_rng().gen_range(-1.0..=1.0)
            ],
        }
    }

    pub fn guess(&self, inputs: &[f32]) -> i32 {
        let mut sum: f32 = 0.0;
        for i in 0..self.weights.len() {
            sum += inputs[i] * self.weights[i];
        }
        sign(sum)
    }

    pub fn train(&mut self, inputs: Vec<f32>, target: i32) -> bool {
        let guess = self.guess(&inputs);
        let error = (target - guess) as f32;

        if error == 0.0 {
            return true;
        }

        // Tune the weights
        for i in 0..self.weights.len() {
            self.weights[i] += error * inputs[i] * Self::LEARNING_RATE;
        }

        false
    }
}

pub fn perceptron() {
    let mut p = Perceptron::new();
    println!("Before training: {p:?}");

    let inputs = vec![-1.0, 0.5, 1.0];
    println!("Guess: {}", p.guess(&inputs));

    let upper = 10000000;
    let mut correct = 0;
    for _ in 0..upper {
        let point = Point::new();
        if p.train(vec![point.x, point.y, point.bias], point.label) {
            correct += 1;
        }
    }

    println!("{correct}/{upper} correct");

    println!("After training: {p:?}");
    println!("Guess: {}", p.guess(&inputs));
}
