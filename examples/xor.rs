use std::io;
use std::io::Write;
use std::time::Instant;

use neural::{matrix::init, nn::NeuralNetwork};
use rand::seq::SliceRandom;

#[inline(always)]
fn secs_to_human(secs: u64) -> String {
    let mut secs = secs;
    let mut s = String::new();
    if secs > 60 * 60 {
        let hours = secs / (60 * 60);
        secs -= hours * 60 * 60;
        s.push_str(&format!("{}h", hours));
    }
    if secs > 60 {
        let mins = secs / 60;
        secs -= mins * 60;
        s.push_str(&format!("{}m", mins));
    }
    s.push_str(&format!("{}s", secs));
    s
}

fn main() {
    init();

    let training_data = vec![
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![0.0, 0.0], vec![0.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    let mut nn = NeuralNetwork::new(2, vec![4, 4], 1);
    nn.set_learning_rate(0.1);

    println!("Before training:");
    println!(
        "  NN says: {:?} (should be ~1.0)",
        nn.feedforward(vec![1.0, 0.0])
    );
    println!(
        "  NN says: {:?} (should be ~1.0)",
        nn.feedforward(vec![0.0, 1.0])
    );
    println!(
        "  NN says: {:?} (should be ~0.0)",
        nn.feedforward(vec![0.0, 0.0])
    );
    println!(
        "  NN says: {:?} (should be ~0.0)",
        nn.feedforward(vec![1.0, 1.0])
    );

    let mut rng = rand::thread_rng();
    let start = Instant::now();
    let mut last = 0;
    const TRAINING_ITERATIONS: usize = 100000;
    print!("Training... Elapsed time: 0s [0/{TRAINING_ITERATIONS} 0.00%]");
    std::io::stdout().flush().unwrap();
    for index in 0..TRAINING_ITERATIONS {
        let (inputs, target) = training_data.choose(&mut rng).unwrap();
        nn.train(&inputs, &target);
        let elapsed_secs = start.elapsed().as_secs();
        if elapsed_secs - last > 0 {
            print!(
                "\r\x1B[0JTraining... Elapsed time: {} [{index}/{TRAINING_ITERATIONS} {:.2}%]",
                secs_to_human(elapsed_secs),
                index as f32 / TRAINING_ITERATIONS as f32 * 100.0
            );
            io::stdout().flush().unwrap();
            last = elapsed_secs;
        }
    }

    println!("\n########################################");
    println!("After training:");
    println!(
        "  NN says: {:?} (should be ~1.0)",
        nn.feedforward(vec![1.0, 0.0])
    );
    println!(
        "  NN says: {:?} (should be ~1.0)",
        nn.feedforward(vec![0.0, 1.0])
    );
    println!(
        "  NN says: {:?} (should be ~0.0)",
        nn.feedforward(vec![0.0, 0.0])
    );
    println!(
        "  NN says: {:?} (should be ~0.0)",
        nn.feedforward(vec![1.0, 1.0])
    );

    // (Correct) Example output:
    // Before training:
    //   NN says: [0.91719854] (should be ~1.0)
    //   NN says: [0.9171817] (should be ~1.0)
    //   NN says: [0.9171671] (should be ~0.0)
    //   NN says: [0.9172116] (should be ~0.0)
    // ########################################
    // After training:
    //   NN says: [0.9826742] (should be ~1.0)
    //   NN says: [0.9826742] (should be ~1.0)
    //   NN says: [0.014388413] (should be ~0.0)
    //   NN says: [0.014388318] (should be ~0.0)
}
