use neural::nn::NeuralNetwork;
use rand::seq::SliceRandom;

fn main() {
    let training_data = vec![
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![0.0, 0.0], vec![0.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    let mut nn = NeuralNetwork::new(2, vec![256, 256], 1);
    nn.set_learning_rate(0.01);

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

    for _ in 0..100000 {
        let (inputs, target) = training_data.choose(&mut rand::thread_rng()).unwrap();
        nn.train(&inputs, &target);
    }

    println!("########################################");
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
