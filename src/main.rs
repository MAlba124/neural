use rand::seq::SliceRandom;
use neural::nn::NeuralNetwork;

fn xor_problem() {
    let training_data = vec![
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![0.0, 0.0], vec![0.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    let mut nn = NeuralNetwork::new(2, 5, 1);

    // Train 10 million times
    for _ in 0..1000000 {
        let (inputs, target) = training_data.choose(&mut rand::thread_rng()).unwrap();
        nn.train(inputs.clone(), target.clone());
    }

    println!("NN says: {:?} (should be true)", nn.feedforward(vec![1.0, 0.0]));
    println!("NN says: {:?} (should be true)", nn.feedforward(vec![0.0, 1.0]));
    println!("NN says: {:?} (should be false)", nn.feedforward(vec![0.0, 0.0]));
    println!("NN says: {:?} (should be false)", nn.feedforward(vec![1.0, 1.0]));

    // (Correct) Example output:
    // NN says: [0.9720577] (should be true)
    // NN says: [0.9817155] (should be true)
    // NN says: [0.024509529] (should be false)
    // NN says: [0.022616664] (should be false)
}

fn main() {
    xor_problem();
}
