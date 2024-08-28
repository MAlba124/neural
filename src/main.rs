use rand::seq::SliceRandom;
use neural::nn::NeuralNetwork;

fn xor_problem() {
    let training_data = vec![
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![0.0, 0.0], vec![0.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    let mut nn = NeuralNetwork::new(2, 2, 1);

    // Train 10 million times
    for _ in 0..10000000 {
        let (inputs, target) = training_data.choose(&mut rand::thread_rng()).unwrap();
        nn.train(inputs.clone(), target.clone());
    }

    println!("NN says:{:?} (should be true)", nn.feedforward(vec![1.0, 0.0]));
    println!("NN says:{:?} (should be true)", nn.feedforward(vec![0.0, 1.0]));
    println!("NN says:{:?} (should be false)", nn.feedforward(vec![0.0, 0.0]));
    println!("NN says:{:?} (should be false)", nn.feedforward(vec![1.0, 1.0]));

    // (Correct) Example output:
    // NN says:[4.367069] (should be true)
    // NN says:[4.367085] (should be true)
    // NN says:[-4.153192] (should be false)
    // NN says:[-4.364472] (should be false)
    //
    // Sometimes it's wrong:
    // NN says:[4.6742916] (should be true)
    // NN says:[0.035147905] (should be true)
    // NN says:[-4.855547] (should be false)
    // NN says:[0.03604436] (should be false)
}

fn main() {
    xor_problem();
}
