use crate::matrix::Matrix;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// Assumes y comes from sigmoid()
fn dsigmoid(y: f32) -> f32 {
    y * (1.0 - y)
}

#[derive(Debug)]
pub struct Layer {
    pub weights: Matrix,
    pub bias: Matrix,
}

#[derive(Debug)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
    learning_rate: f32,
}

impl NeuralNetwork {
    pub fn new(n_input: usize, hidden: Vec<usize>, n_output: usize) -> Self {
        assert!(hidden.len() > 0 && n_output > 0);

        let mut layers = Vec::new();

        let mut layer_arch = vec![n_input];
        layer_arch.extend(hidden);
        layer_arch.push(n_output);

        let mut input_weights_count = n_input;
        for neuron_count in layer_arch {
            let mut weights = Matrix::new(neuron_count, input_weights_count);
            weights.randomize();
            let mut bias = Matrix::new(neuron_count, 1);
            bias.randomize();

            layers.push(
                Layer {
                    weights, bias,
                }
            );
            input_weights_count = neuron_count;
        }

        Self {
            learning_rate: 0.003,
            layers,
        }
    }

    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    pub fn feedforward(&self, input: Vec<f32>) -> Vec<f32> {
        let mut inputs = Matrix::from_vec(input);

        for layer in &self.layers {
            let mut new = layer.weights.product(&inputs);
            new.add_matrix(&layer.bias);
            new.map(&sigmoid);
            inputs = new;
        }

        inputs.to_vec()
    }

    pub fn train(&mut self, inputs: Vec<f32>, targets: Vec<f32>) {
        let mut inputs = Matrix::from_vec(inputs);
        let orig_inputs = inputs.clone();

        let mut results = Vec::new();
        for layer in &self.layers {
            let mut new = layer.weights.product(&inputs);
            new.add_matrix(&layer.bias);
            new.map(&sigmoid);
            results.push(new.clone());
            inputs = new;
        }

        let targets = Matrix::from_vec(targets);
        let outputs = results.last().unwrap();
        let mut errors = targets.subtract_matrix(&outputs);

        for (index, layer) in self.layers.iter_mut().enumerate().rev() {
            let mut gradients = results[index].map_ret(&dsigmoid);
            gradients.multiply_matrix(&errors);
            gradients.multiply_scalar(self.learning_rate);

            let transposed = if index == 0 {
                orig_inputs.transpose()
            } else {
                results[index - 1].transpose()
            };
            let weights_deltas = gradients.product(&transposed);

            layer.weights.add_matrix(&weights_deltas);
            layer.bias.add_matrix(&gradients);

            let weights_t = layer.weights.transpose();
            errors = weights_t.product(&errors);
        }
    }
}
