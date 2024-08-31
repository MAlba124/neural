use crate::matrix::Matrix;

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// Assumes y comes from sigmoid()
#[inline(always)]
fn dsigmoid(y: f32) -> f32 {
    y * (1.0 - y)
}

#[derive(Debug)]
pub struct Layer {
    pub weights: Matrix,
    pub bias: Matrix,
    // Reusable buffer
    pub gradients: Matrix,
    pub transposed: Matrix,
    pub weights_t: Matrix,
    pub weights_deltas: Matrix,
}

#[derive(Debug)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
    // Reusable buffers for the feed forward step
    results: Vec<Matrix>,
    learning_rate: f32,
}

impl NeuralNetwork {
    pub fn new(n_input: usize, hidden: Vec<usize>, n_output: usize) -> Self {
        assert!(!hidden.is_empty() && n_output > 0);

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
                    weights,
                    bias,
                    gradients: Matrix::new(1, 1),
                    transposed: Matrix::new(1, 1),
                    weights_t: Matrix::new(input_weights_count, neuron_count),
                    weights_deltas: Matrix::new(1, 1),
                }
            );
            input_weights_count = neuron_count;
        }

        let mut results = Vec::new();
        let mut inputs = (n_input, 1);
        for layer in layers.iter_mut() {
            let size = (layer.weights.size().0, inputs.1);
            layer.gradients = Matrix::new(size.0, size.1);
            layer.transposed = Matrix::new(inputs.1, inputs.0);
            layer.weights_deltas = Matrix::new(layer.gradients.size().0, layer.transposed.size().1);
            results.push(Matrix::new(size.0, size.1));
            inputs = size;
        }

        Self {
            learning_rate: 0.003,
            results,
            layers,
        }
    }

    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    pub fn feedforward(&mut self, input: Vec<f32>) -> Vec<f32> {
        let inputs = Matrix::from_slice(&input);

        for (index, layer) in self.layers.iter().enumerate() {
            let inps = if index > 0 {
                    self.results[index - 1].clone()
                } else {
                    inputs.clone()
                };
            layer.weights.product_into(
                &inps,
                &mut self.results[index]
            );
            self.results[index].add_matrix(&layer.bias);
            self.results[index].map(&sigmoid);
        }

        self.results.last().unwrap().to_vec()
    }

    pub fn train(&mut self, inputs: &[f32], targets: &[f32]) {
        let inputs = Matrix::from_slice(inputs);
        let orig_inputs = inputs.clone();

        for (index, layer) in self.layers.iter().enumerate() {
            let inps = if index > 0 {
                    self.results[index - 1].clone()
                } else {
                    inputs.clone()
                };
            layer.weights.product_into(
                &inps,
                &mut self.results[index]
            );
            self.results[index].add_matrix(&layer.bias);
            self.results[index].map(&sigmoid);
        }

        let targets = Matrix::from_slice(targets);
        let outputs = self.results.last().unwrap();
        let mut errors = targets.subtract_matrix(outputs);

        // Skip first element so we can ommit a branching
        for (index, layer) in self.layers.iter_mut().enumerate().skip(1).rev() {
            self.results[index].map_into(&dsigmoid, &mut layer.gradients);
            layer.gradients.multiply_matrix(&errors);
            layer.gradients.multiply_scalar(self.learning_rate);

            self.results[index - 1].transpose_into(&mut layer.transposed);

            layer.gradients.product_into(&layer.transposed, &mut layer.weights_deltas);

            layer.weights.add_matrix(&layer.weights_deltas);
            layer.bias.add_matrix(&layer.gradients);

            layer.weights.transpose_into(&mut layer.weights_t);
            errors = layer.weights_t.product(&errors);
        }

        let layer = &mut self.layers[0];
        self.results[0].map_into(&dsigmoid, &mut layer.gradients);
        layer.gradients.multiply_matrix(&errors);
        layer.gradients.multiply_scalar(self.learning_rate);

        orig_inputs.transpose_into(&mut layer.transposed);
        layer.gradients.product_into(&layer.transposed, &mut layer.weights_deltas);

        layer.weights.add_matrix(&layer.weights_deltas);
        layer.bias.add_matrix(&layer.gradients);

        layer.weights.transpose_into(&mut layer.weights_t);
    }
}
