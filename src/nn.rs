use crate::matrix::Matrix;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// Assumes y comes from sigmoid()
fn dsigmoid(y: f32) -> f32 {
    y * (1.0 - y)
}

#[derive(Debug)]
pub struct NeuralNetwork {
    weights_ih: Matrix,
    weights_ho: Matrix,
    bias_h: Matrix,
    bias_o: Matrix,
    learning_rate: f32,
}

impl NeuralNetwork {
    pub fn new(n_input: usize, n_hidden: usize, n_output: usize) -> Self {
        let mut weights_ih = Matrix::new(n_hidden, n_input);
        let mut weights_ho = Matrix::new(n_output, n_hidden);
        let mut bias_h = Matrix::new(n_hidden, 1);
        let mut bias_o = Matrix::new(n_output, 1);
        weights_ih.randomize();
        weights_ho.randomize();
        bias_h.randomize();
        bias_o.randomize();
        Self {
            weights_ih,
            weights_ho,
            bias_h,
            bias_o,
            learning_rate: 0.003,
        }
    }

    pub fn feedforward(&self, input: Vec<f32>) -> Vec<f32> {
        let inputs = Matrix::from_vec(input);
        let mut hidden = self.weights_ih.product(&inputs);
        hidden.add_matrix(&self.bias_h);

        // Activation function
        hidden.map(&sigmoid);

        let mut output = self.weights_ho.product(&hidden);
        output.add_matrix(&self.bias_o);

        output.to_vec()
    }

    pub fn train(&mut self, inputs: Vec<f32>, targets: Vec<f32>) {
        let inputs = Matrix::from_vec(inputs);
        let mut hidden = self.weights_ih.product(&inputs);
        hidden.add_matrix(&self.bias_h);
        // Activation function
        hidden.map(&sigmoid);

        let mut outputs = self.weights_ho.product(&hidden);
        outputs.add_matrix(&self.bias_o);
        outputs.map(&sigmoid);

        let targets = Matrix::from_vec(targets);

        let output_errors = targets.subtract_matrix(&outputs);

        // Calculate gradient
        let mut gradients = outputs.map_ret(&dsigmoid);
        gradients.multiply_matrix(&output_errors);
        gradients.multiply_scalar(self.learning_rate);

        // Calculate deltas
        let hidden_t = hidden.transpose();
        // let weights_ho_deltas =  gradients.multiply_matrix_ret(&hidden_t);
        let weights_ho_deltas =  gradients.product(&hidden_t);

        // Adjust the wieghts by deltas
        self.weights_ho.add_matrix(&weights_ho_deltas);
        // Adjust the bias by its deltas
        self.bias_o.add_matrix(&gradients);

        let who_t = self.weights_ho.transpose();
        // let hidden_errors = who_t.multiply_matrix_ret(&output_errors);
        let hidden_errors = who_t.product(&output_errors);

        // Calculate hidden gradient
        let mut hidden_gradient = hidden.map_ret(&dsigmoid);
        hidden_gradient.multiply_matrix(&hidden_errors);
        hidden_gradient.multiply_scalar(self.learning_rate);

        // Calculate input->hidden deltas
        let inputs_t = inputs.transpose();
        // let weights_ih_deltas =  hidden_gradient.multiply_matrix_ret(&inputs_t);
        let weights_ih_deltas =  hidden_gradient.product(&inputs_t);
        self.weights_ih.add_matrix(&weights_ih_deltas);
        // Adjust the bias by its deltas
        self.bias_h.add_matrix(&hidden_gradient);
    }
}
