use rand::Rng;

use crate::config::NN_ACTIVATION;


#[allow(dead_code)] // For network_shape in the Brain struct
#[derive(Clone)] 
pub struct Brain {
    layers: Vec<Layer>,
    network_shape: Vec<usize>,
}

#[derive(Clone)] 
pub struct Layer {
    weights_array: Vec<Vec<f32>>,
    biases_array: Vec<f32>,
    node_array: Vec<f32>,
    n_nodes: usize,
    n_inputs: usize,
}


impl Layer {
    pub fn new(n_inputs: usize, n_nodes: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            weights_array: (0..n_nodes).map(|_| (0..n_inputs).map(|_| ((rng.gen::<f32>()* 2.0) - 1.0)).collect()).collect(),
            biases_array: (0..n_nodes).map(|_| ((rng.gen::<f32>()* 2.0) - 1.0)).collect(),
            node_array: vec![0.0; n_nodes],
            n_nodes,
            n_inputs,
        }
    }
    // ...

    pub fn forward(&mut self, inputs_array: &[f32]) {
        self.node_array = vec![0.0; self.n_nodes];
        for i in 0..self.n_nodes {
            for j in 0..self.n_inputs {
                self.node_array[i] += self.weights_array[i][j] * inputs_array[j];
            }
            self.node_array[i] += self.biases_array[i];
        }
    }

    pub fn activation(&mut self) {  
        // ReLU activation function 
        if NN_ACTIVATION.eq("RELU") {
            for i in 0..self.n_nodes {
                if self.node_array[i] < 0.0 {
                    self.node_array[i] = 0.0;
                }
            }
        }
        // Sigmoid activation function
        if NN_ACTIVATION.eq("SIGM") {
            for i in 0..self.n_nodes {
                let sigm = self.node_array[i];
                self.node_array[i] = 1.0/(1.0 + (-sigm).exp());
            }
        } 
    }
    
    pub fn mutate(&mut self, mutation_chance: f32, mutation_amount: f32) {
        let mut rng = rand::thread_rng();
        for i in 0..self.n_nodes {
            for j in 0..self.n_inputs {
                if rand::random::<f32>() < mutation_chance {
                    self.weights_array[i][j] += (rng.gen::<f32>()* 2.0 - 1.0) * mutation_amount;
                }
            }
            if rand::random::<f32>() < mutation_chance {
                self.biases_array[i] += (rng.gen::<f32>()* 2.0 - 1.0) * mutation_amount;
            }
        }
    }
}

impl Brain {
    pub fn new(network_shape: Vec<usize>) -> Self {
        let mut layers = Vec::new();
        for i in 0..(network_shape.len() - 1) {
            layers.push(Layer::new(network_shape[i], network_shape[i+1]));
        }
        Self {
            layers,
            network_shape,
        }
    }

    pub fn process(&mut self, inputs: &[f32]) -> &[f32] {
        let mut previous_layer_output: Vec<f32> = inputs.to_vec();
        let mut all_layers_output: Vec<Vec<f32>> = Vec::new();
        all_layers_output.push(previous_layer_output.clone());
    
        for layer in &mut self.layers {
            layer.forward(&previous_layer_output);
            layer.activation();
            previous_layer_output = layer.node_array.clone();
            all_layers_output.push(previous_layer_output.clone());
        }
        for layer in &mut self.layers {
            layer.forward(all_layers_output.remove(0).as_slice());
            layer.activation();
        }
    
        &self.layers.last().unwrap().node_array
    }
    pub fn mutate(&mut self, mutation_chance: f32, mutation_amount: f32) {
        for layer in &mut self.layers {
            layer.mutate(mutation_chance, mutation_amount);
        }
    }
}