

pub const INNOVATION_NUM:usize = 1;

#[derive(Clone,Debug)] 
pub struct NeatBrain {
    pub nodes: Vec<Node>,
    pub connections: Vec<Connection>,
    pub layers: u32,
}
#[derive(Clone,Debug)] 
pub struct Node {
    pub node_id: usize, // ID of the Node
    pub subtype: u32, // 0 for sensor, 1 for hidden and 2 for output
    pub layer: u32,
    pub value: f32,
}
#[derive(Clone,Debug)] 
pub struct Connection {
    pub connection_id: u32, // ID number of the connection
    pub input: usize, // Id of the Input Node
    pub output: usize, // Id of the Output Node
    pub weight: f32,
    pub status: bool, // TRUE is active, FALSE is deactivated
}

use rand::Rng;


impl NeatBrain { 
    /// [#nodes #connections_on_start # output]
    pub fn new_brain(network_shape: Vec<usize>, innovation_num: &mut usize) -> Self {


        // Create new nodes for input
        let mut nodes = Vec::new();
        for _ in 0..(network_shape[0]) {
            nodes.push(Node::new(innovation_num,0,1));
        }

        // Create new nodes for output
        for _ in 0..(network_shape[2]) {
            nodes.push(Node::new(innovation_num,2, 2));
        }

        let connections = Vec::new();
        let layers = 2;

        // Creates a brain with empty connections
        let mut brain = Self {
            nodes,
            connections,
            layers,
        };

        // Creates network_shape[1] amount of new connections
        for _ in 0..(network_shape[1]) {
            brain.new_connection(innovation_num);
        }
        brain

    }


    fn new_connection(&mut self,innovation_num: &mut usize) {
        let node_ids = self.nodes.iter().map(|node| node.node_id).collect::<Vec<usize>>();
        
        // Collect all potential input nodes
        let available_inputs = self.nodes.iter().filter(|n| n.subtype != 2).map(|n| n.node_id).collect::<Vec<usize>>();

        let mut rng = rand::thread_rng();
        
        let mut input: usize;
        let mut output: usize;
    
        loop {
            input = *available_inputs.get(rng.gen_range(0..available_inputs.len())).unwrap_or(&0);
            
            // Collect only output nodes that are in the same or higher layer as the selected input && not input neurons
        let available_outputs: Vec<usize> = node_ids.iter().filter(|&&id| {
            let id_index = self.nodes.iter().position(|r| r.node_id == id).unwrap();
            self.nodes[id_index].subtype != 0 && self.nodes[id_index].layer >= self.nodes[self.nodes.iter().position(|r| r.node_id == input).unwrap()].layer
        }).copied().collect::<Vec<usize>>();        

            output = *available_outputs.get(rng.gen_range(0..available_outputs.len())).unwrap_or(&0);
    
            if self.connections.iter().all(|c| c.input != input || c.output != output) && input != output {
                break;
            } else {
                // println!("Connection between node_id {} and node_id {} already exists or is invalid, trying again...", input, output);
            }
        }

        // If two neurons are in the same layer, create a new layer for the output layer and update all layers of downstream neurons.
        let input_index = self.nodes.iter().position(|r| r.node_id == input).unwrap();
        let output_index = self.nodes.iter().position(|r| r.node_id == output).unwrap();

        if self.nodes[input_index].layer == self.nodes[output_index].layer {
            self.nodes[output_index].layer += 1;
            self.update_layers(self.nodes[output_index].node_id);
        }

        *innovation_num += 1;

        let node_weight: f32 = rng.gen_range(-1.0..1.0);

        let new_connection = Connection {
            connection_id: self.connections.len() as u32 + 1,
            input,
            output,
            weight: node_weight,
            status: true,
        };
    
        self.connections.push(new_connection);
        // println!("Successfully created new connection between node_id {} and node_id {}", input, output);
    }
    
    // The `update_layers()` function goes through all `nodes` and increments the layer of every node that is on a higher layer than the `output` node, excluding the `output` node itself.
    fn update_layers(&mut self, output: usize) {
        let output_node = self.nodes.iter().find(|&node| node.node_id == output);
        match output_node {
            Some(node) => {
                let new_layer = node.layer;
                for node in &mut self.nodes {
                    if node.node_id != output && node.layer > new_layer {
                        node.layer += 1;
                    }
                }
                self.layers += 1;
            },
            None => {
                // Handle the case where there's no node with the provided output id
            },
        }
    }


    pub fn compute(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        
        let nodes_len = self.nodes.len();
        let mut new_node_values: Vec<f32> = vec![0.; nodes_len];
        let input_nodes: Vec<Node> = self.nodes.iter().filter(|node| node.subtype == 0).cloned().collect();

        for (i, node) in input_nodes.iter().enumerate() {
            let index = self.nodes.iter().position(|n| n.node_id == node.node_id).unwrap();
            new_node_values[index] = inputs[i];   

        }


        for layer in 1..=self.layers {
            for node in self.nodes.iter().filter(|node| node.layer == layer) {
                let mut input_sum: f32 = 0.;

                // Finds connection, if active it calculates the final node
                for conn in &self.connections {
                    if conn.output == node.node_id && conn.status{
                        let input_node_index = self.nodes.iter().position(|n| n.node_id == conn.input).unwrap();

                        input_sum += &new_node_values[input_node_index] * conn.weight;

                        let node_index = self.nodes.iter().position(|n| n.node_id == node.node_id).unwrap();

                        new_node_values[node_index] = self.activate(input_sum);
                    }
                }


            }
        }

        // Assign new node values to actual nodes
        for (i, new_value) in new_node_values.iter().enumerate() {
            self.nodes[i].value = *new_value;
        }
    
        let output_values: Vec<f32> = self.nodes.iter()
            .filter(|node| node.subtype == 2)
            .map(|node| node.value)
            .collect();
        
        // println!("{:?}",output_values);

        output_values
    }


    // Iterates over the whole brain
    pub fn mutate(&mut self, innovation_num: &mut usize, mutation_chance: f32, mutation_amount: f32,node_creation_chance: f32, connection_creation_chance: f32) {
        let mut rng = rand::thread_rng();


        // Iterate over each connection and mutate some weights
        for connection in &mut self.connections {
            // If random value is less than mutation_chance, mutate the weight
            if rng.gen::<f32>() < mutation_chance {
                let change = (rng.gen::<f32>() * 2.0 - 1.0) * mutation_amount;
                connection.weight += change;

                // Clipping the weight value to be within a certain range, if necessary
                connection.weight = connection.weight.max(-1.0).min(1.0);
            }
        }

        

        //Split connection into node and two connections
        if rng.gen::<f32>() < node_creation_chance {
            let conn_index = rng.gen_range(0..self.connections.len()); // pick a random connection
            let old_connection = &mut self.connections[conn_index].clone();
    
            if old_connection.status {

                self.connections[conn_index].status = false; // deactivate it
            
                let new_node_id = self.nodes.len(); // generate a new node id
                
                let input_node_layer: u32 = self.nodes.iter().find(|node| node.node_id == old_connection.input).expect("Input node not found").layer; // Get input layer
                let new_node_layer = input_node_layer + 1;
                // println!("A New node is created in layer {}",new_node_layer);


                let output_node_id= old_connection.output; // Get output id
                // Iterate to all the nodes and bump their layer +1 if their downstream node is the same as the node. Recursive.
                self.bump_layers(new_node_layer, output_node_id);

                let new_node = Node {
                    node_id: new_node_id,
                    subtype: 1, // for hidden 
                    layer: new_node_layer,  //TODO: You need to add +1 to current layer and check if new layer is same as output. If yes- that layer has to go +1.
                    value: 0.0,
                };
                
                let inov1 = *innovation_num as u32;
                let inov2 = *innovation_num as u32+ 2;


                let new_conn1 = Connection {
                    connection_id: inov1,
                    input: old_connection.input,
                    output: new_node_id,
                    weight: 1.0,
                    status: true,
                };
                
                let new_conn2 = Connection {
                    connection_id: inov2, // ++innovation_num,
                    input: new_node_id,
                    output: old_connection.output,
                    weight: old_connection.weight,
                    status: true,
                };
    
                self.nodes.push(new_node);
                self.connections.push(new_conn1);
                self.connections.push(new_conn2);
    
                *innovation_num += 2; // increase the global innovation number by 2
            }
        }

        // Create new connection
        if rng.gen::<f32>() < connection_creation_chance {
            self.new_connection(innovation_num);
        }

        // Deactivate connection

        // Create memory Neuron


    }

    // // Function to bump layers if new connections nodes are created on the same layer
    // pub fn bump_layers(&mut self, new_node_layer: u32 ,output_node_id:usize) {
        
    //     let output_node_layer: u32 = self.nodes.iter().find(|node| node.node_id == output_node_id).expect("Input node not found").layer; // Get output node id layer
    //     if output_node_layer > new_node_layer {
    //         // new node layer is larger than output node layer, sequence is correct. Do nothing
    //         return;
    //     } else {
    //         // Bump the layer
    //         self.nodes.iter_mut().find(|node| node.node_id == output_node_id).expect("Input node not found").layer += 1;

    //         // iterate through all connections that have output_node_id as initial connection



    //     }
    // }

    pub fn bump_layers(&mut self, new_node_layer: u32, output_node_id: usize) {
        let output_node_layer: u32 = self.nodes.iter().find(|node| node.node_id == output_node_id).unwrap().layer;
        
        if output_node_layer == new_node_layer { // If on the same layer

            if self.nodes.iter().find(|node| node.node_id == output_node_id).unwrap().subtype == 1 { // For middle neurons

                self.nodes.iter_mut().find(|node| node.node_id == output_node_id).unwrap().layer += 1; // Bump the layer of output_node of the connection
                
                // get all connections that have this output node as input
                let connections_to_bump_ids: Vec<usize> = self.connections.iter().filter(|connection| connection.input == output_node_id).map(|c| c.output).collect();
                
                // Modify the connections' layers
                for connection_output_id in connections_to_bump_ids {
                    self.bump_layers(new_node_layer + 1, connection_output_id);
                }
                // println!("A Node's Layer is bumped to layer {}", self.nodes.iter_mut().find(|node| node.node_id == output_node_id).unwrap().layer);
            } else if self.nodes.iter().find(|node| node.node_id == output_node_id).unwrap().subtype == 2 { // For output neurons
                for node in &mut self.nodes {
                    if node.subtype == 2 {
                        node.layer += 1;
                        // println!("An output Layer is bumped to layer {}",node.layer);
                    }
                }            
                self.layers +=1;
            }
        }
    }


    // TODO
    fn activate(&self, x: f32) -> f32 {
        x
    }



 //new()

    //calculate with nodes and connection
    //     Add input to corresponding Node
    //     for each Connection that connects with the main node, find corresponding node and multiply by connection weight. Add to vec. 
    //     return the output in the correct order

 //calculate () Note: for calculations, should we us layers? That way we can go though every layer and calculate those Nodes

 //mutate()
    // Add Connection between two nodes with random value
    // Add Node, deactivate prev connection and add two connections
    // Weight shifting: Add node to 
}

impl Node {
    // New node with new brain
    pub fn new(innovation_num: &mut usize, subtype: u32,node_layer: u32) -> Self {
        *innovation_num += 1;
        Self {
            node_id: *innovation_num, // ID of the Node
            subtype: subtype, // 0 for sensor, 1 for hidden and 2 for output
            layer: node_layer,
            value: 0.0,
        }
    }
}

impl Connection {

}

// Types of
//      New
//      Mutate: Includes creating nodes, connections and 
//      Forward: calculates the nn outputs