# RNN-to-dynamic-system
Piece of code to extract the weights of an existing RNN network and use them to predict a sequence as a dynamic system


## Folder Structure

The code consists of the following main components:

1. `net_to_model.m`: The primary function for testing the code.
2. `utilities`: A folder containing utility functions required for the process.
3. `tester.m`: A script to test the functionality of the training process using a provided dataset.
4. `data`: Contains dataset for testing and validation purposes.
5. `net`: Contains trained network for testing and validation purposes.

## net_to_model.m

This function converts a RNN network, either LSTM or GRU, into a dynamic system by extracting the weights; later it predicts a sequence and a graphical comparison between this, the network prediction and the real response is shown.

Note: the network and model prediction will overlap as working the same way.

Note: the network is built by the previous [ISS-Promoted-RNN](https://github.com/StefanoDeCarli/ISS-Promoted-RNN) script.

## Format of Weights and States

### Weights Format

The weights are extracted using the `RNN_extract_weights` function. This function identifies whether the network is a GRU or LSTM and extracts the weights accordingly. The extracted weights are stored in a structured format with layers and gate-specific weights.

#### Example Weights Structure for LSTM
```matlab
weights = struct();
weights.net_type = 'lstm';
weights.hidden_units = [256, 256, 128, 64];
weights.description = 'Weights of a lstm network with [256,256,128,64] hidden units';
weights.layer_1.W = struct('f', W1_f, 'i', W1_i, 'g', W1_g, 'o', W1_o);
weights.layer_1.R = struct('f', R1_f, 'i', R1_i, 'g', R1_g, 'o', R1_o);
weights.layer_1.b = struct('f', b1_f, 'i', b1_i, 'g', b1_g, 'o', b1_o);
% ... similarly for other layers
weights.layer_fc = struct('weights', fc_weights, 'bias', fc_bias);
```

#### Example Weights Structure for GRU
```matlab
weights = struct();
weights.net_type = 'gru';
weights.hidden_units = [256, 256, 128, 64];
weights.description = 'Weights of a gru network with [256,256,128,64] hidden units';
weights.layer_1.W = struct('r', W1_r, 'z', W1_z, 'h', W1_h);
weights.layer_1.R = struct('r', R1_r, 'z', R1_z, 'h', R1_h);
weights.layer_1.b = struct('r', b1_r, 'z', b1_z, 'h', b1_h);
% ... similarly for other layers
weights.layer_fc = struct('weights', fc_weights, 'bias', fc_bias);
```

### States Format
The states are initialized using the `initialize_state` function. This function uses the weights structure to determine the number of hidden units and the type of network (LSTM or GRU) and initializes the states accordingly.

#### Example State Structure for LSTM
```matlab
state = struct();
state.layer_1.h = zeros(256, 1);  % hidden state for layer 1
state.layer_1.c = zeros(256, 1);  % cell state for layer 1
% ... similarly for other layers
```
#### Example State Structure for GRU
```matlab
state = struct();
state.layer_1.h = zeros(256, 1);  % hidden state for layer 1
% ... similarly for other layers
```

utilities
-------------

This folder contains utility functions required for the training process. Ensure that this folder is included in your MATLAB path.

Getting Started
---------------

1.  Clone or download the repository to your local machine. 
2.  Run the `net_to_model.m` script to test the sample trained network and save the results.
3.  Do whatever you want with the file :blush:

License
-------

This project is licensed under the MIT License. See the LICENSE file for more details.

Happy coding!
