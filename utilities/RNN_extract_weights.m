% Function to get the weights of a standard RNN network
function weights = RNN_extract_weights(net)

% Extract the amount of recurrent layers, assuming a structure with RNNs and a final fully connected layer
num_RNN_layers = (height(net.Learnables) - 2) / 3;

% Initialize the hidden_units array
hidden_units = zeros(num_RNN_layers, 1);

% Determine the type of network (GRU or LSTM) by examining the first entry in net.Learnables
first_layer = net.Learnables.Layer{1};
if contains(first_layer, 'gru', 'IgnoreCase', true)
    net_type = 'gru';
elseif contains(first_layer, 'lstm', 'IgnoreCase', true)
    net_type = 'lstm';
else
    error('Unsupported network type. Only GRU and LSTM layers are supported.');
end

% Loop through each RNN layer, extracting the weights and creating the new struct object
for i = 1:num_RNN_layers
    layer_name = ['layer_' num2str(i)];

    % Extract InputWeights, RecurrentWeights, and Bias for the current layer
    inputWeights = net.Learnables.Value{3*i - 2};
    recurrentWeights = net.Learnables.Value{3*i - 1};
    bias = net.Learnables.Value{3*i};

    % Extract the width from RecurrentWeights to get the number of hidden units
    hidden_units(i) = size(recurrentWeights, 2);

    % Split weights and biases into gates
    if strcmp(net_type, 'lstm')
        [W_i, W_f, W_g, W_o] = split_lstm_weights(inputWeights, hidden_units(i));
        [R_i, R_f, R_g, R_o] = split_lstm_weights(recurrentWeights, hidden_units(i));
        [b_i, b_f, b_g, b_o] = split_lstm_biases(bias, hidden_units(i));

        % Populating weights for the current LSTM layer
        weights_layer = struct();
        weights_layer.W = struct('f', W_f, 'i', W_i, 'g', W_g, 'o', W_o);
        weights_layer.R = struct('f', R_f, 'i', R_i, 'g', R_g, 'o', R_o);
        weights_layer.b = struct('f', b_f, 'i', b_i, 'g', b_g, 'o', b_o);
    else
        [W_r, W_z, W_h] = split_gru_weights(inputWeights, hidden_units(i));
        [R_r, R_z, R_h] = split_gru_weights(recurrentWeights, hidden_units(i));
        [b_r, b_z, b_h] = split_gru_biases(bias, hidden_units(i));

        % Populating weights for the current GRU layer
        weights_layer = struct();
        weights_layer.W = struct('r', W_r, 'z', W_z, 'h', W_h);
        weights_layer.R = struct('r', R_r, 'z', R_z, 'h', R_h);
        weights_layer.b = struct('r', b_r, 'z', b_z, 'h', b_h);
    end

    weights.net_type = net_type;
    weights.hidden_units = hidden_units;

    weights.(layer_name) = weights_layer;
end

% Extract weights and biases for the Fully Connected Layer
fc_weights = net.Learnables.Value{num_RNN_layers * 3 + 1};
fc_bias = net.Learnables.Value{num_RNN_layers * 3 + 2};
weights.layer_fc = struct('weights', fc_weights, 'bias', fc_bias);

% Create a dynamic description
hidden_units_str = strjoin(arrayfun(@num2str, hidden_units', 'UniformOutput', false), ',');
weights.description = ['Weights of a ' net_type ' network with [' hidden_units_str '] hidden units'];

weights = orderfields(weights);
end

function [W_i, W_f, W_g, W_o] = split_lstm_weights(W, num_units)
W_i = W(1:num_units, :);
W_f = W(num_units+1:2*num_units, :);
W_g = W(2*num_units+1:3*num_units, :);
W_o = W(3*num_units+1:end, :);
end

function [b_i, b_f, b_g, b_o] = split_lstm_biases(b, num_units)
b_i = b(1:num_units);
b_f = b(num_units+1:2*num_units);
b_g = b(2*num_units+1:3*num_units);
b_o = b(3*num_units+1:end);
end

function [W_r, W_z, W_h] = split_gru_weights(W, num_units)
W_r = W(1:num_units, :);
W_z = W(num_units+1:2*num_units, :);
W_h = W(2*num_units+1:end, :);
end

function [b_r, b_z, b_h] = split_gru_biases(b, num_units)
b_r = b(1:num_units);
b_z = b(num_units+1:2*num_units);
b_h = b(2*num_units+1:end);
end
