% Function to intialize the state and memories of a standard RNN network
function state = initialize_state(weights)

% Check wether it is a gru or lstm network
net_type =  weights.net_type;

% Loop through each layer in the weights struct
for i = 1:length(weights.hidden_units)
    layer_name = ['layer_' num2str(i)];
    num_units = weights.hidden_units(i);

    if strcmp(net_type, 'lstm')
        % Initialize LSTM states (hidden and cell)
        state.(layer_name) = struct();
        state.(layer_name).h = zeros(num_units, 1);  % hidden state
        state.(layer_name).c = zeros(num_units, 1);  % cell state
    else
        % Initialize GRU states (hidden only)
        state.(layer_name) = struct();
        state.(layer_name).h = zeros(num_units, 1);  % hidden state
    end
end

end