% Function to predict the output of a sequence using a certain network
function [output, state_storage] = net_model(input, weights, state)

% Define sigmoid and tanh as anonymous functions
functions = struct('sigmoid', @(x) 1 ./ (1 + exp(-x)), 'tanh', @(x) tanh(x));

% Extract info
num_outputs = height(weights.layer_fc.weights);
num_timesteps = height(input);
num_layers = height(weights.hidden_units);
last_layer = ['layer_' num2str(num_layers)];
net_type =  weights.net_type;

% Initialize output
output = zeros(num_timesteps,num_outputs);

% Initialize state storage
for i = 1:length(weights.hidden_units)
    layer_name = ['layer_' num2str(i)];
    num_units = weights.hidden_units(i);

    if strcmp(net_type, 'lstm')
        % Initialize LSTM storage states (hidden and cell)
        state_storage.(layer_name) = struct();
        state_storage.(layer_name).h = zeros(num_units, num_timesteps);  % hidden state
        state_storage.(layer_name).c = zeros(num_units, num_timesteps);  % cell state
    else
        % Initialize GRU storage states (hidden only)
        state_storage.(layer_name) = struct();
        state_storage.(layer_name).h = zeros(num_units, num_timesteps);  % hidden state
    end
end

% Process each timestep through the network
for t = 1:num_timesteps
    % ---- Enter the simil net ----

    % Current timestep input
    u_t = input(t, :)';
    state.layer_1.u = u_t;
    
    % ---- Recurrent Layers ----
    if strcmp(net_type, 'lstm')
       for i = 1:num_layers
           layer_name = ['layer_' num2str(i)];
           layer_name_next = ['layer_' num2str(i+1)];

           % Compute new state
           state.(layer_name) = lstm_step(state.(layer_name), weights.(layer_name), functions);

           % Store state
           state_storage.(layer_name).h(:,t) = state.(layer_name).h;
           state_storage.(layer_name).c(:,t) = state.(layer_name).c;

           % Save hidden state of the layer as next input
           if i ~= num_layers
            state.(layer_name_next).u = state.(layer_name).h;
           end
       end
    else
        for i = 1:num_layers
            layer_name = ['layer_' num2str(i)];
            layer_name_next = ['layer_' num2str(i+1)];

            % Compute new state
            state.(layer_name) = gru_step(state.(layer_name), weights.(layer_name), functions);

            % Store state
            state_storage.(layer_name).h(:,t) = state.(layer_name).h;

            % Save hidden state of the layer as next input
            if i ~= num_layers
                state.(layer_name_next).u = state.(layer_name).h;
            end
        end
    end
    
    % ---- Fully Connected Layer ----
    % Compute the output with the fully connected layer weights
    output(t,:) = (weights.layer_fc.weights * state.(last_layer).h + weights.layer_fc.bias)';

    % ---- Exit the simil net ----
end

end
