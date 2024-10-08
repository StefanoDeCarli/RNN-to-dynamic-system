% Clear the workspace and initialize consistent random values
clc;
clear;
close all;
random_seed = 1;
rng(random_seed);

% Utility functions directory
addpath(genpath([pwd, filesep, 'utilities']));

% Load dataset
data = load(fullfile('data', 'SMI_data.mat'));
varName = fieldnames(data);   % Get the field name(s) in the structure
data = data.(varName{1});     % Access the contents using dynamic field referencing

% Load net
net_to_use = load(fullfile('net', 'test_net.mat'));
varName = fieldnames(net_to_use);   % Get the field name(s) in the structure
net_to_use = net_to_use.(varName{1});     % Access the contents using dynamic field referencing

dataset = data.validation_30s;

% Just for this dataset
dataset.x = transpose_cell(dataset.x);
dataset.y = transpose_cell(dataset.y);

% Select the trial to test the model onto
trial = 2;
input = dataset.x{trial};

% Extract the weights from the net
weigths = RNN_extract_weights(net_to_use.net);
% Initialize the state for each prediction
state = initialize_state(weigths);

% Execute the prediction as a dynamic system for the whole sequence
[output, state_storage] = net_model(input,weigths,state);

% Predict with network
nn_pred = predict(net_to_use.net, dataset.x{trial})';

% Select the output to visualize in comparison
which_output = 7;
t = (1:height(output));

% Plot the results
F = figure;
hold on;
plot(t, output(:,which_output), 'LineWidth', 2);
plot(t, nn_pred(which_output,:), 'LineWidth', 2, 'LineStyle','--');
plot(t, dataset.y{trial}(:,which_output), 'LineWidth', 2);
xlabel('Time (t)', 'FontSize', 24);
ylabel('Normalized outputs', 'FontSize', 24);
legend('Model prediction', 'Network prediction', 'Real output', 'Location', 'best', 'FontSize', 24);
title('Comparing model to network to reality', 'FontSize', 24);
F.Color = 'w';
grid on;
hold off;

%% Useful functions for the test
function cell_array = transpose_cell(cell_array)
    for i = 1:length(cell_array)
        cell_array{i} = transpose(cell_array{i});
    end 
end