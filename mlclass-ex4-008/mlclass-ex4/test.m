

clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

load('ex4data1.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

load('ex4weights.mat');

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

lambda = 0;

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

m = size(X, 1);

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


X = [X , ones(m, 1)];
hypothesis1 = sigmoid(X * Theta1'); 								%'

a2 = (1 / m) * sum( -y .* log(hypothesis1) - (1 - y)  .* log(1 - hypothesis1) );

a2 = [a2, ones(size(a2, 1))];
hypothesis2 = sigmoid(a2 * Theta2'); 								%'


J = (1 / m) * sum(-y .* log(hypothesis2) - (1 - y) .* log(1 - hypothesis2));





size(Theta1)
size(Theta2)

% hypothesis = sigmoid(X * nn_params)

%J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

%fprintf(['Cost at parameters (loaded from ex4weights): %f ', '\n(this value should be about 0.287629)\n'], J);