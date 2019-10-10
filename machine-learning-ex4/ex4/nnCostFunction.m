function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. 
%   The parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
%   Y = [
%   0   0   0   0   0   0   0   0   0   1
%   0   0   0   0   0   0   0   0   0   1
%   .   .   .   .   .   .   .   .   .   .
%   1   0   0   0   0   0   0   0   0   0
%   1   0   0   0   0   0   0   0   0   0
%   .   .   .   .   .   .   .   .   .   .
%   0   1   0   0   0   0   0   0   0   0
%   0   1   0   0   0   0   0   0   0   0
%   .   .   .   .   .   .   .   .   .   .
%   0   0   1   0   0   0   0   0   0   0
%   0   0   1   0   0   0   0   0   0   0
%   .   .   .   .   .   .   .   .   .   .
%   .   .   .   .   .   .   .   .   .   .
%   0   0   0   0   0   0   0   0   1   0
%   0   0   0   0   0   0   0   0   1   0
%   ]


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% We will create Y= [0 1 0
%                    0 0 1
%                    .....
%                    1 0 0]
% From y =         [2, 3,... 1]
Identity_Matrix = eye(num_labels);
Y = Identity_Matrix(y, :);

% Feedforward
a1 = [ones(m, 1), X]; % Add bias column: 401f x m

z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1), a2];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

% Now, compute error
positive_error = Y .* log(a3);  % Classifier,Sample .* Output,Sample = Y' .* a3 = Classifier,Sample
negative_error = (1-Y) .* log(1 - a3); % Classifier,Sample .* Prediction,Sample = Y' .* third_activation
combined_unregularized_error = sum(sum(positive_error + negative_error)) * (-1 / m);
J = combined_unregularized_error;

% Add in regularization (thought it should be 0)

regularization = lambda / (2 * m);
summations = sum(sum(Theta1(:, 2:end) .^ 2)); % Don't sum over the bias term
summations = summations + sum(sum(Theta2(:, 2:end) .^ 2)); % Don't sum over the bias term
regularization = regularization * summations;

J = combined_unregularized_error + regularization;

% Now, calculate sigmas
s3 = a3 - Y;

z2_with_bias = [ones(size(z2, 1), 1), z2];
s2 = s3 * Theta2 .* sigmoidGradient(z2_with_bias);
s2 = s2(:, 2:end);

% Compute Deltas
delta_1 = [s2' * a1];
delta_2 = [s3' * a2];

% Unregularized Gradient
Theta1_grad = (delta_1 ./ m);
Theta2_grad = (delta_2 ./ m);


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Regularize non-bias terms
Theta1_grad(:, 2:end) += (lambda / m) .* Theta1(:, 2:end);
Theta2_grad(:, 2:end) += (lambda / m) .* Theta2(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
