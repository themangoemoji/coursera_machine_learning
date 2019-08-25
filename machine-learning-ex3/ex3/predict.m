function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%[max_value, index_of_max_value] = max((sigmoid(([ones(size(X,1),1), sigmoid(X * Theta1')]) * Theta2'))');

X = [ones(size(X,1),1), X]; % Add bias column
first_activation = sigmoid(X * Theta1');
first_activation = [ones(size(first_activation, 1), 1), first_activation]; % Add bias column

second_activation = sigmoid(first_activation * Theta2');

[max_value, index_of_max_value] = max(second_activation'); % Turn matrix to get max per row
p = index_of_max_value';


% =========================================================================


end
