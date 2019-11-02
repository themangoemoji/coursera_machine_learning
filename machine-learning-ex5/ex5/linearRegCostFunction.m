function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%



% Error
errors = sum(((X * theta) - y) .^ 2);
errors = errors ./ (2*m);

theta_without_bias = theta;
theta_without_bias(1,1) = 0;
regularized_error = errors + (lambda * sum((theta_without_bias .^ 2 / (2 * m))));
J = regularized_error;


% Gradient
gradient = (X' * ((X*theta) - y)) / m;
theta_without_bias = theta;
theta_without_bias(1,1) = 0;
gradient += ((theta_without_bias * lambda) / m);
grad = gradient;

% =========================================================================

grad = grad(:);

end
