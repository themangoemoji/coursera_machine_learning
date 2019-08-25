function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% Compute regularized cost from normal cost func
hypothesis = sigmoid(X * theta);
regularization = (lambda / (2*m)) * sum(theta(2:length(theta)) .^ 2);

pos_unregularized_cost = (-y' * log(hypothesis));
neg_unregularized_cost = ((1-y') * log(1-hypothesis));
unregularized_cost = (pos_unregularized_cost - neg_unregularized_cost)/m;
J = unregularized_cost + regularization;

X_0 = X(:,1);
grad_theta_0 = ((1/m) * (X_0' * (hypothesis - y)));

Xs_greater_than_0 = X(:,2:size(X)(2));
thetas_gtr_0 = theta(2:length(theta),:);
grad_theta_gtr_0 = ((1/m) * (Xs_greater_than_0' * (hypothesis - y))) + ((lambda/m) * thetas_gtr_0);

grad = [grad_theta_0; grad_theta_gtr_0]; % Combine gradients for all theta

% =============================================================

end
