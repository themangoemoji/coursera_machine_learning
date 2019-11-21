function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Shrink size of X/y
feature_size = size(X, 1);
percent_removed = 8/9;
random_removals = floor(feature_size * percent_removed);
tempx = X;
tempy = y;
k = randperm(size(X,1));
tempx(k(1:random_removals),:) = [];
tempy(k(1:random_removals),:) = [];


model = svmTrain(tempx, tempy, C, @linearKernel, 1e-3, 20);
prediction = svmPredict(model, tempx);
err = sum(prediction == tempy);

values = [0.0001, 0.0003, 0.001, 0.003, 0.01 0.03 0.1 0.3 1 3 10 30];
[vec1, vec2] = meshgrid(values, values);
combination_errors = [vec1(:), vec2(:)];
% This is a matrix of C, sigma combination pairs with a space for their error val
combination_errors = [combination_errors, zeros(size(combination_errors),1)];


for combination = 1:size(combination_errors, 1)
	C = combination_errors(combination, 1);
	sigma = combination_errors(combination, 2);
	model = svmTrain(tempx, tempy, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
	err = mean(double(svmPredict(model, Xval) ~= yval));
	combination_errors(combination, 3) = err;
	
end

steps = [1:size(combination_errors,1)];
errors = combination_errors(:,3);
plot(steps,errors);

for C = values
  for sigma = values
    fprintf('.');
    model = svmTrain(tempx, tempy, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    err   = mean(double(svmPredict(model, Xval) ~= yval));
    if( err <= error_min )
      C_final     = C;
      sigma_final = sigma;
      error_min   = err;
      fprintf('new min found C, sigma = %f, %f with error = %f', C_final, sigma_final, error_min)
    end
  end
end
C     = C_final;
sigma = sigma_final;
fprintf('Best value C, sigma = [%f %f] with prediction error = %f\n', C, sigma, error_min);

keyboard();

% =========================================================================

end
