function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

m = length(y); % number of training examples

% initialize output variables
J = 0;
grad = zeros(size(theta));

% compute sigmoid of all elements in X
h = sigmoid(X * theta);

% compute cost (regularized) and gradient
theta(1) = 0; % set local theta(1) to 0 to avoid regularization of theta 0
J = (1/m) * ((-y' * log(h)) - ((1 - y)' * log(1 - h))) + ((lambda/(2*m)) * (theta' * theta));
grad = ((1/m) * (X' * (h - y))) + ((lambda/m) * theta);


end
