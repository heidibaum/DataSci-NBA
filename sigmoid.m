function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% initialize output
g = zeros(size(z));

% compute sigmoid function on all elements of z
g = 1 ./ (1 + (e.^(-z)));

end
