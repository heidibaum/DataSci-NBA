function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% get dimensions of X
[m, n] = size(X);

% initialize output
mu = zeros(n, 1);
sigma2 = zeros(n, 1);


% compute mean and variances of all features
mu = mean(X);
difference = bsxfun(@minus, X, mu);
sigma2 = (1/m) * sum(difference.^2);

% transpose into column vectors
mu = mu';
sigma2 = sigma2';

end
