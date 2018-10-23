function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% compute means for each var
mu = mean(X);
% replace each x with x-mu
X_norm = bsxfun(@minus, X, mu);

% compute stdevs for each var
sigma = std(X_norm);
% replace each x_norm with x_norm/sigma
X_norm = bsxfun(@rdivide, X_norm, sigma);


end
