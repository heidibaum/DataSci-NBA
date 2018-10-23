function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%


[m, n] = size(X);

% initialize variables 
U = zeros(n);
S = zeros(n);


% compute covariance matrix
Sigma = (1/m) * X' * X;

% compute eigenvectors and eigenvalues of covariance matrix
[U,S,V] = svd(Sigma);


end
