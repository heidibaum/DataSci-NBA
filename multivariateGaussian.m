function p = multivariateGaussian(X, mu, sigma2)
%MULTIVARIATEGAUSSIAN Computes the probability density function of the
%multivariate gaussian distribution.
%    p = MULTIVARIATEGAUSSIAN(X, mu, Sigma2) Computes the probability 
%    density function of the examples X under the multivariate gaussian 
%    distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
%    treated as the covariance matrix. If Sigma2 is a vector, it is treated
%    as the \sigma^2 values of the variances in each dimension (a diagonal
%    covariance matrix)
%

k = length(mu);

if (size(sigma2, 2) == 1) || (size(sigma2, 1) == 1)
    sigma2 = diag(sigma2);
end

X_norm = bsxfun(@minus, X, mu(:)');
p = (2 * pi) ^ (- k / 2) * det(sigma2) ^ (-0.5) * ...
    exp(-0.5 * sum(bsxfun(@times, X_norm * pinv(sigma2), X_norm), 2));

end