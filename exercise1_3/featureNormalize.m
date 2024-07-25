function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.


% Get the number of samples and features
[nSamples, nFeat] = size(X);

% Initialize output variables --- Preallocation
X_norm = zeros(nSamples, nFeat);
mu = zeros(1, nFeat);
sigma = zeros(1, nFeat);

% Normalize each feature
for j = 1:nFeat
    mu(j) = mean(X(:, j));
    sigma(j) = std(X(:, j));
    X_norm(:, j) = (X(:, j) - mu(j)) / sigma(j);
end


% ============================================================

end
