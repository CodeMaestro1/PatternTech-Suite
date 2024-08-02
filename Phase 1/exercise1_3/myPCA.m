function [U, S] = myPCA(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = principalComponentAnalysis(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% ====================== YOUR CODE GOES HERE ======================
% Instructions: You should first compute the covariance matrix. Then, 
%  compute the eigenvectors and eigenvalues of the covariance matrix. 
%
% Note that the dataset X is normalized, when calculating the covariance

% Make sure each feature from the data is zero mean
%X_centered = X - mean(X);

% ====================== YOUR CODE HERE ======================

% Compute the covariance matrix
Sigma = (1 / m) * (X' * X);


% Compute eigenvalues and eigenvectors
[V, D] = eig(Sigma);

% Extract the diagonal of D as a vector
eigenval = diag(D);

% Sort the eigenvalues in descending order and get the order
[eigenval,order]=sort(eigenval,1,'descend'); %Sort them

% Reorder the eigenvectors according to the sorted eigenvalues
eigenvec=V(:,order); %Corresponding eigenvectors

U = eigenvec;
S = eigenval; 



% =========================================================================

end
