function [ eigenval, eigenvec, order] = myPCA(X)
%PCA Run principal component analysis on the dataset X
%   [ eigenval, eigenvec, order] = mypca(X) computes eigenvectors of the autocorrelation matrix of X
%   Returns the eigenvectors, the eigenvalues (on diagonal) and the order 
%

% Useful values
[m, n] = size(X);
eigenvec = zeros(m);
eigenval = zeros(n);

% Make sure each feature from the data is zero mean
X_centered = X - mean(X);

% ====================== YOUR CODE HERE ======================

% Compute the covariance matrix
Sigma = (1/(m)) * (X_centered' * X_centered);


% Compute eigenvalues and eigenvectors
[V, D] = eig(Sigma);

% Extract the diagonal of D as a vector
eigenval = diag(D);

% Sort the eigenvalues in descending order and get the order
[eigenval,order]=sort(eigenval,1,'descend'); %Sort them

% Reorder the eigenvectors according to the sorted eigenvalues
eigenvec=V(:,order); %Corresponding eigenvectors
    
% =========================================================================

end
