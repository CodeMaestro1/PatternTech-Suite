function [X_rec] = recoverDataLDA(Z, v)

% You need to return the following variables correctly.
X_rec = zeros(size(Z, 1), length(v));

% ====================== YOUR CODE HERE ======================

% Z is m x 1, and v is n x 1, we perform outer product to get m x n matrix
% where each row is a reconstruction of the original data point
for i = 1:size(Z, 1)
    X_rec(i, :) = Z(i) * v';  % Outer product to scale v by the scalar Z(i)
end


% =============================================================

end
