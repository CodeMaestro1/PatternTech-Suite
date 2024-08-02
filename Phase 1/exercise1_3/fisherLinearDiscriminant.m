function v = fisherLinearDiscriminant(X1, X2)

    m1 = size(X1, 1);
    m2 = size(X2, 1);

    mu1 = mean(X1); % mean value of X1
    mu2 = mean(X2); % mean value of X2

    % Calculate scatter matrix for X1
    S1 = cov(X1); 
    % Calculate scatter matrix for X2
    S2 = cov(X2); 

    % Within class scatter matrix
    Sw = S1 + S2;  

    % Calculate the difference between the mean vectors
    meanDifference = (mu1 - mu2)'; 

    % Calculate the inverse of the within-class scatter matrix
    Sw_inv = inv(Sw);

    % Optimal direction for maximum class separation 
    v = Sw_inv * meanDifference; 

    % Return a vector of unit norm
    v = v / norm(v); 
end

