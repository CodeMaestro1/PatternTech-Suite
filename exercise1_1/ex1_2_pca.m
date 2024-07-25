clearvars
clear all;
clc;
close all;

% Pattern Recognition
%  Exercise | Principle Component Analysis
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     myPCA.m
%     projectData.m
%     recoverData.m
%
%  Add code where needed.
%

%% Initialization
clear all; close all; clc

%% ================== Part 1: Load Example Dataset  ===================
%  We start this exercise by using a small dataset that is easily to
%  visualize
%
fprintf('Visualizing example dataset for PCA.\n\n');

%  The following command loads the breast cancert dataset. You should now have the 
%  variable X in your environment
Data=csvread('data/breast_cancer_data.csv');
NSamples = 100; %Get the first NSamples only for better visualization
X=Data(1:NSamples,1:end-1); % Get all features
Y=Data(1:NSamples,end);

% Visualize the example dataset in 2D using sample feature pairs
% Repeat using different feature pairs on the 2D space 
figure(1)
plot(X(Y==0, 1), X(Y==0, 2), 'bo',X(Y==1, 1), X(Y==1, 2), 'ro' );
axis square;
title('Data Samples in 2D')
xlabel('Feature 1');
ylabel('Feature 2');
hold on

fprintf('Program paused. Press enter to continue.\n');

fprintf('Data Samples in 2D')
% Define the feature pairs you want to plot
featurePairs = [3 4; 17 10; 9 12; 1 12; 1 2];  % Each row defines a pair of features to plot
numPairs = size(featurePairs, 1);

% Compute covariance matrix and mean vector
C = cov(X);
mu = mean(X);

for i = 1:numPairs
    % Create a new figure for each pair
    figure;

    % Plot the ith pair of features
    plot(X(Y==0, featurePairs(i, 1)), X(Y==0, featurePairs(i, 2)), 'bo', ...
         X(Y==1, featurePairs(i, 1)), X(Y==1, featurePairs(i, 2)), 'ro');

    % Compute Mahalanobis distance for each observation
    numObs = size(X, 1);  % Number of observations
    D = zeros(numObs, 1);  % Initialize array to store distances
    for j = 1:numObs
        x = X(j, featurePairs(i, :));  % Extract the features for the j-th observation
        D(j) = sqrt((x - mu(featurePairs(i, :))) * inv(C(featurePairs(i, :), featurePairs(i, :))) * (x - mu(featurePairs(i, :)))');
    end

    % Set plot properties
    axis square;
    xlabel(sprintf('Feature %d', featurePairs(i, 1)));
    ylabel(sprintf('Feature %d', featurePairs(i, 2)));
    title(sprintf('Feature %d vs Feature %d (Mahalanobis Distance)', featurePairs(i, 1), featurePairs(i, 2)));

    % Display Mahalanobis distance in the title
    distanceMean = mean(D);
    titleText = sprintf('Feature %d vs Feature %d (Mean Mahalanobis Distance: %.2f)', featurePairs(i, 1), featurePairs(i, 2), distanceMean);
    title(titleText);
end

pause




%% =============== Part 2: Principal Component Analysis ===============
%  You should now implement PCA, a feature extraction algorithm. 
%  Complete the code in myPCA.m
%  Consider only 2D samples by taking the first two features of the dataset

fprintf('\nRunning PCA on example dataset taking the first 2 features.\n\n');
X=Data(1:NSamples,1:2); % Get 2 first features

% Before running PCA, it is important to first normalize X
% Add the necessary code to perform standardization in featureNormalize
[X_norm, mu, sigma] = featureNormalize(X);

%  Run PCA
[eigvals, eigvecs, order] = myPCA(X_norm);

% Plot the samples on the first two principal components
figure;
plot(X_norm(Y==0, 1), X_norm(Y==0, 2), 'bo',X_norm(Y==1, 1), X_norm(Y==1, 2), 'ro' );
hold on
% Plot the principal component vectors (arrows)
scale = 3; % Scale factor for the arrows
quiver(0, 0, eigvecs(1,1)*scale*eigvals(1), eigvecs(2,1)*scale*eigvals(1), 'r', 'LineWidth', 2, 'MaxHeadSize', 1);
quiver(0, 0, eigvecs(1,2)*scale*eigvals(2), eigvecs(2,2)*scale*eigvals(2), 'b', 'LineWidth', 2, 'MaxHeadSize', 1);
title('Normalized Samples and principal components')
xlabel('Normalized feature 1')
ylabel('Normalized feature 2')
hold off

pause; 

% Projection of the data onto the principal components
% ADD YOUR CODE

% Define K if not already defined
K = 2; % Projecting onto the first 2 principal components

% Project data
X_PCA = X_norm * eigvecs(:, 1:K);


% Plot the projection of the data onto the principal components
figure;
hold on
plot(X_PCA(Y==0, 1), X_PCA(Y==0, 2), 'bo','MarkerFaceColor', 'b', 'MarkerEdgeColor', 'none', 'MarkerSize', 4 )
plot(X_PCA(Y==1, 1), X_PCA(Y==1, 2), 'ro','MarkerFaceColor', 'r', 'MarkerEdgeColor', 'none', 'MarkerSize', 4 );
title('PCA of Breast Cancer Dataset');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
axis square;
hold off;

% Calculate the Explained Variance
% ADD YOUR CODE
ExplainedVar = eigvals/sum(eigvals);
fprintf(' Explained Variance(1st PC = %f) (2nd PC = %f)\n', ExplainedVar(1), ExplainedVar(2));

% Print the first principal component
fprintf(' 1st Principal Component = %f %f \n', eigvecs(1,1), eigvecs(2,1));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =================== Part 3: Dimensionality Reduction ===================
%  Perform dimensionality reduction by projecting the data onto the 
%  first principal component. Then plot the data in this reduced in 
%  dimension space.  This will show you what the data looks like when 
%  using only the corresponding eigenvectors to reconstruct it.
%
%  You should complete the code in projectData.m
%
fprintf('\nDimensionality reduction on example dataset.\n\n');

%  Project the data onto K = 1 dimension
K = 1;
Z = projectData(X_norm, eigvecs, K);
fprintf('Projection of the first example: %f\n', Z(1));

%  Use the K principal components to recover the Data in 2D
X_recover  = recoverData(Z, eigvecs, K);
fprintf('Approximation of the first example: %f %f\n', X_recover(1, 1), X_recover(1, 2));

%  Draw the lines connecting the projected points to the original points
figure;
hold on;
plot(X_recover(Y==0, 1), X_recover(Y==0, 2), 'co','MarkerFaceColor', 'c', 'MarkerEdgeColor', 'none', 'MarkerSize', 5);
plot(X_recover(Y==1, 1), X_recover(Y==1, 2), 'yo', 'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'none', 'MarkerSize', 5);

plot(X_norm(Y==0, 1), X_norm(Y==0, 2), 'bo','MarkerFaceColor', 'b', 'MarkerEdgeColor', 'none', 'MarkerSize', 5 )
plot(X_norm(Y==1, 1), X_norm(Y==1, 2), 'ro','MarkerFaceColor', 'r', 'MarkerEdgeColor', 'none', 'MarkerSize', 5 );
title('PCA sample projections');

axis square
for i = 1:size(X_norm, 1)
    drawLine(X_norm(i,:), X_recover(i,:), '--k', 'LineWidth', 1);
end
axis([-3 3.5 -3 3.5]); 
axis square
hold off

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =============== Part 4: Perform PCA on all features in the dataset =============
%  Apply PCA on the initial dataset and then choose the first 2 Principal Componets
%  to reduce the dimensionality of the dataset to the 2D space

%  The following code will load the dataset into your environment
%

X=Data(1:NSamples,1:end-1); % Get all features from the breast cancer dataset

% Before running PCA, it is important to first normalize X
% Add the necessary code to perform standardization in featureNormalize
[X_norm, mu, sigma] = featureNormalize(X);

%  Run PCA
[eigvals, eigvecs, order] = myPCA(X_norm);

figure
hold on
% Plot the samples on the first two principal components
plot(X_norm(Y==0, order(1)), X_norm(Y==0, order(2)), 'bo','MarkerFaceColor', 'b', 'MarkerEdgeColor', 'none', 'MarkerSize', 5 )
plot(X_norm(Y==1, order(1)), X_norm(Y==1, order(2)), 'ro','MarkerFaceColor', 'r', 'MarkerEdgeColor', 'none', 'MarkerSize', 5 );
title('PCA sample projections');
% Plot the principal component vectors (arrows)
scale = 1; % Scale factor for the arrows
quiver(0, 0, eigvecs(1,1)*scale*eigvals(1), eigvecs(2,1)*scale*eigvals(1), 'r', 'LineWidth', 2, 'MaxHeadSize', 1);
quiver(0, 0, eigvecs(1,2)*scale*eigvals(2), eigvecs(2,2)*scale*eigvals(2), 'b', 'LineWidth', 2, 'MaxHeadSize', 1);
hold off;
axis square;

% Projection of the data onto the principal components
% ADD YOUR CODE
X_PCA = X_norm * eigvecs(:, 1:2);  % Using the first two principal components

pause
% Plot the samples on the first two principal components
figure;
scatter(X_PCA(:,1), X_PCA(:,2), 30, Y, 'filled');
title('Breast Cancer Dataset - PCA reduced in 2D');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
hold off;



% Calculate the Explained Variance of the first 2 Principal Components
% ADD YOUR CODE
totalVariance = sum(eigvals);  % Total variance from all components
ExplainedVar = eigvals(1:2) / totalVariance;  % Proportion of total variance explained by the first two PCs
fprintf(' Explained Variance(1st PC = %f) (2nd PC = %f)\n', ExplainedVar(1), ExplainedVar(2));


% Print the first principal component
fprintf(' 1st Principal Component = %f %f \n', eigvecs(1,1), eigvecs(2,1));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =============== Part 5: Loading and Visualizing Face Data =============
%  We start the exercise by first loading and visualizing the dataset.
%  The following code will load the dataset into your environment
%
fprintf('\nLoading face dataset.\n\n');

%  Load Face dataset
load ('data/faces.mat')

%  Display the first 100 faces in the dataset
displayData(X(1:100, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 6: PCA on Face Data: Eigenfaces  ===================
%  Run PCA and visualize the eigenvectors which are in this case eigenfaces
%  We display the first 36 eigenfaces.
%
fprintf(['\nRunning PCA on face dataset.\n' ...
         '(this mght take a minute or two ...)\n\n']);

%  Before running PCA, it is important to first normalize X by subtracting 
%  the mean value from each feature
[X_norm, mu, sigma] = featureNormalize(X);

%  Run PCA
[eigvals, eigvecs, order] = myPCA(X_norm);

%  Visualize the top 36 eigenvectors found (eigenfaces)
displayData(eigvecs(:, 1:36)');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ============= Part 7: Dimensionality Reduction for Faces =================
%  Project images to the eigen space using the top k eigenvectors 
%  If you are applying a machine learning algorithm 
fprintf('\nDimension reduction for face dataset.\n\n');

K = 100;
Z = projectData(X_norm, eigvecs, K);

fprintf('The projected data Z has a size of: ')
fprintf('%d ', size(Z));

fprintf('\n\nProgram paused. Press enter to continue.\n');
pause;

%% ==== Part 8: Visualization of Faces after PCA Dimensionality Reduction ====
%  Project images to the eigen space using the top K eigen vectors and 
%  visualize only using those K dimensions
%  Compare to the original input, which is also displayed

fprintf('\nVisualizing the projected (reduced dimension) faces.\n\n');

K = 100;
X_rec  = recoverData(Z, eigvecs, K);

% Display normalized data
subplot(1, 2, 1);
displayData(X_norm(1:100,:));
title('Original faces');
axis square;

% Display reconstructed data from only k eigenfaces
subplot(1, 2, 2);
displayData(X_rec(1:100,:));
title('Recovered faces');
axis square;