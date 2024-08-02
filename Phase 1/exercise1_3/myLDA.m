function A = myLDA(Samples, Labels, NewDim)
% Input:    
%   Samples: The Data Samples 
%   Labels: The labels that correspond to the Samples
%   NewDim: The New Dimension of the Feature Vector after applying LDA

	%A=zeros(NumFeatures,NewDim);
    
	[NumSamples, NumFeatures] = size(Samples);
    NumLabels = length(Labels);
    if(NumSamples ~= NumLabels) then
        fprintf('\nNumber of Samples are not the same with the Number of Labels.\n\n');
        exit
    end
    Classes = unique(Labels);
    NumClasses = length(Classes);  %The number of classes

    % Initialize matrices
    Sw = zeros(NumFeatures, NumFeatures); % Within-class scatter matrix
    m0 = mean(Samples, 1); % Global mean
    Sb = zeros(NumFeatures, NumFeatures); % Between-class scatter matrix

    %For each class i
	%Find the necessary statistics
    for i = 1:NumClasses
        % Class-specific samples and statistics
        classSamples = Samples(Labels == Classes(i), :);
        mu(i, :) = mean(classSamples, 1); % %Calculate the Class Mean
        P(i) = size(classSamples, 1) / NumSamples; %Calculate the Class Prior Probability
        
        % Within-class scatter -- Calculate the Within Class Scatter Matrix
        classScatter = classSamples - mu(i, :); % Deviation from mean
        Sw = Sw + (classScatter' * classScatter); % Sum of squares

        % Between-class scatter -- %Calculate the Between Class Scatter Matrix
        meanDiff = (mu(i, :) - m0)';
        Sb = Sb + P(i) * (meanDiff * meanDiff');
    end
    
    %Eigen matrix EigMat=inv(Sw)*Sb
    EigMat = inv(Sw)*Sb;
    
    %Perform Eigendecomposition
    [eigenvectors, eigenvalues] = eig(EigMat);

    % Extract eigenvalues as a vector
    eigenvalues = diag(eigenvalues);

    % Sort eigenvalues (and vectors) in descending order
    [sortedEigenvalues, sortOrder] = sort(eigenvalues, 'descend');
    sortedEigenvectors = eigenvectors(:, sortOrder);

    
	%% You need to return the following variable correctly.
	% Select the NewDim eigenvectors corresponding to the top NewDim eigenvalues
    % Assuming they are NewDim <= NumClasses - 1
    A = sortedEigenvectors(:, 1:NewDim);  % Return the LDA projection vectors
