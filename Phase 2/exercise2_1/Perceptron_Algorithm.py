import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
import altair as alt
from IPython.display import display

def normalize_features(X):
    '''Normalize features to zero mean and unit variance'''
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

def plot_samples(data_list, w=None):
    '''Plotting the samples'''
    plt.figure()
    for data in data_list:
        X, y = data
        for label in np.unique(y):
            plt.scatter(X[y == label, 0], X[y == label, 1], label=f'Class {int(label)}')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    if w is not None:
        # Plot decision boundary
        x_min, x_max = plt.xlim()
        xx = np.linspace(x_min, x_max, 100)
        yy = -(w[0] + w[1] * xx) / w[2]
        plt.plot(xx, yy, 'k--')
    
    plt.show()

def zero_weights(n_features):
    '''Create vector of zero weights'''
    return np.zeros(1 + n_features)  # Add 1 for the bias term

def net_input(X, w):
    '''Compute net input as dot product'''
    return np.dot(X, w[1:]) + w[0]

def predict(X, w):
    '''Return class label after unit step'''
    return np.where(net_input(X, w) >= 0.0, 1, -1)

def fit_batch(X, y, eta=0.05, n_iter=100):
    '''Batch form of the Perceptron algorithm'''
    iterations = 0
    errors = []
    w = zero_weights(X.shape[1])
    for _ in range(n_iter):
        output = net_input(X, w)
        errors_vector = y - np.where(output >= 0.0, 1, -1)
        if np.all(errors_vector == 0):
            break
        delta_w = eta * np.dot(errors_vector, X)
        w[1:] += delta_w
        w[0] += eta * errors_vector.sum()
        error = np.count_nonzero(errors_vector)
        errors.append(error)
        iterations += 1
    return w, errors, iterations

def main():
    # Read data from CSV
    data = pd.read_csv(r'C:\Users\mypc1\Desktop\Thema_1\Thema1_Pattern.csv')

    # Extract features and labels for each class
    X_omega1 = data[data['class'] == 'omega1'][['x1', 'x2']].values
    X_omega2 = data[data['class'] == 'omega2'][['x1', 'x2']].values
    X_omega3 = data[data['class'] == 'omega3'][['x1', 'x2']].values
    X_omega4 = data[data['class'] == 'omega4'][['x1', 'x2']].values

    y_omega1 = np.ones(len(X_omega1))  # Class 1
    y_omega2 = np.ones(len(X_omega2)) * 2  # Class 2
    y_omega3 = np.ones(len(X_omega3)) * 3  # Class 3
    y_omega4 = np.ones(len(X_omega4)) * 4  # Class 4
    
    # Combine all classes for normalization
    X_combined = np.vstack((X_omega1, X_omega2, X_omega3, X_omega4))
    X_combined_normalized = normalize_features(X_combined)

    # Split normalized features back to classes
    X_omega1 = X_combined_normalized[:len(X_omega1)]
    X_omega2 = X_combined_normalized[len(X_omega1):len(X_omega1)+len(X_omega2)]
    X_omega3 = X_combined_normalized[len(X_omega1)+len(X_omega2):len(X_omega1)+len(X_omega2)+len(X_omega3)]
    X_omega4 = X_combined_normalized[len(X_omega1)+len(X_omega2)+len(X_omega3):]

    plot_samples([(X_omega1, y_omega1), (X_omega2, y_omega2), (X_omega3, y_omega3), (X_omega4, y_omega4)])

    # Define class pairs for classification
    class_pairs = [
        (X_omega1, y_omega1, X_omega2, y_omega2),
        (X_omega2, y_omega2, X_omega3, y_omega3),
        (X_omega3, y_omega3, X_omega4, y_omega4),
    ]

    for i, (X1, y1, X2, y2) in enumerate(class_pairs, start=1):
        # Combine features and labels
        X = np.vstack((X1, X2))
        y = np.hstack((y1, y2))
        
        # Convert labels to binary (-1, 1)
        y = np.where(y == y1[0], 1, -1)

        # Fit the model
        w, errors, iterations = fit_batch(X, y)

        # Accuracy
        y_pred = predict(X, w)
        accuracy = accuracy_score(y, y_pred) * 100
        print(f'Pair {i}: Classes {int(y1[0])} vs {int(y2[0])}')
        print(f'Accuracy: {accuracy}%')
        print(f'Iterations: {iterations}\n')

        # Plot the samples and decision boundary
        plot_samples([(X1, y1), (X2, y2)], w)

        # Plot errors over time
        error_df = pd.DataFrame({'error': errors, 'time-step': np.arange(0, len(errors))})
        chart = alt.Chart(error_df).mark_line().encode(
            x="time-step", y="error"
        ).properties(
            title=f'Errors over time for Pair {i}'
        )
        display(chart)
        chart.save(f'chart_{i}.html')

if __name__ == "__main__":
    main()
