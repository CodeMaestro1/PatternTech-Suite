###### THL311 - HW2 ##############
## Complete the missing code to implement Logistic Regression
## First make sure you have installed necessary modules
## pip install numpy matplotlib scipy


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def plotData(X, y):
    pos = y == 1
    neg = y == 0
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k', label='Admitted')
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', c='y', label='Not admitted')

def sigmoid(z):
    ### ADD YOUR CODE HERE
    
    sigmoid_function = 1 / (1 + 2*np.exp(-z))
    return sigmoid_function

# Calculate the cost function
def costFunction(theta, X, y):
    ### ADD YOUR CODE HERE
    m = len(y)
    h = sigmoid(X @ theta)
    J = (1/m) * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))
    return J

# Calculate the gradient of the cost function
def gradient(theta, X, y):
    ### ADD YOUR CODE HERE
    m = len(y)
    h = sigmoid(X @ theta)
    error = h - y
    grad = (1/m) * (X.T @ error)
    return grad

def plotDecisionBoundary(theta, X, y):
    plotData(X[:, 1:3], y)
    x_value = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
    y_value = -(theta[0] + theta[1] * x_value) / theta[2]
    plt.plot(x_value, y_value, label='Decision Boundary')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()

# Class Prediction
def predict(theta, X):
    ### ADD YOUR CODE HERE
    prob = sigmoid(X @ theta)
    return (prob >= 0.5).astype(int)

# Initialization
np.set_printoptions(suppress=True)

# Load Data
data = np.loadtxt('exam_scores_data1.txt', delimiter=',')
X = data[:, [0, 1]]
y = data[:, 2]

# ==================== Part 1: Plotting ====================
print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')
plotData(X, y)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()
input('\nProgram paused. Press enter to continue.\n')

# ============ Part 2: Compute Cost and Gradient ============
m, n = X.shape
X = np.concatenate([np.ones((m, 1)), X], axis=1)
initial_theta = np.zeros(n + 1)
cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)
print('Cost at initial theta (zeros):', cost)
print('Gradient at initial theta (zeros):', grad)
input('\nProgram paused. Press enter to continue.\n')

# ============= Part 3: Optimizing using minimize  =============
options = {'gtol': 1e-6}  # Set tolerance to 1e-6
res = minimize(fun=costFunction, x0=initial_theta, args=(X, y), jac=gradient, method='TNC', options=options)

theta = res.x
cost = res.fun
print('Cost at theta found by minimize:', cost)
print('theta:', theta)
plotDecisionBoundary(theta, X, y)
plt.show()
input('\nProgram paused. Press enter to continue.\n')

# ============== Part 4: Predict and Accuracies ==============
# Check the result for a student with marks 45 and 85.
### ADD YOUR CODE HERE

X_test = np.array([1, 45, 85])
prob = sigmoid(np.dot(X_test, theta))
print('For a student with scores 45 and 85, we predict an admission probability of {:.4f}'.format(prob))
p = predict(theta, X)

### ADD YOUR CODE HERE
train_accuracy = np.mean(p == y) * 100
print('Train Accuracy:', train_accuracy)

