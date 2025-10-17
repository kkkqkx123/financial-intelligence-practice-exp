import numpy as np
from scipy.optimize import minimize

def func(X, y):
    '''
    Classification algorithm - Support Vector Machine (SVM).

    Input:  X: Training sample features, P-by-N
            y: Training sample labels, 1-by-N

    Output: w: learned parameters, (P+1)-by-1
    '''
    P, N = X.shape
    
    # Add bias term to X
    X_bias = np.vstack((np.ones((1, N)), X))
    
    # Convert y to ensure labels are -1 and 1
    y = y.flatten()
    
    # Define the objective function for SVM
    def objective(w):
        w = w.reshape(-1, 1)
        # Hinge loss + L2 regularization
        margins = y * (w.T @ X_bias).flatten()
        hinge_loss = np.maximum(0, 1 - margins)
        return 0.5 * np.sum(w[1:]**2) + np.sum(hinge_loss)
    
    # Initial guess
    w_init = np.zeros(P + 1)
    
    # Optimize
    result = minimize(objective, w_init, method='BFGS')
    
    return result.x.reshape(-1, 1)

def logistic_regression(X, y):
    '''
    Logistic Regression classifier.

    Input:  X: Training sample features, P-by-N
            y: Training sample labels, 1-by-N

    Output: w: learned parameters, (P+1)-by-1
    '''
    P, N = X.shape
    
    # Add bias term to X
    X_bias = np.vstack((np.ones((1, N)), X))
    
    # Convert y to 0 and 1 for logistic regression
    y_binary = (y.flatten() + 1) // 2
    
    # Define the objective function for logistic regression
    def objective(w):
        w = w.reshape(-1, 1)
        z = w.T @ X_bias
        # Prevent overflow
        z = np.clip(z, -500, 500)
        predictions = 1 / (1 + np.exp(-z.flatten()))
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.mean(y_binary * np.log(predictions) + (1 - y_binary) * np.log(1 - predictions))
        return loss
    
    # Initial guess
    w_init = np.zeros(P + 1)
    
    # Optimize
    result = minimize(objective, w_init, method='BFGS')
    
    return result.x.reshape(-1, 1)

def perceptron(X, y):
    '''
    Perceptron classifier.

    Input:  X: Training sample features, P-by-N
            y: Training sample labels, 1-by-N

    Output: w: learned parameters, (P+1)-by-1
    '''
    P, N = X.shape
    
    # Add bias term to X
    X_bias = np.vstack((np.ones((1, N)), X))
    
    # Initialize weights
    w = np.zeros((P + 1, 1))
    y = y.flatten()
    
    # Perceptron learning algorithm
    max_iterations = 1000
    learning_rate = 0.1
    
    for iteration in range(max_iterations):
        converged = True
        for i in range(N):
            prediction = np.sign(w.T @ X_bias[:, i])
            if prediction == 0:
                prediction = 1
            
            if prediction != y[i]:
                w += learning_rate * y[i] * X_bias[:, i:i+1]
                converged = False
        
        if converged:
            break
    
    return w