import numpy as np

# You may find below useful for Support Vector Machine
# More details in
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
# from scipy.optimize import minimize

def func(X, y):
    '''
    Classification algorithm.

    Input:  X: Training sample features, P-by-N
            y: Training sample labels, 1-by-N

    Output: w: learned parameters, (P+1)-by-1
    '''
    P, N = X.shape
    w = np.zeros((P+1, 1))

    # YOUR CODE HERE
    # ----------------
    pass
    # ----------------
    return w