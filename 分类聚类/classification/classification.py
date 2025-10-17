import numpy as np
from gen_data import gen_data
from plot import plot
from todo import func

no_iter = 1000  # number of iteration
no_train = # YOUR CODE HERE # number of training data
no_test = # YOUR CODE HERE  # number of testing data
no_data = 100  # number of all data
assert(no_train + no_test == no_data)

cumulative_train_err = 0
cumulative_test_err = 0

for i in range(no_iter):
    X, y, w_gt = gen_data(no_data)
    X_train, X_test = X[:, :no_train], X[:, no_train:]
    y_train, y_test = y[:, :no_train], y[:, no_train:]
    w_l = func(X_train, y_train)
    # Compute training, testing error
    # YOUR CODE HERE
    # ----------------
    # train_err = xxx
    # test_err = xxx
    pass
    # ----------------
    cumulative_train_err += train_err
    cumulative_test_err += test_err

train_err = cumulative_train_err / no_iter
test_err = cumulative_test_err / no_iter

plot(X, y, w_gt, w_l, "Classification")
print("Training error: %s" % train_err)
print("Testing error: %s" % test_err)