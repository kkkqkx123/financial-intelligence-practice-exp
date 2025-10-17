import numpy as np
from gen_data import gen_data
from plot import plot
from todo import func, logistic_regression, perceptron
import matplotlib.pyplot as plt
import os

no_iter = 1000  # number of iteration
no_train = 70  # number of training data (70% for training)
no_test = 30   # number of testing data (30% for testing)
no_data = 100  # number of all data
assert(no_train + no_test == no_data)

def compute_error(X, y, w):
    """Compute classification error rate"""
    X_bias = np.vstack((np.ones((1, X.shape[1])), X))
    predictions = np.sign(w.T @ X_bias)
    # Handle case where prediction is 0
    predictions[predictions == 0] = 1
    errors = np.sum(predictions.flatten() != y.flatten())
    return errors / y.shape[1]

def run_classification_experiment(algorithm_name, algorithm_func):
    """Run classification experiment with specified algorithm"""
    print(f"\n=== Running {algorithm_name} ===")
    
    cumulative_train_err = 0
    cumulative_test_err = 0
    
    for i in range(no_iter):
        X, y, w_gt = gen_data(no_data)
        X_train, X_test = X[:, :no_train], X[:, no_train:]
        y_train, y_test = y[:, :no_train], y[:, no_train:]
        
        # Learn parameters
        w_l = algorithm_func(X_train, y_train)
        
        # Compute training and testing error
        train_err = compute_error(X_train, y_train, w_l)
        test_err = compute_error(X_test, y_test, w_l)
        
        cumulative_train_err += train_err
        cumulative_test_err += test_err
    
    avg_train_err = cumulative_train_err / no_iter
    avg_test_err = cumulative_test_err / no_iter
    
    print(f"{algorithm_name} Results:")
    print(f"Average Training error: {avg_train_err:.4f}")
    print(f"Average Testing error: {avg_test_err:.4f}")
    
    return avg_train_err, avg_test_err

# Run experiments with different algorithms
results = {}

# SVM (original func)
results['SVM'] = run_classification_experiment('Support Vector Machine', func)

# Logistic Regression
results['Logistic Regression'] = run_classification_experiment('Logistic Regression', logistic_regression)

# Perceptron
results['Perceptron'] = run_classification_experiment('Perceptron', perceptron)

# Compare results
print("\n=== Algorithm Comparison ===")
print("Algorithm\t\tTrain Error\tTest Error")
print("-" * 50)
for algo, (train_err, test_err) in results.items():
    print(f"{algo:<20}\t{train_err:.4f}\t\t{test_err:.4f}")

# Save results to file
results_dir = "classification_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

with open(os.path.join(results_dir, "classification_comparison.txt"), "w") as f:
    f.write("Classification Algorithm Comparison Results\n")
    f.write("=" * 50 + "\n\n")
    for algo, (train_err, test_err) in results.items():
        f.write(f"{algo}:\n")
        f.write(f"  Training Error: {train_err:.4f}\n")
        f.write(f"  Testing Error: {test_err:.4f}\n\n")

print(f"\nResults saved to {results_dir}/classification_comparison.txt")

# Generate a final visualization with one of the algorithms
X, y, w_gt = gen_data(no_data)
X_train, X_test = X[:, :no_train], X[:, no_train:]
y_train, y_test = y[:, :no_train], y[:, no_train:]
w_l = func(X_train, y_train)

plot(X, y, w_gt, w_l, "SVM Classification Results")