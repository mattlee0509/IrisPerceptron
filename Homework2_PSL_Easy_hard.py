import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_dataset(file_path):
    """Load dataset and prepare it for training."""
    df = pd.read_csv(file_path)
    # Assuming the dataset structure aligns with the description
    X = df[['x1', 'x2']].values
    y = df['Y'].values * 2 - 1  # Convert to {-1, 1} format
    return X, y

def hinge_loss_gradient(w, X, y, lambd):
    """Compute hinge loss and its gradient with L2 regularization."""
    n = X.shape[0]
    distances = 1 - y * np.dot(X, w)
    hinge_loss = np.where(distances > 0, distances, 0).mean() + lambd * np.dot(w, w)
    
    derivative = np.zeros(len(w))
    for i, (dist, xi, yi) in enumerate(zip(distances, X, y)):
        if dist > 0:
            derivative -= yi * xi
    gradient = derivative / n + 2 * lambd * w
    return hinge_loss, gradient

def gradient_descent(X, y, eta, lambd, max_iter=10000, tol=1e-4):
    """Perform gradient descent to minimize the regularized hinge loss."""
    w = np.zeros(X.shape[1])
    for i in range(max_iter):
        loss, grad = hinge_loss_gradient(w, X, y, lambd)
        w_new = w - eta * grad
        # Check for convergence
        if np.linalg.norm(w - w_new) < tol:
            break
        w = w_new
    return w, i+1

def find_best_hyperplane(file_path, etas, lambdas):
    """Find the best hyperplane for given combinations of eta and lambda."""
    X, y = load_dataset(file_path)
    X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])  # Add intercept term
    
    best_iterations = np.inf
    best_eta = None
    best_lambda = None
    best_w = None
    
    for eta in etas:
        for lambd in lambdas:
            w, iterations = gradient_descent(X_with_intercept, y, eta, lambd)
            if iterations < best_iterations:
                best_iterations = iterations
                best_eta, best_lambda = eta, lambd
                best_w = w
    
    # Plotting the best hyperplane
    x_values = np.linspace(X[:,0].min(), X[:,0].max(), 2)
    y_values = -(best_w[0] + best_w[1]*x_values)/best_w[2]  # w0 + w1*x1 + w2*x2 = 0
    plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm')
    plt.plot(x_values, y_values, label=f'Best Hyperplane: η={best_eta}, λ={best_lambda}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()
    
    print(f"Best combination: η={best_eta}, λ={best_lambda}, Iterations until convergence: {best_iterations}")
def calculate_margin(w, X, y):
    """Calculate the margin of the hyperplane defined by weights w."""
    distances = np.abs(np.dot(X, w)) / np.linalg.norm(w[1:])  # Exclude intercept for norm calculation
    margin = np.min(distances)
    return margin

def analyze_dataset_with_margin(name, X, y, etas, lambdas):
    results = {}
    margins = {}
    for eta in etas:
        for lambd in lambdas:
            w, iterations = gradient_descent(X, y, eta, lambd)
            margin = calculate_margin(w, X, y)
            results[(eta, lambd)] = iterations
            margins[(eta, lambd)] = margin
            print(f"{name} - η: {eta}, λ: {lambd}, Iterations: {iterations}, Margin: {margin:.4f}")
    return results, margins

# Re-analyze both datasets with margin calculations
# Parameters
etas = [0.01, 0.05, 0.1, 0.5, 1]
lambdas = [0, 0.01, 1, 100]

# Adjust file paths as necessary
find_best_hyperplane(r'C:\Users\cools\Downloads\easy.csv', etas, lambdas)
find_best_hyperplane(r'C:\Users\cools\Downloads\hard.csv', etas, lambdas)
print("Analyzing Easy Dataset with Margins")
results_easy_with_margin, margins_easy = analyze_dataset_with_margin("Easy", X_easy, y_easy, etas, lambdas)
print("\nAnalyzing Hard Dataset with Margins")
results_hard_with_margin, margins_hard = analyze_dataset_with_margin("Hard", X_hard, y_hard, etas, lambdas)

