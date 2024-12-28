import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(file_path):
    df = pd.read_csv(file_path)
    
    df['Y'] = df['Y'].replace({0: -1, 1: 1})
    X = df[['x1', 'x2']].values
    y = df['Y'].values
    
    X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
    return X_with_intercept, y


def logistic_loss_gradient(w, X, y, lambd):
    z = np.dot(X, w)
    logistic_loss = np.mean(np.log(1 + np.exp(-y * z))) + lambd * np.sum(w ** 2)
    gradient = -(np.dot(X.T, y * (1 - 1 / (1 + np.exp(-y * z)))) / len(y)) + 2 * lambd * w
    return logistic_loss, gradient


def gradient_descent_logistic(X, y, eta, lambd, max_iterations=10000, convergence_threshold=1e-4):
    w = np.zeros(X.shape[1])
    for iteration in range(max_iterations):
        _, grad = logistic_loss_gradient(w, X, y, lambd)
        w_new = w - eta * grad
        if np.linalg.norm(w_new - w) < convergence_threshold:
            break
        w = w_new
    return w, iteration + 1


def calculate_margin(w, X, y):
    norm_w = np.linalg.norm(w[1:])
    if norm_w == 0: return float('inf')  
    distances = np.abs(np.dot(X, w)) / norm_w
    return np.min(distances)


def analyze_dataset_with_margin(file_path, etas, lambdas):
    X, y = load_dataset(file_path)
    for eta in etas:
        for lambd in lambdas:
            w, iterations = gradient_descent_logistic(X, y, eta, lambd)
            margin = calculate_margin(w, X, y)
            print(f"η: {eta}, λ: {lambd}, Iterations: {iterations}, Margin: {margin:.4f}")


file_path_hard = r'C:\Users\cools\Downloads\hard.csv'
file_path_easy = r'C:\Users\cools\Downloads\easy.csv'


etas = [0.01, 0.05, 0.1, 0.5, 1]  
lambdas = [0, 0.01, 1, 100]  


print("Analyzing Easy Dataset with Logistic Loss and Margins:")
analyze_dataset_with_margin(file_path_easy, etas, lambdas)
print("Analyzing Hard Dataset with Logistic Loss and Margins:")
analyze_dataset_with_margin(file_path_hard, etas, lambdas)


