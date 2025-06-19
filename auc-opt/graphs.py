# graphs.py

import matplotlib.pyplot as plt
import numpy as np

def plot_dataset_with_separator(dataset):
    X = dataset.X
    y = dataset.y
    w = dataset.w_svm

    if X.shape[0] != 2:
        raise ValueError("This plot only works for 2D features (n=2)")

    plt.figure(figsize=(6, 6))

    # Plot data points
    plt.scatter(X[0, y == 0], X[1, y == 0], color='red', label='Class 0')
    plt.scatter(X[0, y == 1], X[1, y == 1], color='blue', label='Class 1')

    # Plot separator
    x_vals = np.linspace(X[0].min(), X[0].max(), 100)
    slope = -w[0] / w[1]
    intercept = 0
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, color='black', linestyle='--', label='w_svm separator')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Synthetic Dataset with Separator')
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def prox_approx_graph(delta, label=""):
    def f(x):
        return np.minimum(1, np.maximum(0, x - delta))
    
    x_vals = np.linspace(-50, 50, 1000)
    y_vals = f(x_vals)

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("ℓ(x)")
    ax.set_title(f"Linear Indicator Approximation ({label})")
    ax.legend().set_visible(False)

    return fig, ax, f  # f is returned so it can be reused

def prox_all_points_graph(ax, w, lambda_vec, sigma, PI, f, label_prefix="initial"):
    N = len(PI.K)
    x_vals = np.empty(N)
    y_vals = np.empty(N)

    for idx, (i, j) in enumerate(PI.K):
        Dij = PI.D[:, idx]
        w_dot_dij = np.dot(w, Dij)
        lambda_ij = lambda_vec[idx]
        x_vals[idx] = w_dot_dij - lambda_ij / sigma
        y_vals[idx] = f(x_vals[idx])

    color = 'blue' if label_prefix == "initial" else 'red'
    ax.scatter(x_vals, y_vals, s=10, alpha=0.3, color=color)
