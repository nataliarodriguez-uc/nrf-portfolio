import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve

def plot_dataset_with_separator(dataset):
    
    """
    Plots a 2D synthetic dataset with two classes and the ground truth separator w_svm.

    Args:
    - dataset: A DataSet object with 2D features (n=2), class labels, and a separating hyperplane.

    Displays:
    - A scatter plot with class-colored points and a dashed line representing w_svm.
    """
    
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
    
    """
    Plots the linear approximation of the 0-1 loss (indicator) used in the proximal operator.

    Args:
    - delta: The threshold shift for the approximation.
    - label: Optional label for display purposes.

    Returns:
    - fig, ax: Matplotlib figure and axis.
    - f: The approximation function ℓ(x) = min(1, max(0, x - delta)).
    """
    
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
    
    """
    Plots (x, ℓ(x)) values for all (i,j) pairs in the problem, where
    x = wᵗ(D_ij) - λ_ij / σ, and ℓ(x) is the linear surrogate loss.

    Args:
    - ax: Matplotlib axis to plot on (reused from prox_approx_graph).
    - w: Current primal variable (weight vector).
    - lambda_vec: Dual variable vector.
    - sigma: Current penalty parameter.
    - PI: ProblemInstance with D and K.
    - f: Surrogate loss function (from prox_approx_graph).
    - label_prefix: "initial" or "final" to color-code the points.
    """
    
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



def plot_pca_projection(X, y, title="PCA Projection"):
    pca = PCA(n_components=2)
    X_proj = pca.fit_transform(X.T)  # X is (n_features, n_samples)

    plt.figure(figsize=(6, 5))
    plt.scatter(X_proj[y == 0, 0], X_proj[y == 0, 1], alpha=0.6, label="Class 0")
    plt.scatter(X_proj[y == 1, 0], X_proj[y == 1, 1], alpha=0.6, label="Class 1")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

def evaluate_auc_on_test(w, PI, dataset_name="dataset", output_dir="results"):
    """
    Evaluates and plots AUC on the test set of a train/test ProblemInstance.
    Also saves the ROC data and AUC value to CSV in the output directory.
    """
    assert hasattr(PI, 'X_test'), "This function requires mode='train_test' in ProblemInstance."

    # Compute scores and AUC
    scores = w @ PI.X_test
    y_true = PI.y_test
    auc = roc_auc_score(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)

    # Plot and save ROC figure
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {dataset_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"{dataset_name}_roc.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved ROC plot to {fig_path}")

    # Save AUC value to file
    auc_csv_path = os.path.join(output_dir, f"{dataset_name}_auc.csv")
    with open(auc_csv_path, "w") as f:
        f.write("auc\n")
        f.write(f"{auc:.6f}\n")
    print(f"Saved AUC value to {auc_csv_path}")

    print(f"Test AUC for {dataset_name}: {auc:.4f}")
    return auc
