"""
Locally Linear Embedding demo on the Swiss Roll dataset.

Goal:
    Illustrate nonlinear dimensionality reduction with LLE.

Main idea:
    1. Build local neighborhoods.
    2. Reconstruct each point as a linear combination of its neighbors.
    3. Keep the reconstruction weights fixed.
    4. Find low-dimensional coordinates that preserve these local reconstructions.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import eigsh, ArpackNoConvergence


# ============================================================
# Parameters
# ============================================================

n_samples = 5000
n_neighbors = 30
target_dim = 2
regularization = 1e-2
random_state = 42


# ============================================================
# 1. Generate data
# ============================================================

X, color = make_swiss_roll(n_samples=n_samples, noise=0.05, random_state=random_state)
color = (color - color.min()) / (color.max() - color.min())
n = X.shape[0]


# ============================================================
# 2. Find nearest neighbors
# ============================================================

nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean").fit(X)
distances, indices = nbrs.kneighbors(X)
neighbor_indices = indices[:, 1:]


# ============================================================
# 3. Compute reconstruction weights W
# ============================================================

rows, cols, data = [], [], []

for i in range(n):
    neighbors = neighbor_indices[i]
    Z = X[neighbors] - X[i]
    C = Z @ Z.T

    trace_C = np.trace(C)
    if trace_C > 0:
        C += regularization * trace_C * np.eye(n_neighbors)
    else:
        C += regularization * np.eye(n_neighbors)

    ones = np.ones(n_neighbors)
    w = np.linalg.solve(C, ones)
    w = w / np.sum(w)

    rows.extend([i] * n_neighbors)
    cols.extend(neighbors)
    data.extend(w)

W = csr_matrix((data, (rows, cols)), shape=(n, n))


# ============================================================
# 4. Build M = (I - W)^T (I - W)
# ============================================================

I = eye(n, format="csr")
A = I - W
M = A.T @ A


# ============================================================
# 5. Sparse eigenvalue problem
# ============================================================

try:
    eigvals, eigvecs = eigsh(
        M,
        k=target_dim + 1,
        sigma=0.0,
        which="LM",
        maxiter=50000,
        tol=1e-7
    )
except ArpackNoConvergence as error:
    print("ARPACK did not fully converge.")
    print("Number of converged eigenvalues:", len(error.eigenvalues))
    if error.eigenvectors is not None and len(error.eigenvalues) >= target_dim + 1:
        eigvals = error.eigenvalues
        eigvecs = error.eigenvectors
    else:
        raise RuntimeError(
            "Not enough eigenvectors converged. Try increasing maxiter, increasing regularization, or reducing n_samples."
        )

idx = np.argsort(eigvals)
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
Y = eigvecs[:, 1:target_dim + 1]


# ============================================================
# 6. Original data
# ============================================================

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap="Spectral", s=8)
ax.set_title("Original data: Swiss Roll in 3D")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
fig.colorbar(scatter, ax=ax, shrink=0.7, label="Latent Swiss Roll parameter")
fig.text(0.5, 0.02, "The Swiss Roll is a 2D manifold embedded in 3D.", ha="center", fontsize=10)
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.show()


# ============================================================
# 7. Nearest-neighbor graph subset
# ============================================================

subsample = min(220, n)
rng = np.random.default_rng(random_state)
subset = rng.choice(n, size=subsample, replace=False)
subset_set = set(subset)

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[subset, 0], X[subset, 1], X[subset, 2], c=color[subset], cmap="Spectral", s=30, label="Data points")
for i in subset:
    for j in neighbor_indices[i]:
        if j in subset_set:
            ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], [X[i, 2], X[j, 2]], linewidth=0.5, alpha=0.5)
ax.set_title(f"{n_neighbors}-nearest-neighbor graph")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
ax.legend()
fig.text(0.5, 0.02, "Only a subset is displayed. LLE uses local neighborhoods to reconstruct each point from its neighbors.", ha="center", fontsize=10, wrap=True)
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.show()


# ============================================================
# 8. Reconstruction weights for one point
# ============================================================

point_id = 0
weights_point = W.getrow(point_id).toarray().ravel()
nonzero_neighbors = weights_point.nonzero()[0]
nonzero_weights = weights_point[nonzero_neighbors]

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(np.arange(len(nonzero_weights)), nonzero_weights)
ax.set_title(f"Reconstruction weights for point {point_id}")
ax.set_xlabel("Neighbor index inside the local neighborhood")
ax.set_ylabel("Weight value")
ax.grid(True, alpha=0.3)
fig.text(0.5, 0.02, "These weights reconstruct one point from its nearest neighbors. The weights sum to 1 and may be negative.", ha="center", fontsize=10, wrap=True)
plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.show()


# ============================================================
# 9. Sparse matrix visualizations
# ============================================================

fig, ax = plt.subplots(figsize=(7, 7))
ax.spy(W, markersize=0.2)
ax.set_title("Sparsity pattern of the reconstruction matrix $W$")
ax.set_xlabel("Index $j$")
ax.set_ylabel("Index $i$")
fig.text(0.5, 0.02, f"W has size {n_samples} × {n_samples}. Since each point is reconstructed using only {n_neighbors} neighbors, most entries are zero.", ha="center", fontsize=10, wrap=True)
plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.show()

zoom_size = min(300, n)
fig, ax = plt.subplots(figsize=(7, 7))
im = ax.imshow(W[:zoom_size, :zoom_size].toarray(), cmap="viridis", aspect="auto")
fig.colorbar(im, ax=ax, label="Weight value")
ax.set_title(f"Zoom on reconstruction matrix $W$ — first {zoom_size} points")
ax.set_xlabel("Index $j$")
ax.set_ylabel("Index $i$")
fig.text(0.5, 0.02, "The complete matrix is too large and sparse to be clearly visualized as a dense image.", ha="center", fontsize=10, wrap=True)
plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.show()

fig, ax = plt.subplots(figsize=(7, 7))
ax.spy(M, markersize=0.2)
ax.set_title("Sparsity pattern of $M=(I-W)^T(I-W)$")
ax.set_xlabel("Index $j$")
ax.set_ylabel("Index $i$")
fig.text(0.5, 0.02, "M contains the reconstruction error structure used to compute the LLE embedding.", ha="center", fontsize=10, wrap=True)
plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.show()

fig, ax = plt.subplots(figsize=(7, 7))
im = ax.imshow(M[:zoom_size, :zoom_size].toarray(), cmap="viridis", aspect="auto")
fig.colorbar(im, ax=ax, label="Matrix value")
ax.set_title(f"Zoom on $M=(I-W)^T(I-W)$ — first {zoom_size} points")
ax.set_xlabel("Index $j$")
ax.set_ylabel("Index $i$")
fig.text(0.5, 0.02, "For n = 5000, dense matrix images are difficult to read because most entries are zero.", ha="center", fontsize=10, wrap=True)
plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.show()


# ============================================================
# 10. Embedding and eigenvalues
# ============================================================

fig, ax = plt.subplots(figsize=(8, 7))
scatter = ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap="Spectral", s=8)
ax.set_title("2D representation obtained with LLE")
ax.set_xlabel("First non-trivial eigenvector")
ax.set_ylabel("Second non-trivial eigenvector")
ax.grid(True, alpha=0.3)
fig.colorbar(scatter, ax=ax, label="Latent Swiss Roll parameter")
fig.text(0.5, 0.02, "LLE preserves local linear reconstruction relationships in the low-dimensional space.", ha="center", fontsize=10, wrap=True)
plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(np.arange(1, len(eigvals) + 1), eigvals, marker="o", label="Eigenvalues")
ax.axvline(x=1, linestyle="--", label="Trivial eigenvalue")
ax.set_title("Smallest eigenvalues of the LLE cost matrix $M$")
ax.set_xlabel("Rank")
ax.set_ylabel("Eigenvalue")
ax.grid(True, alpha=0.3)
ax.legend()
fig.text(0.5, 0.02, "The smallest eigenvalue corresponds to a trivial constant solution. The next eigenvectors define the 2D coordinates.", ha="center", fontsize=10, wrap=True)
plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.show()


# ============================================================
# 11. Summary diagram
# ============================================================

fig, ax = plt.subplots(figsize=(13, 4))
ax.axis("off")
steps = [
    "1. High-dimensional\ndata",
    "2. Nearest-neighbor\ngraph",
    "3. Compute local\nreconstruction weights",
    "4. Build matrix\n$M=(I-W)^T(I-W)$",
    "5. Eigenvectors\nof $M$",
    "6. Low-dimensional\nembedding",
]
x_positions = np.linspace(0.07, 0.93, len(steps))
for i, (x, text) in enumerate(zip(x_positions, steps)):
    ax.text(x, 0.5, text, ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.45", edgecolor="black", facecolor="white"))
    if i < len(steps) - 1:
        ax.annotate("", xy=(x_positions[i + 1] - 0.065, 0.5), xytext=(x + 0.065, 0.5), arrowprops=dict(arrowstyle="->", linewidth=1.8))
ax.set_title("Summary diagram of the Locally Linear Embedding algorithm", fontsize=15)
plt.tight_layout()
plt.show()

nonzero_W = W.count_nonzero()
print("Locally Linear Embedding summary")
print("--------------------------------")
print(f"Number of samples: {n_samples}")
print(f"Number of neighbors: {n_neighbors}")
print(f"Original dimension: {X.shape[1]}")
print(f"Target dimension: {target_dim}")
print(f"Regularization: {regularization}")
print(f"Percentage of nonzero entries in W: {100 * nonzero_W / (n * n):.4f}%")
print("Smallest eigenvalues:")
print(eigvals)
