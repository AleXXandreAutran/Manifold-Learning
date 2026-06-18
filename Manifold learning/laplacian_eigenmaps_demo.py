"""
Laplacian Eigenmaps demo on the Swiss Roll dataset.

Goal:
    Illustrate nonlinear dimensionality reduction with Laplacian Eigenmaps.

Main idea:
    1. Build a nearest-neighbor graph.
    2. Assign weights to local edges with a heat kernel.
    3. Build the graph Laplacian L = D - W.
    4. Use the smallest non-trivial eigenvectors to obtain a low-dimensional embedding.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh, ArpackNoConvergence


# Parameters

n_samples = 5000
n_neighbors = 30
target_dim = 2
sigma = 2.0
random_state = 42


# 1. Generate data

X, color = make_swiss_roll(n_samples=n_samples, noise=0.05, random_state=random_state)
color = (color - color.min()) / (color.max() - color.min())
n = X.shape[0]


# 2. Find nearest neighbors

nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean").fit(X)
distances, indices = nbrs.kneighbors(X)


# 3. Build sparse weight matrix W

rows, cols, data = [], [], []
for i in range(n):
    for j, d in zip(indices[i, 1:], distances[i, 1:]):
        weight = np.exp(-(d ** 2) / (sigma ** 2))
        rows.append(i)
        cols.append(j)
        data.append(weight)
        rows.append(j)
        cols.append(i)
        data.append(weight)

W = csr_matrix((data, (rows, cols)), shape=(n, n))
W.sum_duplicates()
W.data = np.clip(W.data, 0.0, 1.0)


# 4. Build D and L = D - W

degrees = np.asarray(W.sum(axis=1)).ravel()
D = diags(degrees, format="csr")
L = D - W


# 5. Generalized eigenvalue problem L u = lambda D u

try:
    eigvals, eigvecs = eigsh(
        L,
        k=target_dim + 1,
        M=D,
        which="SM",
        maxiter=50000,
        tol=1e-6
    )
except ArpackNoConvergence as error:
    print("ARPACK did not fully converge.")
    print("Number of converged eigenvalues:", len(error.eigenvalues))
    if error.eigenvectors is not None and len(error.eigenvalues) >= target_dim + 1:
        eigvals = error.eigenvalues
        eigvecs = error.eigenvectors
    else:
        raise RuntimeError("Not enough eigenvectors converged. Try reducing n_samples or increasing n_neighbors.")

idx = np.argsort(eigvals)
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
Y = eigvecs[:, 1:target_dim + 1]


# 6. Original data

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


# 7. Graph subset

subsample = min(220, n)
rng = np.random.default_rng(random_state)
subset = rng.choice(n, size=subsample, replace=False)
subset_set = set(subset)

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[subset, 0], X[subset, 1], X[subset, 2], c=color[subset], cmap="Spectral", s=30, label="Data points")
for i in subset:
    for j in indices[i, 1:]:
        if j in subset_set:
            ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], [X[i, 2], X[j, 2]], linewidth=0.5, alpha=0.5)
ax.set_title(f"{n_neighbors}-nearest-neighbor graph")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
ax.legend()
fig.text(0.5, 0.02, "Only a subset of points is displayed to avoid an overloaded graph.", ha="center", fontsize=10)
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.show()


# 8. Sparsity and zoom plots

fig, ax = plt.subplots(figsize=(7, 7))
ax.spy(W, markersize=0.2)
ax.set_title("Sparsity pattern of the weight matrix $W$")
ax.set_xlabel("Index $j$")
ax.set_ylabel("Index $i$")
fig.text(0.5, 0.02, f"W has size {n_samples} × {n_samples}. Since each point is connected only to {n_neighbors} neighbors, most entries are zero.", ha="center", fontsize=10, wrap=True)
plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.show()

zoom_size = min(300, n)
fig, ax = plt.subplots(figsize=(7, 7))
im = ax.imshow(W[:zoom_size, :zoom_size].toarray(), cmap="viridis", aspect="auto")
fig.colorbar(im, ax=ax, label="Weight value")
ax.set_title(f"Zoom on weight matrix $W$ — first {zoom_size} points")
ax.set_xlabel("Index $j$")
ax.set_ylabel("Index $i$")
fig.text(0.5, 0.02, "The full matrix is too large and sparse to be clearly visualized with imshow. A zoom or spy plot is more informative.", ha="center", fontsize=10, wrap=True)
plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.show()

fig, ax = plt.subplots(figsize=(7, 7))
ax.spy(L, markersize=0.2)
ax.set_title("Sparsity pattern of the graph Laplacian $L = D - W$")
ax.set_xlabel("Index $j$")
ax.set_ylabel("Index $i$")
fig.text(0.5, 0.02, "The Laplacian has the same off-diagonal sparsity pattern as W, plus diagonal degree terms from D.", ha="center", fontsize=10, wrap=True)
plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.show()

fig, ax = plt.subplots(figsize=(7, 7))
im = ax.imshow(L[:zoom_size, :zoom_size].toarray(), cmap="viridis", aspect="auto")
fig.colorbar(im, ax=ax, label="Laplacian value")
ax.set_title(f"Zoom on graph Laplacian $L=D-W$ — first {zoom_size} points")
ax.set_xlabel("Index $j$")
ax.set_ylabel("Index $i$")
fig.text(0.5, 0.02, "The visible diagonal comes from D. Off-diagonal values are negative weights for neighboring points.", ha="center", fontsize=10, wrap=True)
plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.show()


# 9. Embedding and eigenvalues

fig, ax = plt.subplots(figsize=(8, 7))
scatter = ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap="Spectral", s=8)
ax.set_title("2D representation obtained with Laplacian Eigenmaps")
ax.set_xlabel("First non-trivial eigenvector")
ax.set_ylabel("Second non-trivial eigenvector")
ax.grid(True, alpha=0.3)
fig.colorbar(scatter, ax=ax, label="Latent Swiss Roll parameter")
fig.text(0.5, 0.02, "The embedding keeps neighboring points close in the low-dimensional space.", ha="center", fontsize=10, wrap=True)
plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(np.arange(1, len(eigvals) + 1), eigvals, marker="o", label="Eigenvalues")
ax.axvline(x=1, linestyle="--", label="Trivial eigenvalue")
ax.set_title("Smallest eigenvalues of the graph Laplacian")
ax.set_xlabel("Rank")
ax.set_ylabel("Eigenvalue")
ax.grid(True, alpha=0.3)
ax.legend()
fig.text(0.5, 0.02, "The first eigenvalue is usually close to zero and corresponds to a trivial constant eigenvector.", ha="center", fontsize=10, wrap=True)
plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.show()


# 10. Summary diagram

fig, ax = plt.subplots(figsize=(13, 4))
ax.axis("off")
steps = [
    "1. High-dimensional\ndata",
    "2. Nearest-neighbor\ngraph",
    "3. Weight matrix\n$W_{ij}=e^{-||x_i-x_j||^2/\\sigma^2}$",
    "4. Degree matrix\n$D_{ii}=\\sum_j W_{ij}$",
    "5. Graph Laplacian\n$L=D-W$",
    "6. Eigenvectors\nof $Lu=\\lambda Du$",
    "7. Low-dimensional\nembedding",
]
x_positions = np.linspace(0.06, 0.94, len(steps))
for i, (x, text) in enumerate(zip(x_positions, steps)):
    ax.text(x, 0.5, text, ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.45", edgecolor="black", facecolor="white"))
    if i < len(steps) - 1:
        ax.annotate("", xy=(x_positions[i + 1] - 0.055, 0.5), xytext=(x + 0.055, 0.5), arrowprops=dict(arrowstyle="->", linewidth=1.8))
ax.set_title("Summary diagram of the Laplacian Eigenmaps algorithm", fontsize=15)
plt.tight_layout()
plt.show()

nonzero_W = W.count_nonzero()
print("Laplacian Eigenmaps summary")
print(f"Number of samples: {n_samples}")
print(f"Number of neighbors: {n_neighbors}")
print(f"Original dimension: {X.shape[1]}")
print(f"Target dimension: {target_dim}")
print(f"Sigma: {sigma}")
print(f"Percentage of nonzero entries in W: {100 * nonzero_W / (n * n):.4f}%")
print("Smallest eigenvalues:")
print(eigvals)
