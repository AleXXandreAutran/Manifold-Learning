"""
Isomap demo on the Swiss Roll dataset.

Goal:
    Illustrate nonlinear dimensionality reduction with Isomap.

Main idea:
    1. Build a nearest-neighbor graph.
    2. Approximate geodesic distances with shortest paths in the graph.
    3. Apply classical MDS through double centering.
    4. Use the leading eigenvectors to obtain a low-dimensional embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path


# Parameters

n_samples = 1500
n_neighbors = 10
target_dim = 2
random_state = 42


# 1. Generate data

X, color = make_swiss_roll(
    n_samples=n_samples,
    noise=0.05,
    random_state=random_state
)

color = (color - color.min()) / (color.max() - color.min())
n = X.shape[0]


# 2. Build nearest-neighbor graph

nbrs = NearestNeighbors(
    n_neighbors=n_neighbors + 1,
    metric="euclidean"
).fit(X)

distances, indices = nbrs.kneighbors(X)

G_dist = np.full((n, n), np.inf)
np.fill_diagonal(G_dist, 0.0)

for i in range(n):
    for j, d in zip(indices[i, 1:], distances[i, 1:]):
        G_dist[i, j] = d
        G_dist[j, i] = d


# 3. Approximate geodesic distances

Delta = shortest_path(G_dist, directed=False, unweighted=False)
Delta_squared = Delta ** 2


# 4. Double centering: K_iso = -1/2 H Delta^2 H

I = np.eye(n)
H = I - np.ones((n, n)) / n
K_iso = -0.5 * H @ Delta_squared @ H


# 5. Spectral decomposition

eigvals, eigvecs = np.linalg.eigh(K_iso)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

positive = eigvals > 1e-10
eigvals_pos = eigvals[positive]
eigvecs_pos = eigvecs[:, positive]

Y = eigvecs_pos[:, :target_dim] @ np.diag(np.sqrt(eigvals_pos[:target_dim]))


# 6. Original 3D data

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap="Spectral", s=12)
ax.set_title("Original data: Swiss Roll in 3D")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
fig.colorbar(scatter, ax=ax, shrink=0.7, label="Latent Swiss Roll parameter")
fig.text(0.5, 0.02, "The Swiss Roll is a 2D manifold embedded in 3D.", ha="center", fontsize=10)
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.show()


# 7. Nearest-neighbor graph on a subset

subsample = min(180, n)
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
fig.text(0.5, 0.02, "Only a subset is displayed to keep the graph readable.", ha="center", fontsize=10)
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.show()


# 8. Euclidean distance vs approximate geodesic distance

p1 = np.argmin(color)
p2 = np.argmax(color)

G_nx = nx.Graph()
for i in range(n):
    for j in indices[i, 1:]:
        G_nx.add_edge(i, j, weight=np.linalg.norm(X[i] - X[j]))

path = nx.shortest_path(G_nx, source=p1, target=p2, weight="weight")
path_points = X[path]

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap="Spectral", s=10, alpha=0.55, label="Data")
ax.plot([X[p1, 0], X[p2, 0]], [X[p1, 1], X[p2, 1]], [X[p1, 2], X[p2, 2]], linewidth=3, label="Direct Euclidean distance")
ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], linewidth=3, label="Approximate geodesic distance")
ax.scatter(X[[p1, p2], 0], X[[p1, p2], 1], X[[p1, p2], 2], s=80, label="Two distant points")
ax.set_title("Euclidean distance vs approximate geodesic distance")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
ax.legend()
fig.text(0.5, 0.02, "Isomap uses graph shortest paths to approximate distances along the manifold.", ha="center", fontsize=10)
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.show()


# 9. Geodesic distance matrix

fig, ax = plt.subplots(figsize=(7, 7))
im = ax.imshow(Delta, cmap="viridis", aspect="auto")
fig.colorbar(im, ax=ax, label="Approximate geodesic distance")
ax.set_title("Geodesic distance matrix $\\Delta_{ij}$")
ax.set_xlabel("Index $j$")
ax.set_ylabel("Index $i$")
fig.text(0.5, 0.02, "For large n, full distance matrices are difficult to visualize because they contain n² entries.", ha="center", fontsize=10, wrap=True)
plt.tight_layout(rect=[0, 0.09, 1, 1])
plt.show()


# 10. Isomap embedding

fig, ax = plt.subplots(figsize=(8, 7))
scatter = ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap="Spectral", s=12)
ax.set_title("2D representation obtained with Isomap")
ax.set_xlabel("First Isomap coordinate")
ax.set_ylabel("Second Isomap coordinate")
ax.grid(True, alpha=0.3)
fig.colorbar(scatter, ax=ax, label="Latent Swiss Roll parameter")
fig.text(0.5, 0.02, "The embedding preserves approximate geodesic distances as well as possible in 2D.", ha="center", fontsize=10, wrap=True)
plt.tight_layout(rect=[0, 0.09, 1, 1])
plt.show()


# 11. Summary diagram

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis("off")
steps = [
    "1. High-dimensional\ndata",
    "2. Nearest-neighbor\ngraph",
    "3. Shortest paths\n≈ geodesic distances",
    "4. Double centering\n$K=-\\frac{1}{2}H\\Delta^2H$",
    "5. Eigenvectors\nof $K$",
    "6. 2D\nembedding",
]
x_positions = np.linspace(0.07, 0.93, len(steps))
for i, (x, text) in enumerate(zip(x_positions, steps)):
    ax.text(x, 0.5, text, ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.45", edgecolor="black", facecolor="white"))
    if i < len(steps) - 1:
        ax.annotate("", xy=(x_positions[i + 1] - 0.065, 0.5), xytext=(x + 0.065, 0.5), arrowprops=dict(arrowstyle="->", linewidth=1.8))
ax.set_title("Summary diagram of the Isomap algorithm", fontsize=15)
plt.tight_layout()
plt.show()

print("Isomap summary")
print(f"Number of samples: {n_samples}")
print(f"Number of neighbors: {n_neighbors}")
print(f"Original dimension: {X.shape[1]}")
print(f"Target dimension: {target_dim}")
print("Leading positive eigenvalues:")
print(eigvals_pos[:5])
