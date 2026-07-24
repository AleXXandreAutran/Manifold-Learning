# Manifold Learning

This repository contains three Python demos for classical nonlinear dimensionality reduction methods applied to the Swiss Roll dataset:

1. **Isomap**
2. **Laplacian Eigenmaps**
3. **Locally Linear Embedding (LLE)**

The goal is to illustrate how high-dimensional nonlinear data can be represented in a lower-dimensional space while preserving different kinds of geometric information.

## Installation

```bash
pip install -r requirements.txt
```

## 1. Isomap

**Purpose:** preserve approximate geodesic distances along the underlying manifold.

Isomap works as follows:

1. Build a nearest-neighbor graph.
2. Approximate geodesic distances using shortest paths in the graph.
3. Convert the squared distance matrix into a kernel-like matrix using double centering:

$$
K_{\mathrm{Iso}} = -\frac{1}{2}H\Delta^2H
$$

4. Compute the leading eigenvectors to obtain the low-dimensional embedding.

Isomap is useful when the data lie on a curved manifold and global distances along the manifold are meaningful.

Run:

```bash
python isomap_demo.py
```

## 2. Laplacian Eigenmaps

**Purpose:** preserve local neighborhood relationships.

Laplacian Eigenmaps works as follows:

1. Build a nearest-neighbor graph.
2. Construct a sparse weight matrix:

$$
W_{ij} = \exp\left(-\frac{\lVert x_i-x_j \rVert^2}{\sigma^2}\right)
$$

3. Compute the degree matrix:

$$
D_{ii} = \sum_j W_{ij}
$$

4. Build the graph Laplacian:

$$
L = D - W
$$

5. Solve the generalized eigenvalue problem:

$$
Lu = \lambda Du
$$

The first eigenvector is usually trivial and is ignored. The next eigenvectors give the low-dimensional coordinates.

Run:

```bash
python laplacian_eigenmaps_demo.py
```

## 3. Locally Linear Embedding (LLE)

**Purpose:** preserve local linear reconstruction relationships.

LLE works as follows:

1. Find the nearest neighbors of each point.
2. Reconstruct each point as a linear combination of its neighbors:

$$
x_i \approx \sum_j W_{ij}x_j
$$

with the constraint:

$$
\sum_j W_{ij} = 1
$$

3. Keep the reconstruction weights fixed and find low-dimensional points that satisfy:

$$
y_i \approx \sum_j W_{ij}y_j
$$

4. Build the matrix:

$$
M = (I-W)^\top(I-W)
$$

5. Use the smallest non-trivial eigenvectors of $M$ as the embedding coordinates.

Run:

```bash
python lle_demo.py
```

## Large sparse matrices

For `n_samples = 5000`, matrices such as $W$, $L$, and $M$ have size:

$$
5000 \times 5000 = 25{,}000{,}000
$$

entries. However, they are usually very sparse because each point is connected only to a limited number of neighbors. For example, with `n_neighbors = 30`, only a small fraction of entries are nonzero.

Because of this, full `imshow` plots can look almost empty or almost uniform. The scripts therefore use:

- `spy` plots to show the sparsity pattern;
- zoomed matrix views to show local structure;
- explanatory text under the figures.

## Relationship with KPCA

These algorithms can be interpreted as **specific instances of Kernel PCA (KPCA)** under particular kernel or kernel-like constructions.

- **Isomap** can be viewed through the kernel matrix obtained by double centering the squared geodesic distance matrix.
- **Laplacian Eigenmaps** is related to KPCA through kernels based on the graph Laplacian or its pseudo-inverse.
- **LLE** can also be connected to KPCA through the matrix built from the reconstruction operator, such as $(I-W)^\top(I-W)$, with suitable normalization or pseudo-inverse interpretations.

In this sense, all three methods are spectral manifold learning methods and can be understood as specific KPCA-like constructions.

## Notes

- For very large datasets, sparse matrix methods are preferred.
- Eigenvalue solvers may require tuning depending on `n_samples`, `n_neighbors`, and numerical conditioning.

## Reference

This work is based on the manifold learning section from:

Mohri, M., Rostamizadeh, A., & Talwalkar, A. (2018). *Foundations of Machine Learning* (2nd ed.). The MIT Press. ISBN: 9780262039406.
