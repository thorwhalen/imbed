# imbed.components.planarization

Planarization functions for embedding visualization.

### Functions

| [`circular_planarizer`](#imbed.components.planarization.circular_planarizer)(vectors)                     | Places vectors in a circle with similar vectors closer together.        |
|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| [`constant_planarizer`](#imbed.components.planarization.constant_planarizer)(embeddings)                  | Generate basic 2D projections from embeddings                           |
| [`factor_analysis_planarizer`](#imbed.components.planarization.factor_analysis_planarizer)(vectors[, ...])       | Factor Analysis for 2D projection.                                      |
| [`fast_ica_planarizer`](#imbed.components.planarization.fast_ica_planarizer)(vectors[, random_state])     | Fast Independent Component Analysis (FastICA) for 2D projection.        |
| [`gaussian_random_projection_planarizer`](#imbed.components.planarization.gaussian_random_projection_planarizer)(vectors)   | Gaussian Random Projection for 2D projection.                           |
| [`grid_planarizer`](#imbed.components.planarization.grid_planarizer)(vectors)                         | Places vectors in a grid pattern.                                       |
| [`identity_planarizer`](#imbed.components.planarization.identity_planarizer)(vectors)                     | Returns the first two dimensions of each vector.                        |
| [`isomap_planarizer`](#imbed.components.planarization.isomap_planarizer)(vectors[, n_neighbors])        | Isomap for 2D projection.                                               |
| [`kernel_pca_planarizer`](#imbed.components.planarization.kernel_pca_planarizer)(vectors[, kernel, ...])    | Kernel PCA for 2D projection.                                           |
| [`lle_planarizer`](#imbed.components.planarization.lle_planarizer)(vectors[, n_neighbors, ...])      | Locally Linear Embedding (LLE) for 2D projection.                       |
| [`mds_planarizer`](#imbed.components.planarization.mds_planarizer)(vectors[, metric, n_init, ...])   | Multidimensional Scaling (MDS) for 2D projection.                       |
| [`nmf_planarizer`](#imbed.components.planarization.nmf_planarizer)(vectors[, init, random_state])    | Non-negative Matrix Factorization (NMF) for 2D projection.              |
| [`random_planarizer`](#imbed.components.planarization.random_planarizer)(vectors[, scale])              | Randomly projects vectors into 2D space.                                |
| [`robust_pca_planarizer`](#imbed.components.planarization.robust_pca_planarizer)(vectors[, random_state])   | Robust PCA for 2D projection, using a robust scaler before PCA.         |
| [`semantic_pca_planarizer`](#imbed.components.planarization.semantic_pca_planarizer)(vectors[, random_state]) | PCA for 2D projection optimized for L2-normalized semantic embeddings.  |
| [`simple_pca_planarizer`](#imbed.components.planarization.simple_pca_planarizer)(vectors[, random_state])   | Principal Component Analysis (PCA) for 2D projection.                   |
| [`sparse_random_projection_planarizer`](#imbed.components.planarization.sparse_random_projection_planarizer)(vectors)     | Sparse Random Projection for 2D projection.                             |
| [`spectral_embedding_planarizer`](#imbed.components.planarization.spectral_embedding_planarizer)(vectors[, ...])    | Spectral Embedding for 2D projection.                                   |
| [`truncated_svd_planarizer`](#imbed.components.planarization.truncated_svd_planarizer)(vectors[, ...])         | Truncated Singular Value Decomposition (SVD) for 2D projection.         |
| [`tsne_planarizer`](#imbed.components.planarization.tsne_planarizer)(vectors[, perplexity, ...])      | t-SNE (t-Distributed Stochastic Neighbor Embedding) for 2D projection.  |
| [`umap_planarizer`](#imbed.components.planarization.umap_planarizer)(vectors[, n_neighbors, ...])     | UMAP (Uniform Manifold Approximation and Projection) for 2D projection. |

### imbed.components.planarization.circular_planarizer(vectors)

Places vectors in a circle with similar vectors closer together.

* **Parameters:**
  **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  A sequence of 2D points arranged in a circle

### imbed.components.planarization.constant_planarizer(embeddings)

Generate basic 2D projections from embeddings

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]

### imbed.components.planarization.factor_analysis_planarizer(vectors, random_state=42)

Factor Analysis for 2D projection.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **random_state** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Random seed for reproducibility
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  A sequence of 2D points from Factor Analysis

### imbed.components.planarization.fast_ica_planarizer(vectors, random_state=42)

Fast Independent Component Analysis (FastICA) for 2D projection.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **random_state** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Random seed for reproducibility
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  A sequence of 2D points from FastICA projection

### imbed.components.planarization.gaussian_random_projection_planarizer(vectors, random_state=42)

Gaussian Random Projection for 2D projection.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **random_state** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Random seed for reproducibility
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  A sequence of 2D points from Gaussian Random Projection

### imbed.components.planarization.grid_planarizer(vectors)

Places vectors in a grid pattern.

* **Parameters:**
  **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  A sequence of 2D points arranged in a grid

### imbed.components.planarization.identity_planarizer(vectors)

Returns the first two dimensions of each vector.
If vectors have fewer than 2 dimensions, pads with zeros.

* **Parameters:**
  **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  A sequence of 2D points

```pycon
>>> identity_planarizer([[1, 2, 3], [4, 5, 6]])
[(1.0, 2.0), (4.0, 5.0)]
>>> identity_planarizer([[1], [2]])
[(1.0, 0.0), (2.0, 0.0)]
```

### imbed.components.planarization.isomap_planarizer(vectors, n_neighbors=5)

Isomap for 2D projection.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **n_neighbors** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of neighbors to consider for each point
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  A sequence of 2D points from Isomap projection

### imbed.components.planarization.kernel_pca_planarizer(vectors, kernel='rbf', gamma=None, random_state=42)

Kernel PCA for 2D projection.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **kernel** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Kernel type (‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘cosine’)
  * **gamma** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’ kernels
  * **random_state** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Random seed for reproducibility
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  A sequence of 2D points from Kernel PCA projection

### imbed.components.planarization.lle_planarizer(vectors, n_neighbors=5, method='standard', random_state=42)

Locally Linear Embedding (LLE) for 2D projection.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **n_neighbors** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of neighbors to consider for each point
  * **method** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – LLE algorithm to use (‘standard’, ‘hessian’, ‘modified’, or ‘ltsa’)
  * **random_state** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Random seed for reproducibility
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  A sequence of 2D points from LLE projection

### imbed.components.planarization.mds_planarizer(vectors, metric=True, n_init=4, max_iter=300, random_state=42)

Multidimensional Scaling (MDS) for 2D projection.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **metric** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, perform metric MDS; otherwise, perform nonmetric MDS
  * **n_init** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of times the SMACOF algorithm will be run with different initializations
  * **max_iter** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum number of iterations of the SMACOF algorithm
  * **random_state** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Random seed for reproducibility
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  A sequence of 2D points from MDS projection

### imbed.components.planarization.nmf_planarizer(vectors, init='nndsvd', random_state=42)

Non-negative Matrix Factorization (NMF) for 2D projection.
Works only for non-negative data.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of non-negative vectors
  * **init** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Method used to initialize the procedure (‘random’, ‘nndsvd’)
  * **random_state** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Random seed for reproducibility
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  A sequence of 2D points from NMF projection

### imbed.components.planarization.random_planarizer(vectors, scale=1.0)

Randomly projects vectors into 2D space.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **scale** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Scale factor for the random projections
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  A sequence of random 2D points

```pycon
>>> random_planarizer([[1, 2, 3], [4, 5, 6]], scale=0.5)
[(0.37454011796069593, 0.4590583266505292), (0.32919921068172773, 0.7365648894035036)]
```

### imbed.components.planarization.robust_pca_planarizer(vectors, random_state=42)

Robust PCA for 2D projection, using a robust scaler before PCA.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **random_state** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Random seed for reproducibility
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  A sequence of 2D points from Robust PCA projection

### imbed.components.planarization.semantic_pca_planarizer(vectors, random_state=None)

PCA for 2D projection optimized for L2-normalized semantic embeddings.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]

### imbed.components.planarization.simple_pca_planarizer(vectors, random_state=None)

Principal Component Analysis (PCA) for 2D projection.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **random_state** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Random seed for reproducibility
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  A sequence of 2D points representing the top 2 principal components

### imbed.components.planarization.sparse_random_projection_planarizer(vectors, density='auto', random_state=42)

Sparse Random Projection for 2D projection.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **density** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Density of the random projection matrix
  * **random_state** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Random seed for reproducibility
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  A sequence of 2D points from Sparse Random Projection

### imbed.components.planarization.spectral_embedding_planarizer(vectors, n_neighbors=10, affinity='nearest_neighbors', random_state=42)

Spectral Embedding for 2D projection.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **n_neighbors** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of neighbors to consider for each point (when affinity=’nearest_neighbors’)
  * **affinity** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – How to construct the affinity matrix (‘nearest_neighbors’, ‘rbf’, or ‘precomputed’)
  * **random_state** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Random seed for reproducibility
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  A sequence of 2D points from Spectral Embedding

### imbed.components.planarization.truncated_svd_planarizer(vectors, algorithm='randomized', n_iter=5, random_state=42)

Truncated Singular Value Decomposition (SVD) for 2D projection.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **algorithm** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – SVD solver algorithm (‘arpack’ or ‘randomized’)
  * **n_iter** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of iterations for randomized SVD solver
  * **random_state** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Random seed for reproducibility
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  A sequence of 2D points from Truncated SVD projection

### imbed.components.planarization.tsne_planarizer(vectors, perplexity=30.0, learning_rate=200.0, n_iter=1000, metric='cosine', norm='l2', init='pca', random_state=42)

t-SNE (t-Distributed Stochastic Neighbor Embedding) for 2D projection.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **perplexity** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – The perplexity parameter for t-SNE
  * **learning_rate** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – The learning rate for t-SNE
  * **n_iter** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of iterations
  * **metric** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Distance metric to use
  * **init** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Initialization method for t-SNE
  * **random_state** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Random seed for reproducibility
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  A sequence of 2D points from t-SNE projection

### imbed.components.planarization.umap_planarizer(vectors, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)

UMAP (Uniform Manifold Approximation and Projection) for 2D projection.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **n_neighbors** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of neighbors to consider for each point
  * **min_dist** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Minimum distance between points in the embedding
  * **metric** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Distance metric to use
  * **random_state** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Random seed for reproducibility
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  A sequence of 2D points from UMAP projection
