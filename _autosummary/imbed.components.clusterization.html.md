# imbed.components.clusterization

Clusterization module for imbed.

### Functions

| [`affinity_propagation_clusterer`](#imbed.components.clusterization.affinity_propagation_clusterer)(vectors[, ...])    | Affinity Propagation clustering using scikit-learn.                            |
|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| [`birch_clusterer`](#imbed.components.clusterization.birch_clusterer)(vectors[, n_clusters, ...])       | Birch clustering using scikit-learn.                                           |
| [`bisecting_kmeans_clusterer`](#imbed.components.clusterization.bisecting_kmeans_clusterer)(vectors[, ...])        | Bisecting K-means clustering implementation.                                   |
| [`constant_clusterer`](#imbed.components.clusterization.constant_clusterer)(vectors)                       | Returns alternating [0, 1, 0, 1, ...] cluster IDs regardless of input vectors. |
| [`dbscan_clusterer`](#imbed.components.clusterization.dbscan_clusterer)(vectors[, eps, ...])             | DBSCAN clustering using scikit-learn.                                          |
| [`fit_kmeans`](#imbed.components.clusterization.fit_kmeans)(vectors[, n_clusters, normalize])      | Fit a KMeans model and return it for inference on new vectors.                 |
| [`get_clusterer`](#imbed.components.clusterization.get_clusterer)(name)                               | Get a clusterer by name.                                                       |
| [`gmm_clusterer`](#imbed.components.clusterization.gmm_clusterer)(vectors[, n_components, ...])       | Gaussian Mixture Model clustering using scikit-learn.                          |
| [`hierarchical_clusterer`](#imbed.components.clusterization.hierarchical_clusterer)(vectors[, ...])            | Hierarchical clustering using scikit-learn.                                    |
| [`kmeans_clusterer`](#imbed.components.clusterization.kmeans_clusterer)(vectors[, n_clusters, ...])      | K-means clustering using scikit-learn.                                         |
| [`kmeans_lite_clusterer`](#imbed.components.clusterization.kmeans_lite_clusterer)(vectors[, n_clusters, ...]) | K-means clustering implementation.                                             |
| [`list_available_clusterers`](#imbed.components.clusterization.list_available_clusterers)()                       | Return a list of names of all available clusterers.                            |
| [`meanshift_clusterer`](#imbed.components.clusterization.meanshift_clusterer)(vectors[, quantile, ...])     | Mean-shift clustering using scikit-learn.                                      |
| [`minibatch_kmeans_clusterer`](#imbed.components.clusterization.minibatch_kmeans_clusterer)(vectors[, ...])        | Mini-batch K-means clustering using scikit-learn.                              |
| [`nearest_neighbor_clusterer`](#imbed.components.clusterization.nearest_neighbor_clusterer)(vectors[, ...])        | Clustering based on nearest neighbors.                                         |
| [`optics_clusterer`](#imbed.components.clusterization.optics_clusterer)(vectors[, min_samples, xi, ...]) | OPTICS clustering using scikit-learn.                                          |
| [`random_clusterer`](#imbed.components.clusterization.random_clusterer)(vectors[, n_clusters])           | Randomly assigns cluster IDs to vectors.                                       |
| [`scan_for_clusterers`](#imbed.components.clusterization.scan_for_clusterers)()                             | Scan the module for all registered clusterers.                                 |
| [`spectral_clusterer`](#imbed.components.clusterization.spectral_clusterer)(vectors[, n_clusters, ...])    | Spectral clustering using scikit-learn.                                        |
| [`threshold_clusterer`](#imbed.components.clusterization.threshold_clusterer)(vectors[, threshold, ...])    | Clusters vectors based on a simple distance threshold.                         |

### imbed.components.clusterization.affinity_propagation_clusterer(vectors, damping=0.5, max_iter=200)

Affinity Propagation clustering using scikit-learn.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **damping** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Damping factor
  * **max_iter** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum number of iterations
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]
* **Returns:**
  Cluster assignments for each input vector

### imbed.components.clusterization.birch_clusterer(vectors, n_clusters=3, threshold=0.5, branching_factor=50)

Birch clustering using scikit-learn.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **n_clusters** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of clusters
  * **threshold** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – The radius of the subcluster for a sample to be added
  * **branching_factor** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum number of CF subclusters in each node
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]
* **Returns:**
  Cluster assignments for each input vector

### imbed.components.clusterization.bisecting_kmeans_clusterer(vectors, n_clusters=3, max_iter=100)

Bisecting K-means clustering implementation.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **n_clusters** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of clusters to form
  * **max_iter** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum number of iterations per bisection
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]
* **Returns:**
  Cluster assignments for each input vector

### imbed.components.clusterization.constant_clusterer(vectors)

Returns alternating [0, 1, 0, 1, …] cluster IDs regardless of input vectors.
This is just for testing purposes.

* **Parameters:**
  **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]
* **Returns:**
  A sequence of cluster IDs (alternating 0 and 1)

```pycon
>>> constant_clusterer([[1, 2], [3, 4], [5, 6], [7, 8]])
[0, 1, 0, 1]
>>> constant_clusterer([[1, 2], [3, 4], [5, 6]])
[0, 1, 0]
```

### imbed.components.clusterization.dbscan_clusterer(vectors, eps=0.5, min_samples=5, metric='cosine')

DBSCAN clustering using scikit-learn.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **eps** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – The maximum distance between two samples for them to be considered neighbors
  * **min_samples** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – The number of samples in a neighborhood for a point to be considered a core point
  * **metric** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The distance metric to use (default: ‘cosine’ for semantic embeddings)
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]
* **Returns:**
  Cluster IDs with -1 representing noise points

### imbed.components.clusterization.fit_kmeans(vectors, n_clusters=8, , normalize=True, \*\*kmeans_kwargs)

Fit a KMeans model and return it for inference on new vectors.

Unlike [`kmeans_clusterer()`](#imbed.components.clusterization.kmeans_clusterer), which throws the model away after
labeling the input, this returns the fitted model so the same cluster
codebook can label *new* vectors later (key for inference pipelines).

The returned object exposes:

- `.labels_`: cluster IDs for the input vectors.
- `.cluster_centers_`: centroid matrix.
- `.predict(new_vectors) -> list[int]`: classify new vectors against
  the fitted centroids.
- Direct call: `model(vectors) -> list[int]` (so it matches the
  `Clusterer = Callable[[Vectors], ClusterIDs]` protocol exactly).

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – Training vectors.
  * **n_clusters** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Cluster count.
  * **normalize** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True (default), L2-normalize vectors first so Euclidean
    distance approximates cosine distance — appropriate for semantic
    embeddings.
  * **\*\*kmeans_kwargs** – Passed to `sklearn.cluster.KMeans`.
* **Return type:**
  `_FittedKMeans` instance.

### imbed.components.clusterization.get_clusterer(name)

Get a clusterer by name.

* **Parameters:**
  **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the clusterer
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)
* **Returns:**
  The clusterer function if found, None otherwise

```pycon
>>> get_clusterer('constant_clusterer') == constant_clusterer
True
>>> get_clusterer('nonexistent_clusterer') is None
True
```

### imbed.components.clusterization.gmm_clusterer(vectors, n_components=2, covariance_type='full', random_state=42)

Gaussian Mixture Model clustering using scikit-learn.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **n_components** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of mixture components
  * **covariance_type** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Covariance parameter type
  * **random_state** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Random state for reproducibility
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]
* **Returns:**
  Cluster assignments for each input vector

### imbed.components.clusterization.hierarchical_clusterer(vectors, n_clusters=2, linkage='average', affinity='cosine')

Hierarchical clustering using scikit-learn.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **n_clusters** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of clusters to form
  * **linkage** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Linkage criterion [‘ward’, ‘complete’, ‘average’, ‘single’]
    (Note: ‘ward’ requires affinity=’euclidean’)
  * **affinity** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Distance metric to use (default: ‘cosine’ for semantic embeddings)
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]
* **Returns:**
  Cluster assignments for each input vector

### imbed.components.clusterization.kmeans_clusterer(vectors, n_clusters=8, , init='k-means++', n_init='auto', max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd')

K-means clustering using scikit-learn.

#### NOTE
KMeans uses Euclidean distance internally. For semantic embeddings based on
cosine similarity, it’s recommended to L2-normalize your input vectors before passing
them to this function. When vectors are L2-normalized, Euclidean distance becomes
equivalent to cosine distance.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors (L2-normalize for cosine-like behavior)
  * **n_clusters** – Number of clusters to form
  * **random_state** – Random seed for reproducibility
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]
* **Returns:**
  Cluster assignments for each input vector

### imbed.components.clusterization.kmeans_lite_clusterer(vectors, n_clusters=3, max_iter=100, tol=0.0001, seed=None)

K-means clustering implementation.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **n_clusters** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of clusters to form
  * **max_iter** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum number of iterations
  * **tol** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Tolerance for convergence
  * **seed** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Random seed for reproducibility
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]
* **Returns:**
  Cluster assignments for each input vector

### imbed.components.clusterization.list_available_clusterers()

Return a list of names of all available clusterers.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
* **Returns:**
  List of clusterer names

```pycon
>>> 'constant_clusterer' in list_available_clusterers()
True
```

### imbed.components.clusterization.meanshift_clusterer(vectors, quantile=0.3, n_samples=None)

Mean-shift clustering using scikit-learn.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **quantile** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Quantile for bandwidth estimation
  * **n_samples** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Number of samples to use for bandwidth estimation
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]
* **Returns:**
  Cluster assignments for each input vector

### imbed.components.clusterization.minibatch_kmeans_clusterer(vectors, n_clusters=3, batch_size=100, max_iter=100)

Mini-batch K-means clustering using scikit-learn.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **n_clusters** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of clusters
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Size of mini-batches
  * **max_iter** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum number of iterations
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]
* **Returns:**
  Cluster assignments for each input vector

### imbed.components.clusterization.nearest_neighbor_clusterer(vectors, threshold=1.0, n_neighbors=5)

Clustering based on nearest neighbors.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **threshold** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Distance threshold for neighbors
  * **n_neighbors** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of neighbors to consider
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]
* **Returns:**
  Cluster assignments for each input vector

### imbed.components.clusterization.optics_clusterer(vectors, min_samples=5, xi=0.05, min_cluster_size=0.05)

OPTICS clustering using scikit-learn.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **min_samples** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of samples in a neighborhood
  * **xi** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Determines the minimum steepness on the reachability plot
  * **min_cluster_size** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Minimum cluster size as a fraction of total samples
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]
* **Returns:**
  Cluster assignments for each input vector

### imbed.components.clusterization.random_clusterer(vectors, n_clusters=2)

Randomly assigns cluster IDs to vectors.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **n_clusters** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of clusters to create (default: 2)
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]
* **Returns:**
  Randomly assigned cluster IDs

```pycon
>>> random.seed(42)
>>> random_clusterer([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], n_clusters=3)
[2, 1, 0, 2, 2]
```

### imbed.components.clusterization.scan_for_clusterers()

Scan the module for all registered clusterers.
This function simply returns the global clusterers dictionary.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]]]
* **Returns:**
  Dictionary of registered clusterers

### imbed.components.clusterization.spectral_clusterer(vectors, n_clusters=2, affinity='cosine')

Spectral clustering using scikit-learn.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **n_clusters** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of clusters to form
  * **affinity** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Affinity type [‘nearest_neighbors’, ‘rbf’, ‘precomputed’, ‘cosine’]
    (default: ‘cosine’ for semantic embeddings)
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]
* **Returns:**
  Cluster assignments for each input vector

### imbed.components.clusterization.threshold_clusterer(vectors, threshold=1.0, distance_func=<function \_cosine_distance>)

Clusters vectors based on a simple distance threshold.
Vectors within threshold distance are put in the same cluster.

* **Parameters:**
  * **vectors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A sequence of vectors
  * **threshold** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Distance threshold for cluster assignment
  * **distance_func** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]], [`float`](https://docs.python.org/3/builtins/functions.html#float)]) – Function to calculate distance between two vectors
    (default: cosine distance for semantic embeddings)
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]
* **Returns:**
  Cluster IDs for each input vector
