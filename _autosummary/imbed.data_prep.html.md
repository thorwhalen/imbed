# imbed.data_prep

Data preparation

### Functions

| [`clusters_df`](#imbed.data_prep.clusters_df)(embeddings[, n_clusters])          | Convenience function to get a table with cluster indices for different cluster sizes.   |
|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| [`kmeans_cluster_indices`](#imbed.data_prep.kmeans_cluster_indices)(data_src[, n_clusters]) | Cluster data using KMeans or MiniBatchKMeans depending on the input type.               |
| [`re_clusters`](#imbed.data_prep.re_clusters)(X, labels, k)                      | Re-cluster the dataset X to have exactly k clusters.                                    |

### Classes

| `ImbedArtifactsMixin`()   |    |
|---------------------------|----|

### imbed.data_prep.clusters_df(embeddings, n_clusters=[5, 8, 13, 21, 34])

Convenience function to get a table with cluster indices for different cluster sizes.

### imbed.data_prep.kmeans_cluster_indices(data_src, n_clusters=8, \*\*learner_kwargs)

Cluster data using KMeans or MiniBatchKMeans depending on the input type.

If `data_src` is a numpy array, uses `KMeans`. If `data_src` is an iterable
of numpy arrays, uses `MiniBatchKMeans` and processes the batches iteratively.

### Parameters

- data_src: A numpy array, an iterable of numpy arrays (batches), or a factory thereof.
- n_clusters: Number of clusters for KMeans.
- learner_kwargs: Additional arguments for KMeans or MiniBatchKMeans.

### Returns

- ```
  labels_
  ```

  : Cluster labels for the data.

### Example

```pycon
>>> np.random.seed(0)  # Set seed for reproducibility
>>> data = np.array([[1, 2], [1, 4], [1, 0], [1, 1], [10, 4], [10, 0]])
>>> labels = kmeans_cluster_indices(data, n_clusters=2, random_state=42)
>>> [sorted(data[labels == i].tolist()) for i in np.unique(labels)]
[[[1, 0], [1, 1], [1, 2], [1, 4]], [[10, 0], [10, 4]]]
```

For MiniBatchKMeans case:

```pycon
>>> np.random.seed(0)  # Set seed for reproducibility
>>> get_data_batches = lambda: (data[i:i+2] for i in range(0, len(data), 2))
>>> labels = kmeans_cluster_indices(get_data_batches, n_clusters=2, random_state=42)
>>> [sorted(data[labels == i].tolist()) for i in np.unique(labels)]
[[[1, 0], [1, 1], [1, 2], [1, 4]], [[10, 0], [10, 4]]]
```

### imbed.data_prep.re_clusters(X, labels, k)

Re-cluster the dataset X to have exactly k clusters.

### Parameters

- X: array-like of shape (n_samples, n_features)
  : The input data.
- labels: array-like of shape (n_samples,)
  : Cluster labels for each point in the dataset.
- k: int
  : The desired number of clusters.

### Returns

- new_labels: array-like of shape (n_samples,)
  : The new cluster labels for the dataset.

* Handling Cases:
  : * Equal Clusters: If the current number of clusters equals k, the function
      returns the original labels.
    * More Clusters: If the current number of clusters is more than k, it merges
      clusters using hierarchical clustering (AgglomerativeClustering).
    * Fewer Clusters: If the current number of clusters is fewer than k, it splits
      the largest cluster iteratively until the desired number of clusters is reached.
* Merging Clusters:
  : * Uses hierarchical clustering on the centroids of the current clusters to
      merge them down to k clusters.
* Splitting Clusters:
  : * Iteratively splits the largest cluster using AgglomerativeClustering until
      the number of clusters reaches k.
