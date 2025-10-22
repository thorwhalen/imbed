"""Data preparation"""

from collections.abc import Mapping

from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import pandas as pd


# TODO: Make incremental version
def kmeans_cluster_indices(data_matrix, n_clusters: int = 8, **learner_kwargs):
    kmeans = KMeans(n_clusters=n_clusters, **learner_kwargs)
    kmeans.fit(data_matrix)
    return kmeans.labels_


from typing import Union
from collections.abc import Iterable, Callable

Batch = np.ndarray
DataSrc = Union[Batch, Iterable[Batch], Callable[[], Iterable[Batch]]]


def kmeans_cluster_indices(data_src: DataSrc, n_clusters: int = 8, **learner_kwargs):
    """
    Cluster data using KMeans or MiniBatchKMeans depending on the input type.

    If `data_src` is a numpy array, uses `KMeans`. If `data_src` is an iterable
    of numpy arrays, uses `MiniBatchKMeans` and processes the batches iteratively.

    Parameters:
    - data_src: A numpy array, an iterable of numpy arrays (batches), or a factory thereof.
    - n_clusters: Number of clusters for KMeans.
    - learner_kwargs: Additional arguments for KMeans or MiniBatchKMeans.

    Returns:
    - labels_: Cluster labels for the data.

    Example:
    >>> np.random.seed(0)  # Set seed for reproducibility
    >>> data = np.array([[1, 2], [1, 4], [1, 0], [1, 1], [10, 4], [10, 0]])
    >>> labels = kmeans_cluster_indices(data, n_clusters=2, random_state=42)
    >>> [sorted(data[labels == i].tolist()) for i in np.unique(labels)]
    [[[1, 0], [1, 1], [1, 2], [1, 4]], [[10, 0], [10, 4]]]

    For MiniBatchKMeans case:

    >>> np.random.seed(0)  # Set seed for reproducibility
    >>> get_data_batches = lambda: (data[i:i+2] for i in range(0, len(data), 2))
    >>> labels = kmeans_cluster_indices(get_data_batches, n_clusters=2, random_state=42)
    >>> [sorted(data[labels == i].tolist()) for i in np.unique(labels)]  # doctest: +NORMALIZE_WHITESPACE
    [[[1, 0], [1, 1], [1, 2], [1, 4]], [[10, 0], [10, 4]]]

    """
    if isinstance(data_src, np.ndarray):
        # Use KMeans for a single numpy array
        kmeans = KMeans(n_clusters=n_clusters, **learner_kwargs)
        return kmeans.fit_predict(data_src)
    else:
        minibatch_kmeans = MiniBatchKMeans(n_clusters=n_clusters, **learner_kwargs)
        # At this point, we assume that
        if not callable(data_src):
            iterable_data_src = data_src
            data_src = lambda: iterable_data_src
        _batches = data_src()
        if not isinstance(_batches, Iterable):
            raise ValueError(
                f"data_src must be an (twice traversable) iterable or a factory returnig one: {data_src}"
            )
        for batch in _batches:
            if not isinstance(batch, np.ndarray):
                raise ValueError("All elements of the iterable must be numpy arrays")
            minibatch_kmeans.partial_fit(batch)
        # After fitting, got through the batches again, gathering the predicted labels
        _batches_again = data_src()
        labels_iter = map(minibatch_kmeans.predict, _batches_again)
        return np.concatenate(list(labels_iter))


fibonacci_sequence = [5, 8, 13, 21, 34]


def clusters_df(embeddings, n_clusters=fibonacci_sequence):
    """
    Convenience function to get a table with cluster indices for different cluster sizes.
    """

    # TODO: Move to format transformation logic (with meshed?)
    keys = None
    if isinstance(embeddings, pd.DataFrame):
        keys = embeddings.index.values
        if "embedding" in embeddings.columns:
            embeddings = embeddings.embedding
        embeddings = np.array(embeddings.to_list())
    elif isinstance(embeddings, Mapping):
        keys = list(embeddings.keys())
        embeddings = np.array(list(embeddings.values()))
    else:
        keys = range(len(embeddings))

    def cluster_key_and_indices():
        for k in n_clusters:
            yield f"cluster_{k:02.0f}", kmeans_cluster_indices(embeddings, n_clusters=k)

    return pd.DataFrame(dict(cluster_key_and_indices()), index=keys)


def re_clusters(X, labels, k):
    """
    Re-cluster the dataset X to have exactly k clusters.

    Parameters:
    - X: array-like of shape (n_samples, n_features)
        The input data.
    - labels: array-like of shape (n_samples,)
        Cluster labels for each point in the dataset.
    - k: int
        The desired number of clusters.

    Returns:
    - new_labels: array-like of shape (n_samples,)
        The new cluster labels for the dataset.


    * Handling Cases:
        * Equal Clusters: If the current number of clusters equals k, the function
        returns the original labels.
        * More Clusters: If the current number of clusters is more than k, it merges
        clusters using hierarchical clustering (AgglomerativeClustering).
        * Fewer Clusters: If the current number of clusters is fewer than k, it splits
        the largest cluster iteratively until the desired number of clusters is reached.
    * Merging Clusters:
        * Uses hierarchical clustering on the centroids of the current clusters to
        merge them down to k clusters.
    * Splitting Clusters:
        * Iteratively splits the largest cluster using AgglomerativeClustering until
        the number of clusters reaches k.
    """
    # Number of initial clusters
    initial_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # If the current number of clusters is equal to the desired number of clusters
    if initial_clusters == k:
        return labels

    # If the current number of clusters is more than the desired number, we need to merge clusters
    if initial_clusters > k:
        agg_clustering = AgglomerativeClustering(n_clusters=k)
        cluster_centers = np.array(
            [X[labels == i].mean(axis=0) for i in range(initial_clusters)]
        )
        new_labels = agg_clustering.fit_predict(cluster_centers)
        re_labels = np.copy(labels)
        for old_cluster, new_cluster in enumerate(new_labels):
            re_labels[labels == old_cluster] = new_cluster
        return re_labels

    # If the current number of clusters is less than the desired number, we need to
    # split clusters
    if initial_clusters < k:
        re_labels = np.copy(labels)
        current_max_label = initial_clusters - 1
        while len(set(re_labels)) - (1 if -1 in re_labels else 0) < k:
            largest_cluster = max(set(re_labels), key=list(re_labels).count)
            sub_X = X[re_labels == largest_cluster]
            sub_cluster = AgglomerativeClustering(n_clusters=2).fit(sub_X)
            for sub_label in set(sub_cluster.labels_):
                current_max_label += 1
                re_labels[re_labels == largest_cluster] = np.where(
                    sub_cluster.labels_ == sub_label,
                    current_max_label,
                    re_labels[re_labels == largest_cluster],
                )
        return re_labels


class ImbedArtifactsMixin:
    def segments(self):
        import oa

        df = self.embeddable
        segments = dict(zip(df.doi, df.segment))
        assert len(segments) == len(df), "oops, duplicate DOIs"
        assert all(map(oa.text_is_valid, df.segment)), "some segments are invalid"

        return segments

    def clusters_df(self):
        from imbed.data_prep import clusters_df

        return clusters_df(self.embeddings_df)
