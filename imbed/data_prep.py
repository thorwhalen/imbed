"""Data preparation"""

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np


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

    # If the current number of clusters is less than the desired number, we need to split clusters
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
