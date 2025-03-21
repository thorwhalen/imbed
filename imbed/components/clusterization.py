from functools import partial
from contextlib import suppress
from typing import (
    Callable,
    Sequence,
    List,
    Optional,
    Any,
    Union,
    Dict,
    TypeVar,
    Iterable,
    cast,
)
from collections.abc import Callable as CallableABC
import random
import math
import itertools

# Type definitions for better static analysis
Vector = Sequence[float]
Vectors = Sequence[Vector]
ClusterIDs = Sequence[int]
Clusterer = Callable[[Vectors], ClusterIDs]

suppress_import_errors = partial(suppress, ImportError, ModuleNotFoundError)

# Dictionary to store all registered clusterers
clusterers: Dict[str, Clusterer] = {}


def register_clusterer(
    clusterer: Union[Clusterer, str], name: Optional[str] = None
) -> Union[Clusterer, Callable[[Clusterer], Clusterer]]:
    """
    Register a clustering function in the global clusterers dictionary.

    Can be used as a decorator with or without arguments:
    @register_clusterer  # uses function name
    @register_clusterer('custom_name')  # uses provided name

    Args:
        clusterer: The clustering function or a name string if used as @register_clusterer('name')
        name: Optional name to register the clusterer under

    Returns:
        The clusterer function or a partial function that will register the clusterer
    """
    if isinstance(clusterer, str):
        name = clusterer
        return partial(register_clusterer, name=name)

    if name is None:
        name = clusterer.__name__

    clusterers[name] = clusterer
    return clusterer


@register_clusterer
def constant_clusterer(vectors: Vectors) -> ClusterIDs:
    """
    Returns alternating [0, 1, 0, 1, ...] cluster IDs regardless of input vectors.
    This is just for testing purposes.

    Args:
        vectors: A sequence of vectors

    Returns:
        A sequence of cluster IDs (alternating 0 and 1)

    >>> constant_clusterer([[1, 2], [3, 4], [5, 6], [7, 8]])  # doctest: +SKIP
    [0, 1, 0, 1]
    >>> constant_clusterer([[1, 2], [3, 4], [5, 6]])  # doctest: +SKIP
    [0, 1, 0]
    """
    return list(itertools.islice(itertools.cycle([0, 1]), len(vectors)))


@register_clusterer
def random_clusterer(vectors: Vectors, n_clusters: int = 2) -> ClusterIDs:
    """
    Randomly assigns cluster IDs to vectors.

    Args:
        vectors: A sequence of vectors
        n_clusters: Number of clusters to create (default: 2)

    Returns:
        Randomly assigned cluster IDs

    >>> random.seed(42)
    >>> random_clusterer([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], n_clusters=3)  # doctest: +SKIP
    [2, 1, 0, 2, 2]
    """
    return [random.randrange(n_clusters) for _ in range(len(vectors))]


def _euclidean_distance(v1: Vector, v2: Vector) -> float:
    """
    Calculate Euclidean distance between two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Euclidean distance between vectors
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


@register_clusterer
def threshold_clusterer(
    vectors: Vectors,
    threshold: float = 1.0,
    distance_func: Callable[[Vector, Vector], float] = _euclidean_distance,
) -> ClusterIDs:
    """
    Clusters vectors based on a simple distance threshold.
    Vectors within threshold distance are put in the same cluster.

    Args:
        vectors: A sequence of vectors
        threshold: Distance threshold for cluster assignment
        distance_func: Function to calculate distance between two vectors

    Returns:
        Cluster IDs for each input vector
    """
    if not vectors:
        return []

    clusters = [0]  # First vector goes to cluster 0
    for i in range(1, len(vectors)):
        # Check distances to previously assigned vectors
        min_dist = float("inf")
        closest_cluster = -1

        for j in range(i):
            dist = distance_func(vectors[i], vectors[j])
            if dist < min_dist:
                min_dist = dist
                closest_cluster = clusters[j]

        # If close enough to an existing cluster, join it; otherwise, create a new one
        if min_dist <= threshold:
            clusters.append(closest_cluster)
        else:
            clusters.append(max(clusters) + 1 if clusters else 0)

    return clusters


# K-means clustering
with suppress_import_errors():
    import numpy as np

    @register_clusterer
    def kmeans_clusterer(
        vectors: Vectors,
        n_clusters: int = 3,
        max_iter: int = 100,
        tol: float = 1e-4,
        seed: Optional[int] = None,
    ) -> ClusterIDs:
        """
        K-means clustering implementation.

        Args:
            vectors: A sequence of vectors
            n_clusters: Number of clusters to form
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            seed: Random seed for reproducibility

        Returns:
            Cluster assignments for each input vector
        """
        X = np.array(vectors)
        n_samples, n_features = X.shape

        # Initialize centroids
        if seed is not None:
            np.random.seed(seed)

        indices = np.random.choice(n_samples, n_clusters, replace=False)
        centroids = X[indices]

        # Iterate until convergence or max iterations
        for _ in range(max_iter):
            # Assign clusters
            distances = np.array(
                [[np.linalg.norm(x - centroid) for centroid in centroids] for x in X]
            )
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array(
                [X[labels == i].mean(axis=0) for i in range(n_clusters)]
            )

            # Check for convergence
            if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
                break

            centroids = new_centroids

        return labels.tolist()


# DBSCAN clustering
with suppress_import_errors():
    import numpy as np
    from sklearn.cluster import DBSCAN

    @register_clusterer
    def dbscan_clusterer(
        vectors: Vectors, eps: float = 0.5, min_samples: int = 5
    ) -> ClusterIDs:
        """
        DBSCAN clustering using scikit-learn.

        Args:
            vectors: A sequence of vectors
            eps: The maximum distance between two samples for them to be considered neighbors
            min_samples: The number of samples in a neighborhood for a point to be considered a core point

        Returns:
            Cluster IDs with -1 representing noise points
        """
        X = np.array(vectors)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        return dbscan.fit_predict(X).tolist()


# Hierarchical clustering
with suppress_import_errors():
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    @register_clusterer
    def hierarchical_clusterer(
        vectors: Vectors, n_clusters: int = 2, linkage: str = "ward"
    ) -> ClusterIDs:
        """
        Hierarchical clustering using scikit-learn.

        Args:
            vectors: A sequence of vectors
            n_clusters: Number of clusters to form
            linkage: Linkage criterion ['ward', 'complete', 'average', 'single']

        Returns:
            Cluster assignments for each input vector
        """
        X = np.array(vectors)
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        return model.fit_predict(X).tolist()


# Mean-shift clustering
with suppress_import_errors():
    import numpy as np
    from sklearn.cluster import MeanShift, estimate_bandwidth

    @register_clusterer
    def meanshift_clusterer(
        vectors: Vectors, quantile: float = 0.3, n_samples: Optional[int] = None
    ) -> ClusterIDs:
        """
        Mean-shift clustering using scikit-learn.

        Args:
            vectors: A sequence of vectors
            quantile: Quantile for bandwidth estimation
            n_samples: Number of samples to use for bandwidth estimation

        Returns:
            Cluster assignments for each input vector
        """
        X = np.array(vectors)
        bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=n_samples)
        model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        return model.fit_predict(X).tolist()


# Spectral clustering
with suppress_import_errors():
    import numpy as np
    from sklearn.cluster import SpectralClustering

    @register_clusterer
    def spectral_clusterer(
        vectors: Vectors, n_clusters: int = 2, affinity: str = "rbf"
    ) -> ClusterIDs:
        """
        Spectral clustering using scikit-learn.

        Args:
            vectors: A sequence of vectors
            n_clusters: Number of clusters to form
            affinity: Affinity type ['nearest_neighbors', 'rbf', 'precomputed']

        Returns:
            Cluster assignments for each input vector
        """
        X = np.array(vectors)
        model = SpectralClustering(
            n_clusters=n_clusters, affinity=affinity, random_state=42
        )
        return model.fit_predict(X).tolist()


# Gaussian Mixture Model clustering
with suppress_import_errors():
    import numpy as np
    from sklearn.mixture import GaussianMixture

    @register_clusterer
    def gmm_clusterer(
        vectors: Vectors,
        n_components: int = 2,
        covariance_type: str = "full",
        random_state: int = 42,
    ) -> ClusterIDs:
        """
        Gaussian Mixture Model clustering using scikit-learn.

        Args:
            vectors: A sequence of vectors
            n_components: Number of mixture components
            covariance_type: Covariance parameter type
            random_state: Random state for reproducibility

        Returns:
            Cluster assignments for each input vector
        """
        X = np.array(vectors)
        model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
        )
        return model.fit_predict(X).tolist()


# Affinity Propagation clustering
with suppress_import_errors():
    import numpy as np
    from sklearn.cluster import AffinityPropagation

    @register_clusterer
    def affinity_propagation_clusterer(
        vectors: Vectors, damping: float = 0.5, max_iter: int = 200
    ) -> ClusterIDs:
        """
        Affinity Propagation clustering using scikit-learn.

        Args:
            vectors: A sequence of vectors
            damping: Damping factor
            max_iter: Maximum number of iterations

        Returns:
            Cluster assignments for each input vector
        """
        X = np.array(vectors)
        model = AffinityPropagation(damping=damping, max_iter=max_iter, random_state=42)
        return model.fit_predict(X).tolist()


# OPTICS clustering
with suppress_import_errors():
    import numpy as np
    from sklearn.cluster import OPTICS

    @register_clusterer
    def optics_clusterer(
        vectors: Vectors,
        min_samples: int = 5,
        xi: float = 0.05,
        min_cluster_size: float = 0.05,
    ) -> ClusterIDs:
        """
        OPTICS clustering using scikit-learn.

        Args:
            vectors: A sequence of vectors
            min_samples: Number of samples in a neighborhood
            xi: Determines the minimum steepness on the reachability plot
            min_cluster_size: Minimum cluster size as a fraction of total samples

        Returns:
            Cluster assignments for each input vector
        """
        X = np.array(vectors)
        model = OPTICS(
            min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size
        )
        return model.fit_predict(X).tolist()


# Birch clustering
with suppress_import_errors():
    import numpy as np
    from sklearn.cluster import Birch

    @register_clusterer
    def birch_clusterer(
        vectors: Vectors,
        n_clusters: int = 3,
        threshold: float = 0.5,
        branching_factor: int = 50,
    ) -> ClusterIDs:
        """
        Birch clustering using scikit-learn.

        Args:
            vectors: A sequence of vectors
            n_clusters: Number of clusters
            threshold: The radius of the subcluster for a sample to be added
            branching_factor: Maximum number of CF subclusters in each node

        Returns:
            Cluster assignments for each input vector
        """
        X = np.array(vectors)
        model = Birch(
            n_clusters=n_clusters,
            threshold=threshold,
            branching_factor=branching_factor,
        )
        return model.fit_predict(X).tolist()


# Mini-batch K-means
with suppress_import_errors():
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    @register_clusterer
    def minibatch_kmeans_clusterer(
        vectors: Vectors,
        n_clusters: int = 3,
        batch_size: int = 100,
        max_iter: int = 100,
    ) -> ClusterIDs:
        """
        Mini-batch K-means clustering using scikit-learn.

        Args:
            vectors: A sequence of vectors
            n_clusters: Number of clusters
            batch_size: Size of mini-batches
            max_iter: Maximum number of iterations

        Returns:
            Cluster assignments for each input vector
        """
        X = np.array(vectors)
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            max_iter=max_iter,
            random_state=42,
        )
        return model.fit_predict(X).tolist()


# UMAP + HDBSCAN clustering (commonly used for single-cell data, embeddings, etc.)
with suppress_import_errors():
    import numpy as np
    import umap
    import hdbscan

    @register_clusterer
    def umap_hdbscan_clusterer(
        vectors: Vectors,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        min_cluster_size: int = 15,
        min_samples: int = 5,
    ) -> ClusterIDs:
        """
        UMAP dimensionality reduction followed by HDBSCAN clustering.

        Args:
            vectors: A sequence of vectors
            n_neighbors: UMAP neighbors parameter
            min_dist: UMAP minimum distance parameter
            min_cluster_size: HDBSCAN minimum cluster size
            min_samples: HDBSCAN minimum samples parameter

        Returns:
            Cluster assignments for each input vector
        """
        X = np.array(vectors)

        # First reduce dimensionality with UMAP
        reducer = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42
        )
        embedding = reducer.fit_transform(X)

        # Then cluster with HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples
        )
        return clusterer.fit_predict(embedding).tolist()


# Nearest-neighbor based clustering
with suppress_import_errors():
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    @register_clusterer
    def nearest_neighbor_clusterer(
        vectors: Vectors, threshold: float = 1.0, n_neighbors: int = 5
    ) -> ClusterIDs:
        """
        Clustering based on nearest neighbors.

        Args:
            vectors: A sequence of vectors
            threshold: Distance threshold for neighbors
            n_neighbors: Number of neighbors to consider

        Returns:
            Cluster assignments for each input vector
        """
        X = np.array(vectors)
        n_samples = X.shape[0]

        # Compute nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=min(n_neighbors + 1, n_samples)).fit(X)
        distances, indices = nbrs.kneighbors(X)

        # Create adjacency matrix
        adjacency = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j, dist in zip(indices[i][1:], distances[i][1:]):  # Skip self
                if dist <= threshold:
                    adjacency[i, j] = 1
                    adjacency[j, i] = 1  # Make it symmetric

        # Assign cluster IDs based on connected components
        visited = [False] * n_samples
        cluster_ids = [-1] * n_samples
        current_cluster = 0

        def _dfs(node, cluster):
            visited[node] = True
            cluster_ids[node] = cluster
            for neighbor in range(n_samples):
                if adjacency[node, neighbor] == 1 and not visited[neighbor]:
                    _dfs(neighbor, cluster)

        for i in range(n_samples):
            if not visited[i]:
                _dfs(i, current_cluster)
                current_cluster += 1

        return cluster_ids


# Simple Bisecting K-means implementation
with suppress_import_errors():
    import numpy as np

    @register_clusterer
    def bisecting_kmeans_clusterer(
        vectors: Vectors, n_clusters: int = 3, max_iter: int = 100
    ) -> ClusterIDs:
        """
        Bisecting K-means clustering implementation.

        Args:
            vectors: A sequence of vectors
            n_clusters: Number of clusters to form
            max_iter: Maximum number of iterations per bisection

        Returns:
            Cluster assignments for each input vector
        """

        def _kmeans_single(X, k=2, max_iter=100):
            """Helper function to perform a single k-means clustering step."""
            n_samples = X.shape[0]
            if n_samples <= k:
                return np.arange(n_samples)

            # Initialize centroids
            indices = np.random.choice(n_samples, k, replace=False)
            centroids = X[indices]

            labels = np.zeros(n_samples, dtype=int)

            for _ in range(max_iter):
                # Assign clusters
                distances = np.array(
                    [
                        [np.linalg.norm(x - centroid) for centroid in centroids]
                        for x in X
                    ]
                )
                new_labels = np.argmin(distances, axis=1)

                # Check for convergence
                if np.all(new_labels == labels):
                    break

                labels = new_labels

                # Update centroids
                for i in range(k):
                    if np.sum(labels == i) > 0:
                        centroids[i] = X[labels == i].mean(axis=0)

            return labels

        X = np.array(vectors)
        n_samples = X.shape[0]

        if n_clusters <= 1 or n_samples <= n_clusters:
            return [0] * n_samples if n_samples > 0 else []

        # Start with all samples in one cluster
        current_labels = np.zeros(n_samples, dtype=int)
        clusters = {0: np.arange(n_samples)}

        # Bisect until we have enough clusters
        while len(clusters) < n_clusters:
            # Find the largest cluster to bisect
            largest_cluster = max(clusters.items(), key=lambda x: len(x[1]))
            cluster_id, cluster_indices = largest_cluster

            # Skip if the cluster has only one point
            if len(cluster_indices) <= 1:
                break

            # Bisect this cluster
            sub_labels = _kmeans_single(X[cluster_indices], k=2, max_iter=max_iter)

            # Remove the original cluster
            del clusters[cluster_id]

            # Create two new clusters
            new_cluster_id1 = len(clusters)
            new_cluster_id2 = len(clusters) + 1

            clusters[new_cluster_id1] = cluster_indices[sub_labels == 0]
            clusters[new_cluster_id2] = cluster_indices[sub_labels == 1]

            # Update the labels
            current_labels[cluster_indices[sub_labels == 0]] = new_cluster_id1
            current_labels[cluster_indices[sub_labels == 1]] = new_cluster_id2

        return current_labels.tolist()


def scan_for_clusterers() -> Dict[str, Clusterer]:
    """
    Scan the module for all registered clusterers.
    This function simply returns the global clusterers dictionary.

    Returns:
        Dictionary of registered clusterers
    """
    return dict(clusterers)


def get_clusterer(name: str) -> Optional[Clusterer]:
    """
    Get a clusterer by name.

    Args:
        name: Name of the clusterer

    Returns:
        The clusterer function if found, None otherwise

    >>> get_clusterer('constant_clusterer') == constant_clusterer  # doctest: +SKIP
    True
    >>> get_clusterer('nonexistent_clusterer') is None
    True
    """
    return clusterers.get(name)


def list_available_clusterers() -> List[str]:
    """
    Return a list of names of all available clusterers.

    Returns:
        List of clusterer names

    >>> 'constant_clusterer' in list_available_clusterers()
    True
    """
    return list(clusterers.keys())
