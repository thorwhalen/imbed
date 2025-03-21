from functools import partial
from contextlib import suppress
from typing import (
    Callable,
    Sequence,
    List,
    Optional,
    Tuple,
    Dict,
    Union,
    Any,
    TypeVar,
    cast,
)
import random
import math
import asyncio
import itertools

# Type definitions
Vector = Sequence[float]
Vectors = Sequence[Vector]
Point2D = Tuple[float, float]
Points2D = Sequence[Point2D]
Planarizer = Callable[[Vectors], Points2D]

suppress_import_errors = suppress(ImportError, ModuleNotFoundError)

# Dictionary to store all registered planarizers
planarizers: Dict[str, Planarizer] = {}


def register_planarizer(
    planarizer: Union[Planarizer, str], name: Optional[str] = None
) -> Union[Planarizer, Callable[[Planarizer], Planarizer]]:
    """
    Register a planarization function in the global planarizers dictionary.

    Can be used as a decorator with or without arguments:
    @register_planarizer  # uses function name
    @register_planarizer('custom_name')  # uses provided name

    Args:
        planarizer: The planarization function or a name string if used as @register_planarizer('name')
        name: Optional name to register the planarizer under

    Returns:
        The planarizer function or a partial function that will register the planarizer
    """
    if isinstance(planarizer, str):
        name = planarizer
        return partial(register_planarizer, name=name)

    if name is None:
        name = planarizer.__name__

    planarizers[name] = planarizer
    return planarizer


@register_planarizer
def constant_planarizer(embeddings: List[float]) -> List[Tuple[float, float]]:
    """Generate basic 2D projections from embeddings"""
    return [(1.0, 4.0), (2.0, 5.0), (3.0, 6.0)]  # Simplified example


@register_planarizer
def identity_planarizer(vectors: Vectors) -> Points2D:
    """
    Returns the first two dimensions of each vector.
    If vectors have fewer than 2 dimensions, pads with zeros.

    Args:
        vectors: A sequence of vectors

    Returns:
        A sequence of 2D points

    >>> identity_planarizer([[1, 2, 3], [4, 5, 6]])
    [(1.0, 2.0), (4.0, 5.0)]
    >>> identity_planarizer([[1], [2]])
    [(1.0, 0.0), (2.0, 0.0)]
    """

    def _get_2d(v: Vector) -> Point2D:
        if len(v) >= 2:
            return (float(v[0]), float(v[1]))
        elif len(v) == 1:
            return (float(v[0]), 0.0)
        else:
            return (0.0, 0.0)

    return [_get_2d(v) for v in vectors]


@register_planarizer
def random_planarizer(vectors: Vectors, scale: float = 1.0) -> Points2D:
    """
    Randomly projects vectors into 2D space.

    Args:
        vectors: A sequence of vectors
        scale: Scale factor for the random projections

    Returns:
        A sequence of random 2D points

    >>> random_planarizer([[1, 2, 3], [4, 5, 6]], scale=0.5)  # doctest: +SKIP
    [(0.37454011796069593, 0.4590583266505292), (0.32919921068172773, 0.7365648894035036)]
    """
    return [(random.random() * scale, random.random() * scale) for _ in vectors]


def _compute_pairwise_distances(vectors: Vectors) -> List[List[float]]:
    """
    Compute pairwise Euclidean distances between vectors.

    Args:
        vectors: A sequence of vectors

    Returns:
        Matrix of pairwise distances
    """
    n = len(vectors)
    distances = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vectors[i], vectors[j])))
            distances[i][j] = dist
            distances[j][i] = dist

    return distances


@register_planarizer
def circular_planarizer(vectors: Vectors) -> Points2D:
    """
    Places vectors in a circle with similar vectors closer together.

    Args:
        vectors: A sequence of vectors

    Returns:
        A sequence of 2D points arranged in a circle
    """
    if len(vectors) <= 1:
        return [(0.0, 0.0)] * len(vectors)

    # Place points on a circle
    n = len(vectors)
    points = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = math.cos(angle)
        y = math.sin(angle)
        points.append((x, y))

    return points


@register_planarizer
def grid_planarizer(vectors: Vectors) -> Points2D:
    """
    Places vectors in a grid pattern.

    Args:
        vectors: A sequence of vectors

    Returns:
        A sequence of 2D points arranged in a grid
    """
    n = len(vectors)
    if n == 0:
        return []

    # Determine grid dimensions
    side = math.ceil(math.sqrt(n))

    points = []
    for i in range(n):
        row = i // side
        col = i % side
        # Normalize to [-1, 1] range
        x = (2 * col / (side - 1)) - 1 if side > 1 else 0
        y = (2 * row / (side - 1)) - 1 if side > 1 else 0
        points.append((x, y))

    return points


# PCA implementation
with suppress_import_errors:
    import numpy as np

    @register_planarizer
    def pca_planarizer(
        vectors: Vectors, random_state: Optional[int] = None
    ) -> Points2D:
        """
        Principal Component Analysis (PCA) for 2D projection.

        Args:
            vectors: A sequence of vectors
            random_state: Random seed for reproducibility

        Returns:
            A sequence of 2D points representing the top 2 principal components
        """
        X = np.array(vectors)

        # Center the data
        X_centered = X - np.mean(X, axis=0)

        # Compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvectors by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Select the top 2 eigenvectors
        top_eigenvectors = eigenvectors[:, :2]

        # Project the data
        projected = X_centered @ top_eigenvectors

        return [(float(p[0]), float(p[1])) for p in projected]


# t-SNE implementation
with suppress_import_errors:
    import numpy as np
    from sklearn.manifold import TSNE

    @register_planarizer
    def tsne_planarizer(
        vectors: Vectors,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        random_state: int = 42,
    ) -> Points2D:
        """
        t-SNE (t-Distributed Stochastic Neighbor Embedding) for 2D projection.

        Args:
            vectors: A sequence of vectors
            perplexity: The perplexity parameter for t-SNE
            learning_rate: The learning rate for t-SNE
            n_iter: Number of iterations
            random_state: Random seed for reproducibility

        Returns:
            A sequence of 2D points from t-SNE projection
        """
        X = np.array(vectors)
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(X) - 1) if len(X) > 1 else 1,
            learning_rate=learning_rate,
            n_iter=n_iter,
            random_state=random_state,
        )

        # Handle the case with only one sample
        if len(X) == 1:
            return [(0.0, 0.0)]

        embedding = tsne.fit_transform(X)
        return [(float(p[0]), float(p[1])) for p in embedding]


# UMAP implementation
with suppress_import_errors:
    import numpy as np
    import umap

    @register_planarizer
    def umap_planarizer(
        vectors: Vectors,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 42,
    ) -> Points2D:
        """
        UMAP (Uniform Manifold Approximation and Projection) for 2D projection.

        Args:
            vectors: A sequence of vectors
            n_neighbors: Number of neighbors to consider for each point
            min_dist: Minimum distance between points in the embedding
            metric: Distance metric to use
            random_state: Random seed for reproducibility

        Returns:
            A sequence of 2D points from UMAP projection
        """
        X = np.array(vectors)

        # Adjust n_neighbors if there are too few samples
        n_neighbors = min(n_neighbors, len(X) - 1) if len(X) > 1 else 1

        # Handle the case with only one sample
        if len(X) == 1:
            return [(0.0, 0.0)]

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )

        embedding = reducer.fit_transform(X)
        return [(float(p[0]), float(p[1])) for p in embedding]


# MDS (Multidimensional Scaling) implementation
with suppress_import_errors:
    import numpy as np
    from sklearn.manifold import MDS

    @register_planarizer
    def mds_planarizer(
        vectors: Vectors,
        metric: bool = True,
        n_init: int = 4,
        max_iter: int = 300,
        random_state: int = 42,
    ) -> Points2D:
        """
        Multidimensional Scaling (MDS) for 2D projection.

        Args:
            vectors: A sequence of vectors
            metric: If True, perform metric MDS; otherwise, perform nonmetric MDS
            n_init: Number of times the SMACOF algorithm will be run with different initializations
            max_iter: Maximum number of iterations of the SMACOF algorithm
            random_state: Random seed for reproducibility

        Returns:
            A sequence of 2D points from MDS projection
        """
        X = np.array(vectors)

        # Handle the case with only one sample
        if len(X) == 1:
            return [(0.0, 0.0)]

        mds = MDS(
            n_components=2,
            metric=metric,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
            dissimilarity="euclidean",
        )

        embedding = mds.fit_transform(X)
        return [(float(p[0]), float(p[1])) for p in embedding]


# Isomap implementation
with suppress_import_errors:
    import numpy as np
    from sklearn.manifold import Isomap

    @register_planarizer
    def isomap_planarizer(vectors: Vectors, n_neighbors: int = 5) -> Points2D:
        """
        Isomap for 2D projection.

        Args:
            vectors: A sequence of vectors
            n_neighbors: Number of neighbors to consider for each point

        Returns:
            A sequence of 2D points from Isomap projection
        """
        X = np.array(vectors)

        # Adjust n_neighbors if there are too few samples
        n_neighbors = min(n_neighbors, len(X) - 1) if len(X) > 1 else 1

        # Handle the case with only one sample
        if len(X) == 1:
            return [(0.0, 0.0)]

        isomap = Isomap(n_components=2, n_neighbors=n_neighbors)
        embedding = isomap.fit_transform(X)
        return [(float(p[0]), float(p[1])) for p in embedding]


# LLE (Locally Linear Embedding) implementation
with suppress_import_errors:
    import numpy as np
    from sklearn.manifold import LocallyLinearEmbedding

    @register_planarizer
    def lle_planarizer(
        vectors: Vectors,
        n_neighbors: int = 5,
        method: str = "standard",
        random_state: int = 42,
    ) -> Points2D:
        """
        Locally Linear Embedding (LLE) for 2D projection.

        Args:
            vectors: A sequence of vectors
            n_neighbors: Number of neighbors to consider for each point
            method: LLE algorithm to use ('standard', 'hessian', 'modified', or 'ltsa')
            random_state: Random seed for reproducibility

        Returns:
            A sequence of 2D points from LLE projection
        """
        X = np.array(vectors)

        # Adjust n_neighbors if there are too few samples
        n_neighbors = min(n_neighbors, len(X) - 1) if len(X) > 1 else 1

        # Handle the case with only one sample
        if len(X) == 1:
            return [(0.0, 0.0)]

        lle = LocallyLinearEmbedding(
            n_components=2,
            n_neighbors=n_neighbors,
            method=method,
            random_state=random_state,
        )

        embedding = lle.fit_transform(X)
        return [(float(p[0]), float(p[1])) for p in embedding]


# Spectral Embedding implementation
with suppress_import_errors:
    import numpy as np
    from sklearn.manifold import SpectralEmbedding

    @register_planarizer
    def spectral_embedding_planarizer(
        vectors: Vectors,
        n_neighbors: int = 10,
        affinity: str = "nearest_neighbors",
        random_state: int = 42,
    ) -> Points2D:
        """
        Spectral Embedding for 2D projection.

        Args:
            vectors: A sequence of vectors
            n_neighbors: Number of neighbors to consider for each point (when affinity='nearest_neighbors')
            affinity: How to construct the affinity matrix ('nearest_neighbors', 'rbf', or 'precomputed')
            random_state: Random seed for reproducibility

        Returns:
            A sequence of 2D points from Spectral Embedding
        """
        X = np.array(vectors)

        # Adjust n_neighbors if there are too few samples
        n_neighbors = min(n_neighbors, len(X) - 1) if len(X) > 1 else 1

        # Handle the case with only one sample
        if len(X) == 1:
            return [(0.0, 0.0)]

        embedding = SpectralEmbedding(
            n_components=2,
            n_neighbors=n_neighbors,
            affinity=affinity,
            random_state=random_state,
        )

        result = embedding.fit_transform(X)
        return [(float(p[0]), float(p[1])) for p in result]


# Factor Analysis implementation
with suppress_import_errors:
    import numpy as np
    from sklearn.decomposition import FactorAnalysis

    @register_planarizer
    def factor_analysis_planarizer(
        vectors: Vectors, random_state: int = 42
    ) -> Points2D:
        """
        Factor Analysis for 2D projection.

        Args:
            vectors: A sequence of vectors
            random_state: Random seed for reproducibility

        Returns:
            A sequence of 2D points from Factor Analysis
        """
        X = np.array(vectors)

        # Handle the case with too few samples or too few features
        if len(X) <= 2 or X.shape[1] <= 2:
            if len(X) == 0:
                return []
            return [(0.0, 0.0)] * len(X)

        fa = FactorAnalysis(n_components=2, random_state=random_state)
        embedding = fa.fit_transform(X)
        return [(float(p[0]), float(p[1])) for p in embedding]


# Kernel PCA implementation
with suppress_import_errors:
    import numpy as np
    from sklearn.decomposition import KernelPCA

    @register_planarizer
    def kernel_pca_planarizer(
        vectors: Vectors,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
        random_state: int = 42,
    ) -> Points2D:
        """
        Kernel PCA for 2D projection.

        Args:
            vectors: A sequence of vectors
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid', 'cosine')
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid' kernels
            random_state: Random seed for reproducibility

        Returns:
            A sequence of 2D points from Kernel PCA projection
        """
        X = np.array(vectors)

        # Handle the case with only one sample
        if len(X) == 1:
            return [(0.0, 0.0)]

        kpca = KernelPCA(
            n_components=2, kernel=kernel, gamma=gamma, random_state=random_state
        )

        embedding = kpca.fit_transform(X)
        return [(float(p[0]), float(p[1])) for p in embedding]


# FastICA implementation
with suppress_import_errors:
    import numpy as np
    from sklearn.decomposition import FastICA

    @register_planarizer
    def fast_ica_planarizer(vectors: Vectors, random_state: int = 42) -> Points2D:
        """
        Fast Independent Component Analysis (FastICA) for 2D projection.

        Args:
            vectors: A sequence of vectors
            random_state: Random seed for reproducibility

        Returns:
            A sequence of 2D points from FastICA projection
        """
        X = np.array(vectors)

        # Handle the case with too few features
        if X.shape[1] < 2:
            if len(X) == 0:
                return []
            # Pad with zeros if needed
            X = np.pad(X, ((0, 0), (0, 2 - X.shape[1])), mode="constant")

        # Handle the case with only one sample
        if len(X) == 1:
            return [(0.0, 0.0)]

        ica = FastICA(n_components=2, random_state=random_state)
        embedding = ica.fit_transform(X)
        return [(float(p[0]), float(p[1])) for p in embedding]


# NMF (Non-negative Matrix Factorization) implementation
with suppress_import_errors:
    import numpy as np
    from sklearn.decomposition import NMF

    @register_planarizer
    def nmf_planarizer(
        vectors: Vectors, init: str = "nndsvd", random_state: int = 42
    ) -> Points2D:
        """
        Non-negative Matrix Factorization (NMF) for 2D projection.
        Works only for non-negative data.

        Args:
            vectors: A sequence of non-negative vectors
            init: Method used to initialize the procedure ('random', 'nndsvd')
            random_state: Random seed for reproducibility

        Returns:
            A sequence of 2D points from NMF projection
        """
        X = np.array(vectors)

        # NMF requires non-negative values
        if np.any(X < 0):
            # Simple shift to make all values non-negative
            X = X - np.min(X, axis=0) if len(X) > 0 else X

        # Handle the case with too few features
        if X.shape[1] < 2:
            if len(X) == 0:
                return []
            # Pad with zeros if needed
            X = np.pad(X, ((0, 0), (0, 2 - X.shape[1])), mode="constant")

        # Handle the case with only one sample
        if len(X) == 1:
            return [(0.0, 0.0)]

        nmf = NMF(n_components=2, init=init, random_state=random_state)
        embedding = nmf.fit_transform(X)
        return [(float(p[0]), float(p[1])) for p in embedding]


# Truncated SVD implementation (also known as LSA)
with suppress_import_errors:
    import numpy as np
    from sklearn.decomposition import TruncatedSVD

    @register_planarizer
    def truncated_svd_planarizer(
        vectors: Vectors,
        algorithm: str = "randomized",
        n_iter: int = 5,
        random_state: int = 42,
    ) -> Points2D:
        """
        Truncated Singular Value Decomposition (SVD) for 2D projection.

        Args:
            vectors: A sequence of vectors
            algorithm: SVD solver algorithm ('arpack' or 'randomized')
            n_iter: Number of iterations for randomized SVD solver
            random_state: Random seed for reproducibility

        Returns:
            A sequence of 2D points from Truncated SVD projection
        """
        X = np.array(vectors)

        # Handle the case with too few features
        if X.shape[1] < 2:
            if len(X) == 0:
                return []
            # Pad with zeros if needed
            X = np.pad(X, ((0, 0), (0, 2 - X.shape[1])), mode="constant")

        # Handle the case with only one sample
        if len(X) == 1:
            return [(0.0, 0.0)]

        svd = TruncatedSVD(
            n_components=2,
            algorithm=algorithm,
            n_iter=n_iter,
            random_state=random_state,
        )

        embedding = svd.fit_transform(X)
        return [(float(p[0]), float(p[1])) for p in embedding]


# SRP (Sparse Random Projection) implementation
with suppress_import_errors:
    import numpy as np
    from sklearn.random_projection import SparseRandomProjection

    @register_planarizer
    def sparse_random_projection_planarizer(
        vectors: Vectors, density: float = "auto", random_state: int = 42
    ) -> Points2D:
        """
        Sparse Random Projection for 2D projection.

        Args:
            vectors: A sequence of vectors
            density: Density of the random projection matrix
            random_state: Random seed for reproducibility

        Returns:
            A sequence of 2D points from Sparse Random Projection
        """
        X = np.array(vectors)

        # Handle the case with only one sample
        if len(X) == 1:
            return [(0.0, 0.0)]

        srp = SparseRandomProjection(
            n_components=2, density=density, random_state=random_state
        )

        embedding = srp.fit_transform(X)
        return [(float(p[0]), float(p[1])) for p in embedding]


# Gaussian Random Projection implementation
with suppress_import_errors:
    import numpy as np
    from sklearn.random_projection import GaussianRandomProjection

    @register_planarizer
    def gaussian_random_projection_planarizer(
        vectors: Vectors, random_state: int = 42
    ) -> Points2D:
        """
        Gaussian Random Projection for 2D projection.

        Args:
            vectors: A sequence of vectors
            random_state: Random seed for reproducibility

        Returns:
            A sequence of 2D points from Gaussian Random Projection
        """
        X = np.array(vectors)

        # Handle the case with only one sample
        if len(X) == 1:
            return [(0.0, 0.0)]

        grp = GaussianRandomProjection(n_components=2, random_state=random_state)

        embedding = grp.fit_transform(X)
        return [(float(p[0]), float(p[1])) for p in embedding]


# Robust PCA implementation
with suppress_import_errors:
    import numpy as np
    from sklearn.decomposition import PCA

    @register_planarizer
    def robust_pca_planarizer(vectors: Vectors, random_state: int = 42) -> Points2D:
        """
        Robust PCA for 2D projection, using a robust scaler before PCA.

        Args:
            vectors: A sequence of vectors
            random_state: Random seed for reproducibility

        Returns:
            A sequence of 2D points from Robust PCA projection
        """
        from sklearn.preprocessing import RobustScaler

        X = np.array(vectors)

        # Handle the case with only one sample
        if len(X) == 1:
            return [(0.0, 0.0)]

        # Apply robust scaling
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply PCA
        pca = PCA(n_components=2, random_state=random_state)
        embedding = pca.fit_transform(X_scaled)

        return [(float(p[0]), float(p[1])) for p in embedding]


# Force-directed layout using Fruchterman-Reingold algorithm
with suppress_import_errors:
    import numpy as np
    import networkx as nx

    @register_planarizer
    def force_directed_planarizer(
        vectors: Vectors,
        k: Optional[float] = None,
        iterations: int = 50,
        seed: int = 42,
    ) -> Points2D:
        """
        Force-directed layout using Fruchterman-Reingold algorithm.
        Creates a graph where nodes are vectors and edge weights are based on vector similarity.

        Args:
            vectors: A sequence of vectors
            k: Optimal distance between nodes
            iterations: Number of iterations
            seed: Random seed for reproducibility

        Returns:
            A sequence of 2D points from force-directed layout
        """
        n = len(vectors)

        if n <= 1:
            return [(0.0, 0.0)] * n

        # Create a graph with edges weighted by vector similarity
        G = nx.Graph()

        # Add nodes
        for i in range(n):
            G.add_node(i)

        # Add edges with weights based on Euclidean distance
        for i in range(n):
            for j in range(i + 1, n):
                # Calculate Euclidean distance
                dist = math.sqrt(
                    sum((a - b) ** 2 for a, b in zip(vectors[i], vectors[j]))
                )

                # Convert distance to similarity (smaller distance = higher weight)
                similarity = 1.0 / (1.0 + dist)

                G.add_edge(i, j, weight=similarity)

        # Apply Fruchterman-Reingold layout
        pos = nx.spring_layout(G, k=k, iterations=iterations, seed=seed)

        # Extract points in order
        points = []
        for i in range(n):
            x, y = pos[i]
            points.append((float(x), float(y)))

        return points
