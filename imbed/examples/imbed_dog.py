"""
Create DOG/ADOG instances from imbed project mall stores.
"""

from typing import (
    Callable,
    Any,
    Dict,
    Sequence,
    NewType,
    Iterable,
    Tuple,
)

from vd.wip.dog import DOG, ADOG
from imbed import imbed_project


# --- Type Definitions ---
Segment = NewType("Segment", str)
Embedding = NewType("Embedding", Sequence[float])
PlanarEmbedding = NewType("PlanarEmbedding", Sequence[float])
Cluster = NewType("Cluster", int)

Segments = Iterable[Segment]
Embeddings = Iterable[Embedding]
PlanarEmbeddings = Iterable[PlanarEmbedding]
Clusters = Iterable[Cluster]

# Callable types for operations
Segmenter = Callable[[Any], Segments]
Embedder = Callable[[Segments], Embeddings]
Planarizer = Callable[[Embeddings], PlanarEmbeddings]
Clusterer = Callable[[Embeddings], Clusters]


def mk_dog_from_mall(mall, *, async_mode: bool = True):
    """
    Create a DOG or ADOG instance from an imbed project mall.

    Args:
        mall: Mall object with stores for segments, embeddings, clusters, planar_embeddings,
              and function implementations (segmenters, embedders, clusterers, planarizers)
        async_mode: If True, create an ADOG (async), otherwise create a DOG (sync)

    Returns:
        DOG or ADOG instance configured with the mall's stores and functions
    """

    # Define operation signatures (abstract function types and their I/O)
    operation_signatures = {
        'segmenter': Segmenter,
        'embedder': Embedder,
        'planarizer': Planarizer,
        'clusterer': Clusterer,
    }

    # Map mall data stores to DOG data store configuration
    data_stores = {
        'segments': {
            'type': Segments,
            'store': mall.segments,
        },
        'embeddings': {
            'type': Embeddings,
            'store': mall.embeddings,
        },
        'planar_embeddings': {
            'type': PlanarEmbeddings,
            'store': mall.planar_embeddings,
        },
        'clusters': {
            'type': Clusters,
            'store': mall.clusters,
        },
    }

    # Map mall function stores to DOG operation implementations
    operation_implementations = {
        'segmenter': dict(mall.segmenters),
        'embedder': dict(mall.embedders),
        'planarizer': dict(mall.planarizers),
        'clusterer': dict(mall.clusterers),
    }

    # Create DOG or ADOG based on async_mode
    if async_mode:
        return ADOG(
            operation_signatures=operation_signatures,
            data_stores=data_stores,
            operation_implementations=operation_implementations,
        )
    else:
        return DOG(
            operation_signatures=operation_signatures,
            data_stores=data_stores,
            operation_implementations=operation_implementations,
        )


# --- Example Usage ---
if __name__ == "__main__":
    # Create a mall with local stores
    mall = imbed_project.get_mall(
        'dog_example', get_project_mall=imbed_project.get_local_mall
    )

    print("=== Creating DOG from Mall ===")
    print(f"Mall keys: {list(mall.keys())}")

    # Create sync DOG
    sync_dog = mk_dog_from_mall(mall, async_mode=False)
    print(f"Sync DOG data stores: {list(sync_dog.data_stores.keys())}")
    print(f"Sync DOG operations: {list(sync_dog.operation_implementations.keys())}")

    # Create async ADOG
    async_dog = mk_dog_from_mall(mall, async_mode=True)
    print(f"Async DOG data stores: {list(async_dog.data_stores.keys())}")
    print(f"Async DOG operations: {list(async_dog.operation_implementations.keys())}")
