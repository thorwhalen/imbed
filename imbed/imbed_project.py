"""Project interface for text embedding system.

This module provides the core Project class that manages segments, embeddings,
planarizations, and clusterings with automatic invalidation and status tracking.
"""

from typing import Optional, Any, Iterator, Callable, Sequence, TypeAlias
from dataclasses import dataclass, field
from collections.abc import MutableMapping, Mapping
from datetime import datetime

from au import async_compute, FileSystemStore, ProcessBackend, ComputationHandle

from imbed.imbed_types import (
    Segment, SegmentKey, SegmentMapping,
    Vector, Vectors, VectorMapping,
    PlanarVectorMapping,
)

ComponentRegistry: TypeAlias = MutableMapping[str, Callable]
StoreFactory: TypeAlias = Callable[[], MutableMapping]

# --- AU async embedding setup ---
_embeddings_store = FileSystemStore("/tmp/embeddings", ttl_seconds=3600)
_embeddings_backend = ProcessBackend(_embeddings_store)

def make_async_embedder(embedder: Callable):
    """Wrap an embedder function as an async AU computation."""
    @async_compute(store=_embeddings_store, backend=_embeddings_backend)
    def _async_embed(segments: SegmentMapping):
        return embedder(segments)
    return _async_embed

@dataclass
class Project:
    """Central project interface - facade for all operations.
    
    Manages segments, embeddings, planarizations, and clusterings with
    automatic computation and invalidation cascade.
    """
    id: str
    
    # Storage interfaces - injected dependencies
    segments: MutableMapping[SegmentKey, Segment]
    vectors: dict[SegmentKey, Vector]
    planar_coords: MutableMapping[str, PlanarVectorMapping]
    cluster_indices: MutableMapping[str, Mapping[SegmentKey, int]]
    
    # Component registries
    embedders: ComponentRegistry
    planarizers: ComponentRegistry
    clusterers: ComponentRegistry
    
    # Configuration
    default_embedder: str = "default"
    _invalidation_cascade: bool = True
    _auto_compute_embeddings: bool = True
    _vector_handles: dict[str, ComputationHandle] = field(default_factory=dict)

    def add_segments(self, segments: SegmentMapping) -> list[SegmentKey]:
        """Add segments and trigger embedding computation.
        
        Args:
            segments: Mapping of segment keys to segment text
            
        Returns:
            List of segment keys that were added
        """
        # Update segments
        self.segments.update(segments)
        
        # Invalidate dependent computations
        if self._invalidation_cascade:
            self.planar_coords.clear()
            self.cluster_indices.clear()
        
        # Schedule embedding computation
        if self._auto_compute_embeddings:
            embedder = self.embedders[self.default_embedder]
            async_embed = make_async_embedder(embedder)
            handle = async_embed(segments)
            for key in segments:
                self._vector_handles[key] = handle
        
        return list(segments.keys())
    
    def wait_for_embeddings(self, segment_keys: Optional[list[SegmentKey]] = None, timeout: float = 10.0) -> bool:
        """Wait for embeddings to complete and store them in self.vectors."""
        if not self._vector_handles:
            return True
        # Wait for any handle (all keys in a batch share the same handle)
        handle = next(iter(self._vector_handles.values()))
        try:
            result = handle.get_result(timeout=timeout)
            self.vectors.update(result)
            self._vector_handles.clear()
            return True
        except Exception:
            return False

    @property
    def valid_vectors(self) -> VectorMapping:
        """Get all valid computed vectors"""
        return dict(self.vectors)
    
    def get_vectors(self, segment_keys: Optional[list[SegmentKey]] = None) -> list[Vector]:
        """Get vectors for specified segments (or all if None)"""
        if segment_keys is None:
            segment_keys = list(self.segments.keys())
        return [self.vectors[key] for key in segment_keys if key in self.vectors]

    def compute(self, component_type: str, name: str, save_key: str = None, data=None):
        """
        Run a planarizer or clusterer and store the result.
        component_type: 'planarizer' or 'clusterer'
        name: component name in the registry
        save_key: key to store result under (default: name + _ + timestamp)
        data: override input data (default: self.vectors)
        """
        if component_type == "planarizer":
            registry = self.planarizers
            store = self.planar_coords
        elif component_type == "clusterer":
            registry = self.clusterers
            store = self.cluster_indices
        else:
            raise ValueError(f"Unknown component_type: {component_type}")

        if name not in registry:
            raise KeyError(f"{component_type} '{name}' not found")

        if data is None:
            data = self.vectors

        result = registry[name](data)
        if save_key is None:
            import time
            save_key = f"{name}_{int(time.time()*1000)}"
        store[save_key] = result
        return save_key


class Projects(MutableMapping[str, Project]):
    """Container for projects with MutableMapping interface.
    
    >>> projects = Projects()
    >>> p = Project(id="test", segments={}, vectors=ComputeStore(), 
    ...             planar_coords={}, cluster_indices={},
    ...             embedders={}, planarizers={}, clusterers={})
    >>> projects["test"] = p
    >>> list(projects)
    ['test']
    >>> projects["test"].id
    'test'
    """
    
    def __init__(self, store_factory: StoreFactory = dict):
        """Initialize with a store factory.
        
        Args:
            store_factory: Callable that returns a MutableMapping
        """
        self._store = store_factory()
    
    def __getitem__(self, key: str) -> Project:
        return self._store[key]
    
    def __setitem__(self, key: str, value: Project) -> None:
        # Validate that it's a Project instance
        if not isinstance(value, Project):
            raise TypeError(f"Expected Project instance, got {type(value)}")
        # Ensure the project ID matches the key
        if value.id != key:
            raise ValueError(f"Project ID '{value.id}' doesn't match key '{key}'")
        self._store[key] = value
    
    def __delitem__(self, key: str) -> None:
        del self._store[key]
    
    def __iter__(self) -> Iterator[str]:
        return iter(self._store)
    
    def __len__(self) -> int:
        return len(self._store)
    
    def __contains__(self, key: str) -> bool:
        return key in self._store
    
    def create_project(self, 
                      project_id: str,
                      *,
                      segments_store_factory: StoreFactory = dict,
                      vectors_store_factory: StoreFactory = dict,
                      planar_store_factory: StoreFactory = dict,
                      cluster_store_factory: StoreFactory = dict,
                      embedders: Optional[ComponentRegistry] = None,
                      planarizers: Optional[ComponentRegistry] = None,
                      clusterers: Optional[ComponentRegistry] = None) -> Project:
        """Create and add a new project.
        
        Args:
            project_id: ID for the new project
            *_store_factory: Factory functions for various stores
            embedders: Component registry for embedders
            planarizers: Component registry for planarizers
            clusterers: Component registry for clusterers
            
        Returns:
            The created Project instance
        """
        project = Project(
            id=project_id,
            segments=segments_store_factory(),
            vectors=ComputeStore(vectors_store_factory),
            planar_coords=planar_store_factory(),
            cluster_indices=cluster_store_factory(),
            embedders=embedders or {},
            planarizers=planarizers or {},
            clusterers=clusterers or {}
        )
        self[project_id] = project
        return project