"""Project interface for text embedding system.

This module provides the core Project class that manages segments, embeddings,
planarizations, and clusterings with automatic invalidation and async computation
support via the au framework.
"""

from typing import Optional, Any, Iterator, Callable, Union, TypeAlias
from dataclasses import dataclass, field
from enum import Enum
from collections.abc import MutableMapping, Mapping, Sequence
import time
import threading
from datetime import datetime

# Import from au for async computation
from au import (
    async_compute,
    ComputationHandle,
    ComputationStatus as AuComputationStatus,
)

# Import existing imbed types
from imbed.imbed_types import (
    Segment,
    SegmentKey,
    SegmentMapping,
    Vector,
    Vectors,
    VectorMapping,
    PlanarVectorMapping,
)

# Type aliases
ComponentRegistry: TypeAlias = MutableMapping[str, Callable]
ClusterIndex: TypeAlias = int
ClusterIndices: TypeAlias = Sequence[ClusterIndex]
ClusterMapping: TypeAlias = Mapping[SegmentKey, ClusterIndex]
StoreFactory: TypeAlias = Callable[[], MutableMapping]


def _generate_id() -> str:
    """Generate a unique ID"""
    import uuid

    return str(uuid.uuid4())[:8]


def _generate_timestamp() -> str:
    """Generate a timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class Project:
    """Central project interface - facade for all operations.

    Manages segments, embeddings, planarizations, and clusterings with
    automatic computation and invalidation cascade. Supports both synchronous
    and asynchronous embedding computation via the au framework.
    """

    id: str

    # Storage interfaces - all simple MutableMappings now
    segments: MutableMapping[SegmentKey, Segment]
    vectors: MutableMapping[SegmentKey, Vector]  # Simple mapping, no ComputeStore!
    planar_coords: MutableMapping[str, PlanarVectorMapping]
    cluster_indices: MutableMapping[str, ClusterMapping]

    # Component registries
    embedders: ComponentRegistry
    planarizers: ComponentRegistry
    clusterers: ComponentRegistry

    # Track active async computations
    _active_computations: MutableMapping[str, ComputationHandle] = field(
        default_factory=dict
    )

    # Configuration
    default_embedder: str = "default"
    _invalidation_cascade: bool = True
    _auto_compute_embeddings: bool = True
    _async_embeddings: bool = True  # Enable async computation
    _async_base_path: Optional[str] = None  # Base path for au storage

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
            self._invalidate_downstream(list(segments.keys()))

        # Trigger embedding computation if enabled
        if self._auto_compute_embeddings:
            if self._async_embeddings:
                # Launch async computation
                handle = self._compute_embeddings_async(segments)
                # Track the computation
                comp_id = f"embeddings_{_generate_timestamp()}"
                self._active_computations[comp_id] = handle
            else:
                # Compute synchronously (original behavior)
                self._compute_embeddings_sync(segments)

        return list(segments.keys())

    def _compute_embeddings_sync(self, segments: SegmentMapping) -> None:
        """Compute embeddings synchronously."""
        embedder = self.embedders[self.default_embedder]

        try:
            # Call embedder with the mapping - it handles batching
            embeddings = embedder(segments)

            # Store results
            if isinstance(embeddings, Mapping):
                self.vectors.update(embeddings)
            else:
                for key, vector in zip(segments.keys(), embeddings):
                    self.vectors[key] = vector

        except Exception as e:
            # In sync mode, we just raise the exception
            raise

    def _compute_embeddings_async(self, segments: SegmentMapping) -> ComputationHandle:
        """Compute embeddings asynchronously using au."""
        # Get the embedder
        embedder = self.embedders[self.default_embedder]

        # Create async version with project-specific storage
        base_path = self._async_base_path or f"/tmp/imbed_computations/{self.id}"
        async_embedder = async_compute(
            base_path=base_path,
            ttl_seconds=3600,  # 1 hour TTL
        )(embedder)

        # Launch computation
        handle = async_embedder(segments)

        # Set up a callback to store results when ready
        self._schedule_result_storage(handle, list(segments.keys()))

        return handle

    def _schedule_result_storage(
        self, handle: ComputationHandle, segment_keys: list[SegmentKey]
    ):
        """Poll for results and store them when ready."""

        def _store_when_ready():
            try:
                # Wait for results (with a reasonable timeout)
                embeddings = handle.get_result(timeout=300)  # 5 min timeout

                # Store in vectors
                if isinstance(embeddings, Mapping):
                    self.vectors.update(embeddings)
                else:
                    for key, vector in zip(segment_keys, embeddings):
                        self.vectors[key] = vector

            except Exception as e:
                print(f"Failed to compute embeddings: {e}")
                # Could also store error state if needed

        # Run in background thread
        thread = threading.Thread(target=_store_when_ready, daemon=True)
        thread.start()

    def compute(
        self,
        component_kind: str,
        component_key: str,
        data: Optional[Sequence] = None,
        *,
        save_key: Optional[str] = None,
        async_mode: Optional[bool] = None,
    ) -> str:
        """Generic computation dispatcher.

        Args:
            component_kind: Type of component ('embedder', 'planarizer', 'clusterer')
            component_key: Key of the component in the registry
            data: Input data (if None, uses appropriate default)
            save_key: Optional key to save results under
            async_mode: Override async behavior (None uses component defaults)

        Returns:
            Save key for retrieving results
        """
        # Get the component
        registry = getattr(self, f"{component_kind}s")
        if component_key not in registry:
            raise ValueError(f"Unknown {component_kind}: {component_key}")
        component = registry[component_key]

        # Generate save key if not provided
        if save_key is None:
            save_key = f"{component_key}_{_generate_timestamp()}"

        # Determine if we should use async
        use_async = (
            async_mode
            if async_mode is not None
            else (self._async_embeddings if component_kind == "embedder" else False)
        )

        # Get default data if not provided
        if data is None:
            if component_kind == "embedder":
                data = self.segments
            else:  # planarizer or clusterer
                data = list(self.vectors.values())

        if use_async and component_kind == "embedder":
            # Launch async computation
            handle = self._compute_embeddings_async(data)
            self._active_computations[save_key] = handle
            return save_key

        # Synchronous computation
        results = component(data)

        # Store results based on component kind
        segment_keys = list(self.segments.keys())

        if component_kind == "embedder":
            # Update vectors store
            if isinstance(results, Mapping):
                self.vectors.update(results)
            else:
                for key, vector in zip(segment_keys, results):
                    self.vectors[key] = vector
            return "vectors"

        elif component_kind == "planarizer":
            # Store as mapping from segment keys to 2D points
            if isinstance(results, Mapping):
                self.planar_coords[save_key] = results
            else:
                result_mapping = dict(zip(segment_keys[: len(list(results))], results))
                self.planar_coords[save_key] = result_mapping

        elif component_kind == "clusterer":
            # Store as mapping from segment keys to cluster indices
            if isinstance(results, Mapping):
                self.cluster_indices[save_key] = results
            else:
                result_mapping = dict(zip(segment_keys[: len(list(results))], results))
                self.cluster_indices[save_key] = result_mapping

        return save_key

    def _invalidate_downstream(self, segment_keys: list[SegmentKey]) -> None:
        """Mark computations as invalid when segments change"""
        # Remove embeddings for changed segments
        for key in segment_keys:
            self.vectors.pop(key, None)

        # Clear all planarizations and clusterings (they depend on all data)
        self.planar_coords.clear()
        self.cluster_indices.clear()

    def wait_for_embeddings(
        self,
        segment_keys: Optional[list[SegmentKey]] = None,
        timeout: float = 10.0,
        poll_interval: float = 0.1,
    ) -> bool:
        """Wait for embeddings to be available.

        This works for both sync and async modes - in sync mode, embeddings
        are immediately available; in async mode, we poll until they appear.
        """
        if segment_keys is None:
            segment_keys = list(self.segments.keys())

        start_time = time.time()
        while (time.time() - start_time) < timeout:
            if all(key in self.vectors for key in segment_keys):
                return True
            time.sleep(poll_interval)
        return False

    def get_computation_status(
        self, computation_id: str
    ) -> Optional[AuComputationStatus]:
        """Get status of a tracked async computation."""
        if computation_id in self._active_computations:
            handle = self._active_computations[computation_id]
            return handle.get_status()
        return None

    def list_active_computations(self) -> list[str]:
        """List IDs of active async computations."""
        # Clean up completed computations first
        completed = []
        for comp_id, handle in self._active_computations.items():
            if handle.is_ready():
                completed.append(comp_id)

        for comp_id in completed:
            del self._active_computations[comp_id]

        return list(self._active_computations.keys())

    @property
    def embedding_status(self) -> dict[str, int]:
        """Get counts of embedding statuses.

        Returns counts of: present, missing, computing
        """
        present = sum(1 for key in self.segments if key in self.vectors)
        total = len(self.segments)
        computing = len(
            [
                h
                for h in self._active_computations.values()
                if h.get_status() == AuComputationStatus.RUNNING
            ]
        )

        return {"present": present, "missing": total - present, "computing": computing}

    @property
    def valid_vectors(self) -> VectorMapping:
        """Get all available computed vectors"""
        return dict(self.vectors)  # Return a copy

    def get_vectors(
        self, segment_keys: Optional[list[SegmentKey]] = None
    ) -> list[Vector]:
        """Get vectors for specified segments (or all if None)"""
        if segment_keys is None:
            segment_keys = list(self.segments.keys())
        return [self.vectors[key] for key in segment_keys if key in self.vectors]

    def set_async_mode(self, enabled: bool) -> None:
        """Enable or disable async embedding computation."""
        self._async_embeddings = enabled

    def cleanup_async_storage(self) -> int:
        """Clean up expired async computation results."""
        cleaned = 0
        # Clean up au storage for each tracked embedder
        for embedder in self.embedders.values():
            if hasattr(embedder, 'cleanup_expired'):
                cleaned += embedder.cleanup_expired()
        return cleaned


class Projects(MutableMapping[str, Project]):
    """Container for projects with MutableMapping interface.

    >>> projects = Projects()
    >>> p = Project(id="test", segments={}, vectors={},
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

    def create_project(
        self,
        project_id: str,
        *,
        segments_store_factory: StoreFactory = dict,
        vectors_store_factory: StoreFactory = dict,
        planar_store_factory: StoreFactory = dict,
        cluster_store_factory: StoreFactory = dict,
        embedders: Optional[ComponentRegistry] = None,
        planarizers: Optional[ComponentRegistry] = None,
        clusterers: Optional[ComponentRegistry] = None,
        async_embeddings: bool = True,
        async_base_path: Optional[str] = None,
    ) -> Project:
        """Create and add a new project.

        Args:
            project_id: ID for the new project
            *_store_factory: Factory functions for various stores
            embedders: Component registry for embedders
            planarizers: Component registry for planarizers
            clusterers: Component registry for clusterers
            async_embeddings: Whether to use async embedding computation
            async_base_path: Base path for au async computation storage

        Returns:
            The created Project instance
        """
        project = Project(
            id=project_id,
            segments=segments_store_factory(),
            vectors=vectors_store_factory(),  # Simple store now!
            planar_coords=planar_store_factory(),
            cluster_indices=cluster_store_factory(),
            embedders=embedders or {},
            planarizers=planarizers or {},
            clusterers=clusterers or {},
            _async_embeddings=async_embeddings,
            _async_base_path=async_base_path,
        )
        self[project_id] = project
        return project
