"""Project interface for text embedding system.

This module provides the core Project class that manages segments, embeddings,
planarizations, and clusterings with automatic invalidation and async computation
support via the au framework.
"""

import uuid
import os
import tempfile
from typing import Optional, Any, Iterator, Callable, Union, TypeAlias, Literal
from dataclasses import dataclass, field, KW_ONLY
from functools import partial, lru_cache
from collections.abc import MutableMapping, Mapping, Sequence
import time
import threading
from datetime import datetime
import os
import tempfile

# Import from au for async computation
from au import (
    async_compute,
    ComputationHandle,
    ComputationStatus as AuComputationStatus,
)
from au.base import StdLibQueueBackend, FileSystemStore, SerializationFormat

from imbed.util import DFLT_PROJECTS_DIR, ensure_segments_mapping

from imbed.imbed_types import (
    Segment,
    SegmentKey,
    SegmentMapping,
    Segments,
    SegmentsSpec,
    Embedding,
    Embeddings,
    EmbeddingMapping,
    PlanarVectorMapping,
)
from imbed.stores_util import (
    Store,
    Mall,
    mk_table_local_store,
    mk_json_local_store,
    mk_dill_local_store,
)

# Type aliases
ComponentRegistry: TypeAlias = MutableMapping[str, Callable]
ClusterIndex: TypeAlias = int
ClusterIndices: TypeAlias = Sequence[ClusterIndex]
ClusterMapping: TypeAlias = Mapping[SegmentKey, ClusterIndex]
StoreFactory: TypeAlias = Callable[[], MutableMapping]

DFLT_PROJECT = "default_project"

_component_kinds = ("segmenters", "embedders", "clusterers", "planarizers")

data_store_names = ("segments", "embeddings", "planar_embeddings", "clusters", "misc")
component_store_names = _component_kinds

mall_keys = tuple(data_store_names + component_store_names)


def validate_mall_keys(mall: Mapping):
    missing_keys = set(mall_keys) - set(mall.keys())
    if missing_keys:
        raise ValueError(f"Missing keys in mall: {missing_keys}")


def get_local_mall(
    project_id: str = DFLT_PROJECT,
    *,
    mall_keys=data_store_names,
    default_store_maker=mk_dill_local_store,
):
    """
    Get the user stores for the package.

    Returns:
        dict: A dictionary containing paths to various user stores.
    """
    mall = {}

    store_makers = {
        "misc": mk_dill_local_store,
        "segments": mk_json_local_store,
        "embeddings": mk_table_local_store,
        "clusters": mk_table_local_store,
        "planar_embeddings": mk_table_local_store,
    }

    assert set(store_makers) == set(
        data_store_names
    ), f"store_makers keys {set(store_makers)} do not match data_store_names {set(data_store_names)}"

    for store_name in data_store_names:
        store_maker = store_makers.get(store_name, default_store_maker)
        mall[store_name] = store_maker(
            DFLT_PROJECTS_DIR, space=project_id, store_kind=store_name
        )

    return mall


def get_component_store(component: str):
    """Get the store for a specific component type"""
    if component == "segmenters":
        from imbed.components.segmentation import segmenters as component_store
    elif component == "embedders":
        from imbed.components.vectorization import embedders as component_store
    elif component == "clusterers":
        from imbed.components.clusterization import clusterers as component_store
    elif component == "planarizers":
        from imbed.components.planarization import planarizers as component_store
    else:
        raise ValueError(f"Unknown component type: {component}")
    return component_store.copy()


@lru_cache
def get_standard_components():
    """Get the standard components for the project.

    Returns:
        A dictionary of standard components, each containing registered processing functions
    """
    return {kind: get_component_store(kind) for kind in _component_kinds}


def get_ram_project_mall(project_id: str = DFLT_PROJECT) -> Mall:
    return {k: dict() for k in mall_keys}
    # previously (to accept everything):
    # from collections import defaultdict
    # return defaultdict(dict)


# DFLT_GET_PROJECT_MALL = get_local_mall
DFLT_GET_PROJECT_MALL = get_ram_project_mall

mall_kinds = {
    "local": get_local_mall,
    "ram": get_ram_project_mall,
}

MallKinds = Literal["local", "ram"]


# assert that the MallKinds type is a valid subset of the mall_kinds keys
def validate_mall_kinds():
    assert set(MallKinds.__args__) <= set(mall_kinds.keys())


validate_mall_kinds()


def get_mall(
    project_id: str = DFLT_PROJECT,
    *,
    get_project_mall: Union[MallKinds, Callable] = DFLT_GET_PROJECT_MALL,
    include_signature_stores=True,
) -> Mall:
    """Get the registry mall containing all function stores

    Returns:
        A dictionary of stores, each containing registered processing functions
    """
    if isinstance(get_project_mall, str):
        get_project_mall_key = get_project_mall
        if get_project_mall_key not in mall_kinds:
            raise ValueError(
                f"Unknown get_project_mall: {get_project_mall_key}. "
                "Expected one of: " + ", ".join(mall_kinds.keys())
            )
        get_project_mall = mall_kinds[get_project_mall_key]
    standard_components = get_standard_components()
    # TODO: Add user-defined components
    project_mall = get_project_mall(project_id)

    _function_stores = standard_components  # TODO: Eventually, some user stores will also be function stroes

    if include_signature_stores:
        from ju import signature_to_json_schema
        from dol import wrap_kvs, AttributeMapping

        signature_values = wrap_kvs(value_decoder=signature_to_json_schema)

        signature_stores = {
            f"{k}_signatures": signature_values(v) for k, v in _function_stores.items()
        }
    else:
        signature_stores = {}

    mall_dict = dict(project_mall, **standard_components, **signature_stores)
    validate_mall_keys(mall_dict)

    return AttributeMapping(**mall_dict)


DFLT_MALL = get_mall(DFLT_PROJECT)

mk_mall_kinds = {
    "local": get_local_mall,
    "ram": get_ram_project_mall,
    "default": DFLT_GET_PROJECT_MALL,
}


def _ensure_mk_mall(mk_mall_spec: Union[str, Callable[[], Mall]]) -> Callable[[], Mall]:
    """Ensure the mk_mall_spec is a callable that returns a Mall"""
    if isinstance(mk_mall_spec, str):
        mk_mall_kind = mk_mall_spec.lower()
        if mk_mall_kind in mk_mall_kinds:
            # Return the corresponding mall getter function
            return mk_mall_kinds[mk_mall_kind]
        else:
            raise ValueError(
                f"Unknown mk_mall_spec: {mk_mall_spec}. "
                "Expected callable, or one of: "
                f"{', '.join(mk_mall_kinds.keys())}"
            )
    elif callable(mk_mall_spec):
        return mk_mall_spec
    else:
        raise TypeError("mk_mall_spec must be a string or a callable returning a Mall")


def _generate_id(*, prefix="", uuid_n_chars=8, suffix="") -> str:
    """Generate a unique ID"""
    return prefix + str(uuid.uuid4())[:uuid_n_chars] + suffix


def _generate_timestamp() -> str:
    """Generate a timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clear_store(store: MutableMapping) -> None:
    """Clear all items in a store"""
    if "clear" in dir(store):
        try:
            store.clear()
            return None
        except NotImplementedError:
            # Fallback for stores that don't support clear method
            for key in store.keys():
                del store[key]
    else:
        # Fallback for stores that don't support clear method
        for key in store.keys():
            del store[key]


@dataclass
class Project:
    """Central project interface - facade for all operations.

    Manages segments, embeddings, planarizations, and clusterings with
    automatic computation and invalidation cascade. Supports both synchronous
    and asynchronous embedding computation via the au framework.
    """

    KW_ONLY
    segments: MutableMapping[SegmentKey, Segment] = field(default_factory=dict)
    embeddings: MutableMapping[SegmentKey, Embedding] = field(default_factory=dict)
    planar_coords: MutableMapping[str, PlanarVectorMapping] = field(
        default_factory=dict
    )
    cluster_indices: MutableMapping[str, ClusterMapping] = field(default_factory=dict)

    # Component registries
    embedders: ComponentRegistry = field(
        default_factory=partial(get_component_store, "embedders")
    )
    planarizers: ComponentRegistry = field(
        default_factory=partial(get_component_store, "planarizers")
    )
    clusterers: ComponentRegistry = field(
        default_factory=partial(get_component_store, "clusterers")
    )

    default_embedder: str = "default"

    # Track active async computations
    _active_computations: MutableMapping[str, ComputationHandle] = field(
        default_factory=dict
    )

    # Configuration
    _invalidation_cascade: bool = True
    _auto_compute_embeddings: bool = True
    _async_embeddings: bool = False  # Default to sync mode for reliability
    _async_base_path: Optional[str] = None  # Base path for au storage
    _id: Optional[str] = None
    _async_backend: Optional[Any] = None  # Backend for async computation

    @classmethod
    def from_mall(
        cls,
        mk_mall: Union[str, Callable[[], Mall]] = DFLT_GET_PROJECT_MALL,
        *,
        default_embedder: str = "default",
        _id: Optional[str] = None,
        **extra_configs,
    ):

        if _id is None:
            _id = _generate_id(prefix="imbed_project_")
        mk_mall = _ensure_mk_mall(mk_mall)
        mall = mk_mall(_id)
        project = cls(
            segments=mall["segments"],
            embeddings=mall["embeddings"],
            planar_coords=mall["planar_embeddings"],
            cluster_indices=mall["clusters"],
            embedders=mall.get("embedders", get_component_store("embedders")),
            planarizers=mall.get("planarizers", get_component_store("planarizers")),
            clusterers=mall.get("clusterers", get_component_store("clusterers")),
            default_embedder=default_embedder,
            **extra_configs,
        )
        project.mall = mall
        return project

    def add_segments(self, segments: SegmentMapping) -> list[SegmentKey]:
        """Add segments and trigger embedding computation.

        Args:
            segments: Mapping of segment keys to segment text

        Returns:
            List of segment keys that were added
        """
        if not isinstance(segments, Mapping):
            raise TypeError("Segments must be a mapping of SegmentKey to Segment")
        # Update segments
        self.segments.update(segments)

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

        # Invalidate dependent computations
        if self._invalidation_cascade:
            self._invalidate_downstream(list(segments.keys()))

        return list(segments.keys())

    def _compute_embeddings_sync(self, segments: SegmentMapping) -> None:
        """Compute embeddings synchronously."""
        embedder = self.embedders[self.default_embedder]

        try:
            # Call embedder with the mapping - it handles batching
            embeddings = embedder(segments)

            # Store results
            if isinstance(embeddings, Mapping):
                self.embeddings.update(embeddings)
            else:
                for key, vector in zip(segments.keys(), embeddings):
                    self.embeddings[key] = vector

        except Exception as e:
            # In sync mode, we just raise the exception
            raise

    def _compute_embeddings_async(self, segments: SegmentMapping) -> ComputationHandle:
        """Compute embeddings asynchronously using au."""
        embedder = self.embedders[self.default_embedder]

        # Use project ID if available, otherwise use a temporary ID for storage path
        project_id = self._id or _generate_id(prefix="imbed_project_")

        base_path = self._async_base_path or os.path.join(
            tempfile.gettempdir(), "imbed_computations", project_id
        )

        # Use provided backend or default to StdLibQueueBackend
        backend = self._async_backend
        store = None
        if backend is None:
            store = FileSystemStore(
                base_path,
                ttl_seconds=3600,
                serialization=SerializationFormat.PICKLE,  # Use pickle for functions
            )
            backend = StdLibQueueBackend(
                store, use_processes=False
            )  # Use threads to avoid pickling issues
        else:
            # If user provided a backend, try to extract its store if possible
            store = getattr(backend, "store", None)

        async_embedder = async_compute(
            backend=backend,
            store=store,
            base_path=base_path,
            ttl_seconds=3600,  # 1 hour TTL
            serialization=SerializationFormat.PICKLE,  # Use pickle for better function serialization
        )(embedder)

        handle = async_embedder(segments)
        self._schedule_result_storage(handle, list(segments.keys()))
        return handle

    def _schedule_result_storage(
        self, handle: ComputationHandle, segment_keys: list[SegmentKey]
    ):
        """Poll for results and store them when ready."""

        def _store_when_ready():
            try:
                # Wait for results (with a reasonable timeout)
                embeddings = handle.get_result(timeout=30)  # 30 sec timeout

                # Store in embeddings
                if isinstance(embeddings, Mapping):
                    self.embeddings.update(embeddings)
                else:
                    for key, vector in zip(segment_keys, embeddings):
                        self.embeddings[key] = vector

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
                # For planarizers and clusterers, we need the embeddings as input
                # But we need to get embeddings for all segments that have them
                data = [
                    self.embeddings[key]
                    for key in self.segments.keys()
                    if key in self.embeddings
                ]

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
            # Update embeddings store
            if isinstance(results, Mapping):
                self.embeddings.update(results)
            else:
                # Assume results are in same order as segments
                segment_keys_for_data = (
                    list(data.keys()) if isinstance(data, Mapping) else segment_keys
                )
                for key, vector in zip(segment_keys_for_data, results):
                    self.embeddings[key] = vector
            return save_key  # Return the save_key, not "embeddings"

        elif component_kind == "planarizer":
            # Store as mapping from segment keys to 2D points
            if isinstance(results, Mapping):
                self.planar_coords[save_key] = results
            else:
                # Map results back to segment keys that have embeddings
                valid_segment_keys = [
                    key for key in self.segments.keys() if key in self.embeddings
                ]
                result_mapping = dict(
                    zip(valid_segment_keys[: len(list(results))], results)
                )
                self.planar_coords[save_key] = result_mapping

        elif component_kind == "clusterer":
            # Store as mapping from segment keys to cluster indices
            if isinstance(results, Mapping):
                self.cluster_indices[save_key] = results
            else:
                # Map results back to segment keys that have embeddings
                valid_segment_keys = [
                    key for key in self.segments.keys() if key in self.embeddings
                ]
                result_mapping = dict(
                    zip(valid_segment_keys[: len(list(results))], results)
                )
                self.cluster_indices[save_key] = result_mapping

        return save_key

    def _invalidate_downstream(self, segment_keys: list[SegmentKey]) -> None:
        """Mark computations as invalid when segments change"""
        # Clear all planarizations and clusterings (they depend on all data)
        # We don't clear embeddings here because they're updated in add_segments
        clear_store(self.planar_coords)
        clear_store(self.cluster_indices)

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
            if all(key in self.embeddings for key in segment_keys):
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
        present = sum(1 for key in self.segments if key in self.embeddings)
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
    def valid_embeddings(self) -> EmbeddingMapping:
        """Get all available computed embeddings"""
        return dict(self.embeddings)  # Return a copy

    def get_embeddings(
        self, segment_keys: Optional[list[SegmentKey]] = None
    ) -> list[Embedding]:
        """Get embeddings for specified segments (or all if None)"""
        if segment_keys is None:
            segment_keys = list(self.segments.keys())
        return [self.embeddings[key] for key in segment_keys if key in self.embeddings]

    def set_async_mode(self, enabled: bool) -> None:
        """Enable or disable async embedding computation."""
        self._async_embeddings = enabled

    def cleanup_async_storage(self) -> int:
        """Clean up expired async computation results."""
        cleaned = 0
        # Clean up au storage for each tracked embedder
        for embedder in self.embedders.values():
            if hasattr(embedder, "cleanup_expired"):
                cleaned += embedder.cleanup_expired()
        return cleaned


class Projects(MutableMapping[str, Project]):
    """Container for projects with MutableMapping interface.

    >>> projects = Projects()
    >>> p = Project(_id='test', segments={}, embeddings={},
    ...             planar_coords={}, cluster_indices={},
    ...             embedders={}, planarizers={}, clusterers={})
    >>> projects["test"] = p
    >>> list(projects)
    ['test']
    >>> projects["test"]._id
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
        # Handle project ID assignment
        if value._id is None:
            value._id = key
        elif value._id != key:
            raise ValueError(f"Project ID '{value._id}' doesn't match key '{key}'")
        self._store[key] = value

    def __delitem__(self, key: str) -> None:
        del self._store[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def append(self, project: Project) -> None:
        """Append a project to the collection.

        Args:
            project: Project instance to add
        """
        if not isinstance(project, Project):
            raise TypeError(f"Expected Project instance, got {type(project)}")
        self[project._id] = project

    def create_project(
        self,
        *,
        project_id: Optional[str] = None,
        segments_store_factory: StoreFactory = dict,
        embeddings_store_factory: StoreFactory = dict,
        planar_store_factory: StoreFactory = dict,
        cluster_store_factory: StoreFactory = dict,
        embedders: Optional[ComponentRegistry] = None,
        planarizers: Optional[ComponentRegistry] = None,
        clusterers: Optional[ComponentRegistry] = None,
        async_embeddings: bool = True,
        async_base_path: Optional[str] = None,
        async_backend: Optional[Any] = None,
        overwrite: bool = False,
    ) -> Project:
        """Create and add a new project.

        Args:
            project_id: ID for the new project (optional)
            *_store_factory: Factory functions for various stores
            embedders: Component registry for embedders
            planarizers: Component registry for planarizers
            clusterers: Component registry for clusterers
            async_embeddings: Whether to use async embedding computation
            async_base_path: Base path for au async computation storage
            async_backend: Backend for async computation (StdLibQueueBackend, RQ, etc)
            overwrite: If True, replace any existing project with the same id

        Returns:
            The created Project instance
        """
        if project_id is not None:
            if project_id in self and not overwrite:
                raise ValueError(f"Project ID '{project_id}' already exists.")
        project = Project(
            segments=segments_store_factory(),
            embeddings=embeddings_store_factory(),
            planar_coords=planar_store_factory(),
            cluster_indices=cluster_store_factory(),
            embedders=embedders or {},
            planarizers=planarizers or {},
            clusterers=clusterers or {},
            _async_embeddings=async_embeddings,
            _async_base_path=async_base_path,
            _async_backend=async_backend,
            _id=project_id,
        )
        self[project._id] = project
        return project
