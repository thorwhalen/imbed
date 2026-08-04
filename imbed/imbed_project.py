"""Project interface for text embedding system.

This module provides the core Project class that manages segments, embeddings,
planarizations, and clusterings with automatic invalidation and async computation
support via the au framework.
"""

import uuid
import os
import tempfile
import warnings
from typing import Optional, Any, Union, TypeAlias, Literal, get_args
from collections.abc import Iterator, Callable
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
from imbed.components.components_util import (
    get_standard_components,
    component_store_names,
    get_component_store,
)

# Type aliases
ComponentRegistry: TypeAlias = MutableMapping[str, Callable]
ClusterIndex: TypeAlias = int
ClusterIndices: TypeAlias = Sequence[ClusterIndex]
ClusterMapping: TypeAlias = Mapping[SegmentKey, ClusterIndex]
StoreFactory: TypeAlias = Callable[[], MutableMapping]

DFLT_PROJECT = "default_project"


#: The component kinds :meth:`Project.compute` knows how to dispatch. Each name
#: ``kind`` needs a ``Project.{kind}s`` registry attribute to resolve components from,
#: and a way to store its results (see :data:`RESULT_STORE_BY_COMPONENT_KIND`).
COMPONENT_KINDS = ("embedder", "planarizer", "clusterer")

#: The subset of :data:`COMPONENT_KINDS` whose work can be handed to the async (``au``)
#: backend. Single source of truth on purpose: :meth:`Project.compute` both *resolves*
#: and *validates* ``async_mode`` against this one name, so the two cannot drift — and
#: drift between them is exactly how ``async_mode=True`` came to be silently downgraded
#: to a synchronous run for the kinds the async branch did not cover.
ASYNC_CAPABLE_COMPONENT_KINDS = frozenset({"embedder"})

#: Which :class:`Project` store each component kind's results are filed in, under the
#: caller's ``save_key``. ``"embedder"`` is deliberately absent: embeddings are merged
#: into :attr:`Project.embeddings` under their *segment* keys, so an embedder's
#: ``save_key`` names the computation rather than a stored value.
RESULT_STORE_BY_COMPONENT_KIND = {
    "planarizer": "planar_coords",
    "clusterer": "cluster_indices",
}

#: What :meth:`Project.compute` may do when the default planarizer/clusterer input it
#: assembles is missing some segments (their embeddings have not landed yet).
MissingEmbeddingsPolicy: TypeAlias = Literal["raise", "warn", "ignore"]

#: :data:`MissingEmbeddingsPolicy`'s members, for runtime validation and messages.
MISSING_EMBEDDINGS_POLICIES = get_args(MissingEmbeddingsPolicy)

#: Default reaction to a partial default input. ``"warn"`` rather than ``"raise"``
#: because "process what is ready" is a legitimate thing to want — pending embeddings
#: are the normal state of an async project — but it must never be what a caller gets
#: *without knowing*, since a result covering part of the project is indistinguishable
#: from one covering all of it. ``"raise"`` is one keyword away for all-or-nothing
#: callers, ``"ignore"`` for those who already know the input is partial.
DFLT_ON_MISSING_EMBEDDINGS: MissingEmbeddingsPolicy = "warn"

#: How many missing segment keys an error/warning names before eliding the rest.
MAX_MISSING_KEYS_SHOWN = 5


def validate_component_kind_tables():
    """Assert the component-kind tables agree with each other.

    They are read in different places — registry lookup, async gating, result storage —
    and a kind present in one but absent from another is precisely the failure mode
    this module keeps hitting: a request that resolves fine and is then dropped.
    """
    assert set(ASYNC_CAPABLE_COMPONENT_KINDS) <= set(COMPONENT_KINDS), (
        f"ASYNC_CAPABLE_COMPONENT_KINDS {set(ASYNC_CAPABLE_COMPONENT_KINDS)} is not a "
        f"subset of COMPONENT_KINDS {set(COMPONENT_KINDS)}"
    )
    assert set(RESULT_STORE_BY_COMPONENT_KIND) <= set(COMPONENT_KINDS), (
        f"RESULT_STORE_BY_COMPONENT_KIND keys "
        f"{set(RESULT_STORE_BY_COMPONENT_KIND)} is not a subset of COMPONENT_KINDS "
        f"{set(COMPONENT_KINDS)}"
    )
    # Every kind must have somewhere for its results to go: merged into `embeddings`
    # (the embedder) or filed under a save key. A kind in neither group would be
    # computed and then discarded, with a save key returned for nothing.
    disposed_of = set(RESULT_STORE_BY_COMPONENT_KIND) | {"embedder"}
    unhandled = set(COMPONENT_KINDS) - disposed_of
    assert not unhandled, f"Component kinds with nowhere to store results: {unhandled}"


validate_component_kind_tables()


data_store_makers = {
    "misc": mk_dill_local_store,
    "segments": mk_json_local_store,
    "embeddings": mk_table_local_store,
    "clusters": mk_table_local_store,
    "planar_embeddings": mk_table_local_store,
    "statuses": mk_json_local_store,
    "cluster_labels": mk_dill_local_store,
}
data_store_names = tuple(data_store_makers.keys())

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

    assert set(data_store_makers) == set(data_store_names), (
        f"store_makers keys {set(data_store_makers)} do not match data_store_names {set(data_store_names)}"
    )

    for store_name in data_store_names:
        store_maker = data_store_makers.get(store_name, default_store_maker)
        mall[store_name] = store_maker(
            DFLT_PROJECTS_DIR, space=project_id, store_kind=store_name
        )

    return mall


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


def named_partial(func, *args, __name__=None, **kwargs):
    if __name__ is None:
        __name__ = func.__name__
    partial_func = partial(func, *args, **kwargs)
    partial_func.__name__ = __name__
    return partial_func


# TODO: Is it possible to do this with dol.wrap_kvs?
# TODO: This is a general tool useful for function stores, but where to put it (a new "function stores" package?)
class PartializedFuncs(Mapping[str, Callable]):
    """
    A mapping that allows retrieval of functions with optional partial application.

    >>> store = {'add': lambda x, y: x + y, 'subtract': lambda x, y: x - y}
    >>> partialized_ops = PartializedFuncs(store)

    When using a non-dict key, it will return the function directly:

    >>> func1 = partialized_ops['add']
    >>> func1(2, 3)
    5

    If the key is a dictionary with one item, it will return a partial function:

    >>> func2 = partialized_ops[{'add': {'y': 3}}]
    >>> func2(2)
    5

    """

    def __init__(self, store: Mapping[str, Callable]):
        self.store = store

    def __getitem__(self, key: str | dict) -> Callable:
        if isinstance(key, dict):
            items_iter = iter(key.items())
            func_key, func_kwargs = next(items_iter)

            if func_key not in self.store:
                raise KeyError(f"Key '{func_key}' not found in store '{self.store}'")

            if next(items_iter, None) is not None:
                raise KeyError(
                    f"Dict key must contain exactly one item: The dict was: {key}"
                )
            # Get the base function and create a partial with the kwargs
            base_func = self.store[func_key]
            return named_partial(base_func, **func_kwargs)

        else:
            return self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        return f"PartializedFuncs({self.store})"


def get_mall(
    project_id: str = DFLT_PROJECT,
    *,
    get_project_mall: MallKinds | Callable = DFLT_GET_PROJECT_MALL,
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
    # wrap the component stores with PartializedFuncs to enable partial application
    # of functions when they are called with a dict key.
    # This allows us to retrieve different "versions" of the base components.
    standard_components = {
        name: PartializedFuncs(store) for name, store in standard_components.items()
    }
    print(standard_components)

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


def _ensure_mk_mall(mk_mall_spec: str | Callable[[], Mall]) -> Callable[[], Mall]:
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


def _unique_key(prefix: str) -> str:
    """Build a unique, roughly time-sortable key from ``prefix``.

    The timestamp keeps generated keys readable and ordered; the random suffix keeps
    them *unique*, which a second-resolution timestamp alone cannot. Without it, two
    computations launched in the same second are handed the same key — and the second
    result then overwrites the first under a key both callers are still holding.

    >>> a, b = _unique_key('umap'), _unique_key('umap')
    >>> a.startswith('umap_') and b.startswith('umap_')
    True
    >>> a == b
    False
    """
    return _generate_id(prefix=f"{prefix}_{_generate_timestamp()}_")


def _validate_component_kind(component_kind: str) -> None:
    """Raise unless ``component_kind`` is one :meth:`Project.compute` can dispatch.

    Unvalidated, the kind goes straight into ``getattr(self, kind + 's')``: a typo
    surfaces as an ``AttributeError`` naming an attribute the caller never wrote, and a
    kind that happens to match a *data* store (``segments``, ``embeddings``) resolves
    to something that is not a component at all.
    """
    if component_kind not in COMPONENT_KINDS:
        raise ValueError(
            f"Unknown component_kind: {component_kind!r}. "
            f"Expected one of: {', '.join(COMPONENT_KINDS)}."
        )


def _validate_async_support(component_kind: str, async_mode: bool | None) -> None:
    """Raise if async was explicitly requested for a kind that has no async path.

    Refusing is the whole point: falling through to the synchronous path hands the
    caller a save key and a result indistinguishable from an asynchronous run, which
    turns an unsupported request into a wrong answer about what actually happened.
    """
    if async_mode and component_kind not in ASYNC_CAPABLE_COMPONENT_KINDS:
        supported = ", ".join(sorted(ASYNC_CAPABLE_COMPONENT_KINDS))
        raise ValueError(
            f"async_mode=True is not supported for component_kind={component_kind!r}: "
            f"only {supported} computations can be run asynchronously. "
            f"Omit async_mode (or pass async_mode=False) to run the {component_kind} "
            f"synchronously."
        )


def _report_missing_embeddings(
    missing: Sequence[SegmentKey],
    *,
    total: int,
    policy: MissingEmbeddingsPolicy = DFLT_ON_MISSING_EMBEDDINGS,
) -> None:
    """Raise, warn about, or ignore segments that have no embedding yet, per ``policy``.

    The policy is validated even when nothing is missing, so a mistyped one is caught
    on the happy path instead of being read as "do nothing" on the day it matters.
    """
    if policy not in MISSING_EMBEDDINGS_POLICIES:
        raise ValueError(
            f"Unknown on_missing_embeddings: {policy!r}. "
            f"Expected one of: {', '.join(MISSING_EMBEDDINGS_POLICIES)}."
        )
    if not missing or policy == "ignore":
        return

    shown = ", ".join(map(str, missing[:MAX_MISSING_KEYS_SHOWN]))
    if len(missing) > MAX_MISSING_KEYS_SHOWN:
        shown += ", ..."
    message = (
        f"{len(missing)} of {total} segments have no embedding yet, so this "
        f"computation covers only the other {total - len(missing)} "
        f"(missing: {shown}). Call wait_for_embeddings() first for a complete input, "
        f"pass on_missing_embeddings='raise' to make this fatal, or "
        f"on_missing_embeddings='ignore' to accept a partial input silently."
    )
    if policy == "raise":
        raise ValueError(message)
    # stacklevel=4: _report_missing_embeddings <- _embeddings_input <- compute <- caller
    warnings.warn(message, stacklevel=4)


def _keyed_results(
    results: Any,
    *,
    keys: list[SegmentKey] | None,
    component_kind: str,
    component_key: str,
) -> dict:
    """Pair a component's results with the segment keys they belong to.

    A component may return a mapping (it knows its own keys) or a plain sequence (its
    results correspond, by position, to the input it was given). In the second case the
    keys have to come from the input — and only the caller of ``compute`` knows them
    when the caller supplied the input, hence ``keys=None`` being an error rather than
    an invitation to guess.

    ``results`` is materialised *before* it is measured: a component returning a
    generator would otherwise be consumed by the length check, leaving nothing to pair
    up and storing an empty result under a key the caller was handed.
    """
    if isinstance(results, Mapping):
        return dict(results)

    results = list(results)

    if keys is None:
        raise ValueError(
            f"Cannot attribute the {component_key!r} {component_kind}'s results to "
            f"segment keys: it returned a plain sequence, and `data` was given as a "
            f"plain sequence too, so nothing records which segment each result belongs "
            f"to. Omit `data` to run on the project's own embeddings (whose segment "
            f"order is known), pass `data` as a mapping of segment key -> value, or "
            f"use a {component_kind} that returns a mapping."
        )
    if len(results) != len(keys):
        raise ValueError(
            f"The {component_key!r} {component_kind} returned {len(results)} results "
            f"for {len(keys)} inputs. Results are matched to segments by position, so "
            f"a count mismatch would file them under the wrong segments: a "
            f"{component_kind} must return exactly one result per input."
        )
    return dict(zip(keys, results))


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
    _async_base_path: str | None = None  # Base path for au storage
    _id: str | None = None
    _async_backend: Any | None = None  # Backend for async computation

    @classmethod
    def from_mall(
        cls,
        mk_mall: str | Callable[[], Mall] = DFLT_GET_PROJECT_MALL,
        *,
        default_embedder: str = "default",
        _id: str | None = None,
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
                # Track the computation. The id must be unique, not just timestamped:
                # batches added within the same second would otherwise overwrite each
                # other here, leaving all but the last untrackable.
                self._active_computations[_unique_key("embeddings")] = handle
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

    def _compute_embeddings_async(
        self, segments: SegmentMapping, *, embedder: Callable | None = None
    ) -> ComputationHandle:
        """Compute embeddings asynchronously using au.

        Args:
            segments: The segments to embed.
            embedder: The embedder to run. Injected by :meth:`compute`, which has
                already resolved the component the caller asked for. Defaults to the
                project's default embedder, which is what :meth:`add_segments` wants
                (its contract *is* "use the default").
        """
        if embedder is None:
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

    def _embeddings_input(
        self, *, on_missing_embeddings: MissingEmbeddingsPolicy
    ) -> tuple[list[SegmentKey], list[Embedding]]:
        """The default planarizer/clusterer input: segment embeddings, with their keys.

        Segments whose embedding has not landed yet cannot contribute a vector, so this
        input is necessarily a *subset* of the project — and in async mode, *which*
        subset is a race. ``on_missing_embeddings`` decides what the caller is told
        about the shortfall; what they must not be told is nothing, since a result
        describing part of the project looks exactly like one describing all of it.

        The keys are returned alongside the vectors because they are the only record of
        which segment each positional result belongs to.
        """
        keys: list[SegmentKey] = []
        missing: list[SegmentKey] = []
        for key in self.segments:
            if key in self.embeddings:
                keys.append(key)
            else:
                missing.append(key)

        _report_missing_embeddings(
            missing, total=len(keys) + len(missing), policy=on_missing_embeddings
        )
        return keys, [self.embeddings[key] for key in keys]

    def compute(
        self,
        component_kind: str,
        component_key: str,
        data: Mapping | Sequence | None = None,
        *,
        save_key: str | None = None,
        async_mode: bool | None = None,
        on_missing_embeddings: MissingEmbeddingsPolicy = DFLT_ON_MISSING_EMBEDDINGS,
    ) -> str:
        """Generic computation dispatcher.

        Runs the component the caller named, on the data the caller gave, in the mode
        the caller asked for — or raises saying why it cannot. It never quietly
        substitutes a different component, a different mode, or a different input:
        every path here returns the same kind of save key, so a silent substitution
        would be indistinguishable from success.

        Args:
            component_kind: Type of component; one of :data:`COMPONENT_KINDS`.
            component_key: Key of the component in the corresponding registry.
            data: Input data. If None, an appropriate default is used (see below).
                For an embedder it must be a ``Mapping`` of segment key -> segment,
                since embeddings are stored per segment key and a bare sequence carries
                no keys to store them under.
            save_key: Key to file the results under. Generated (uniquely) if not given.
                An explicitly given key is honoured as given, and so overwrites any
                previous result stored under it.
            async_mode: Whether to run asynchronously. ``None`` (the default) uses the
                project's ``_async_embeddings`` setting, which applies only to kinds in
                :data:`ASYNC_CAPABLE_COMPONENT_KINDS`. Passing ``True`` for any other
                kind raises: those kinds have no async path, and running them
                synchronously would silently contradict an explicit request.
            on_missing_embeddings: What to do when the *default* input for a
                planarizer/clusterer is incomplete because some segments have no
                embedding yet — one of :data:`MISSING_EMBEDDINGS_POLICIES`, default
                :data:`DFLT_ON_MISSING_EMBEDDINGS`. Only consulted when ``data`` is
                None; when you supply ``data``, its completeness is yours to decide.

        Returns:
            The save key. For a planarizer or clusterer this is the key its results are
            stored under, in ``planar_coords`` / ``cluster_indices`` respectively. For
            an embedder it identifies the *computation*: embedding results are merged
            into ``embeddings`` under their own segment keys, sync or async alike.

        Default input by kind:
            - embedder: all of ``segments``.
            - planarizer, clusterer: the embeddings of every segment that has one, in
              segment order (see ``on_missing_embeddings`` for the ones that do not).

        How results are attributed to segments:
            A component returning a mapping is taken at its word. A component returning
            a plain sequence is matched to its input by position — which requires
            knowing the input's keys, so the sequence case is supported when ``data``
            was defaulted or given as a mapping, and refused (rather than guessed) when
            ``data`` was given as a bare sequence.

        Raises:
            ValueError: unknown ``component_kind``, unknown ``component_key``, async
                requested for a kind that cannot do it, an unknown
                ``on_missing_embeddings``, or results that cannot be attributed to
                segments.
            TypeError: embedder ``data`` that is not a Mapping.
        """
        _validate_component_kind(component_kind)
        _validate_async_support(component_kind, async_mode)

        # Get the component
        registry = getattr(self, f"{component_kind}s")
        if component_key not in registry:
            raise ValueError(f"Unknown {component_kind}: {component_key}")
        component = registry[component_key]

        # Generate save key if not provided
        if save_key is None:
            save_key = _unique_key(component_key)

        # Determine if we should use async. Resolution and the branch below are gated
        # on the same set, so an async request can only ever be honoured or refused.
        use_async = (
            async_mode
            if async_mode is not None
            else (
                self._async_embeddings
                and component_kind in ASYNC_CAPABLE_COMPONENT_KINDS
            )
        )

        # The segment keys the results will correspond to, where they are knowable.
        # Stays None when the caller supplies a keyless `data`: how their sequence lines
        # up with this project's segments is theirs to know, and guessing it is how
        # results end up filed under unrelated segments.
        result_keys: list[SegmentKey] | None = None

        if data is None:
            if component_kind == "embedder":
                data = self.segments
                result_keys = list(self.segments)
            else:
                result_keys, data = self._embeddings_input(
                    on_missing_embeddings=on_missing_embeddings
                )
        elif isinstance(data, Mapping):
            result_keys = list(data)

        if component_kind == "embedder" and not isinstance(data, Mapping):
            raise TypeError(
                f"An embedder's data must be a Mapping of segment key -> segment, got "
                f"{type(data).__name__}. Embeddings are stored per segment key, so a "
                f"bare {type(data).__name__} leaves compute no keys to file them "
                f"under. Pass e.g. {{'my_segment_key': 'my text'}}, or omit `data` to "
                f"embed the project's segments."
            )

        if use_async:
            # Launch async computation with the component the caller asked for. The
            # sync path below honours `component_key`; the async path must agree, or
            # `compute(..., async_mode=True)` silently runs the *default* embedder.
            handle = self._compute_embeddings_async(data, embedder=component)
            self._active_computations[save_key] = handle
            return save_key

        # Synchronous computation
        results = _keyed_results(
            component(data),
            keys=result_keys,
            component_kind=component_kind,
            component_key=component_key,
        )

        if component_kind == "embedder":
            # Embeddings are keyed by segment, not by save_key: merge them in.
            self.embeddings.update(results)
        else:
            store = getattr(self, RESULT_STORE_BY_COMPONENT_KIND[component_kind])
            store[save_key] = results

        return save_key

    def _invalidate_downstream(self, segment_keys: list[SegmentKey]) -> None:
        """Mark computations as invalid when segments change"""
        # Clear all planarizations and clusterings (they depend on all data)
        # We don't clear embeddings here because they're updated in add_segments
        clear_store(self.planar_coords)
        clear_store(self.cluster_indices)

    def wait_for_embeddings(
        self,
        segment_keys: list[SegmentKey] | None = None,
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

    def get_computation_status(self, computation_id: str) -> AuComputationStatus | None:
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
        self, segment_keys: list[SegmentKey] | None = None
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
        project_id: str | None = None,
        segments_store_factory: StoreFactory = dict,
        embeddings_store_factory: StoreFactory = dict,
        planar_store_factory: StoreFactory = dict,
        cluster_store_factory: StoreFactory = dict,
        embedders: ComponentRegistry | None = None,
        planarizers: ComponentRegistry | None = None,
        clusterers: ComponentRegistry | None = None,
        async_embeddings: bool = True,
        async_base_path: str | None = None,
        async_backend: Any | None = None,
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
