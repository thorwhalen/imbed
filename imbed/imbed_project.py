"""Project interface for text embedding system.

This module provides the core Project class that manages segments, embeddings,
planarizations, and clusterings with automatic invalidation and status tracking.
"""

from typing import Optional, Any, Iterator, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections.abc import MutableMapping, Mapping, Sequence
import time
from datetime import datetime

# Import existing imbed types
from imbed.imbed_types import (
    Segment, SegmentKey, SegmentMapping,
    Vector, Vectors, VectorMapping,
    PlanarVectorMapping,
)

# Type aliases
ComponentRegistry = MutableMapping[str, Callable]
ClusterIndex = int
ClusterIndices = Sequence[ClusterIndex]
ClusterMapping = Mapping[SegmentKey, ClusterIndex]


class ComputeStatus(Enum):
    """Status of async computations"""
    PENDING = "pending"
    COMPUTING = "computing"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALIDATED = "invalidated"


class ComputeStore(dict):
    """Store that tracks computation status and invalidation.
    
    >>> store = ComputeStore()
    >>> store['key1'] = (ComputeStatus.COMPLETED, [1, 2, 3])
    >>> store.is_valid('key1')
    True
    >>> store.invalidate('key1')
    >>> store.is_valid('key1')
    False
    """
    
    def invalidate(self, key: str) -> None:
        """Mark a computation as invalid"""
        if key in self:
            status, value = self[key]
            self[key] = (ComputeStatus.INVALIDATED, value)
    
    def invalidate_all(self) -> None:
        """Mark all computations as invalid"""
        for key in list(self.keys()):
            self.invalidate(key)
    
    def is_valid(self, key: str) -> bool:
        """Check if a computation is valid"""
        if key not in self:
            return False
        status, _ = self[key]
        return status == ComputeStatus.COMPLETED
    
    def get_value(self, key: str) -> Optional[Any]:
        """Get the value if valid, None otherwise"""
        if self.is_valid(key):
            return self[key][1]
        return None


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
    automatic computation and invalidation cascade.
    """
    id: str
    user_id: str
    
    # Storage interfaces - injected dependencies
    segments: MutableMapping[SegmentKey, Segment]
    vectors: ComputeStore  # Maps SegmentKey -> (status, Vector)
    planar_coords: MutableMapping[str, PlanarVectorMapping]
    cluster_indices: MutableMapping[str, ClusterMapping]
    
    # Component registries
    embedders: ComponentRegistry
    planarizers: ComponentRegistry
    clusterers: ComponentRegistry
    
    # Configuration
    default_embedder: str = "default"
    _invalidation_cascade: bool = True
    _auto_compute_embeddings: bool = True
    
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
        
        # Schedule embedding computation
        if self._auto_compute_embeddings:
            for key in segments:
                if key not in self.vectors or not self.vectors.is_valid(key):
                    self.vectors[key] = (ComputeStatus.PENDING, None)
                    # Trigger computation (simplified - real version would be async)
                    self._compute_embedding(key)
        
        return list(segments.keys())
    
    def _compute_embedding(self, segment_key: SegmentKey) -> None:
        """Compute embedding for a single segment"""
        try:
            self.vectors[segment_key] = (ComputeStatus.COMPUTING, None)
            embedder = self.embedders[self.default_embedder]
            segment = self.segments[segment_key]
            
            # Handle both singular and batch embedders
            if hasattr(embedder, '__self__'):  # Bound method, likely batch
                vector = list(embedder([segment]))[0]
            else:
                vector = embedder(segment)
                
            self.vectors[segment_key] = (ComputeStatus.COMPLETED, vector)
        except Exception as e:
            self.vectors[segment_key] = (ComputeStatus.FAILED, str(e))
    
    def compute(self, 
                component_type: str,
                component_name: str,
                data: Optional[Sequence] = None,
                *,
                save_key: Optional[str] = None) -> str:
        """Generic computation dispatcher.
        
        Args:
            component_type: Type of component ('embedder', 'planarizer', 'clusterer')
            component_name: Name of the component in the registry
            data: Input data (if None, uses appropriate default)
            save_key: Optional key to save results under
            
        Returns:
            Save key for retrieving results
        """
        # Get the component
        registry = getattr(self, f"{component_type}s")
        if component_name not in registry:
            raise ValueError(f"Unknown {component_type}: {component_name}")
        component = registry[component_name]
        
        # Generate save key if not provided
        if save_key is None:
            save_key = f"{component_name}_{_generate_timestamp()}"
        
        # Get default data if not provided
        if data is None:
            if component_type == "embedder":
                data = list(self.segments.values())
            else:  # planarizer or clusterer
                data = [v for s, v in self.vectors.values() if s == ComputeStatus.COMPLETED]
        
        # Compute results
        if hasattr(component, '__call__'):
            results = list(component(data))
        else:
            raise ValueError(f"Component {component_name} is not callable")
        
        # Store results based on component type
        segment_keys = list(self.segments.keys())
        
        if component_type == "embedder":
            # Update compute store
            for key, vector in zip(segment_keys, results):
                self.vectors[key] = (ComputeStatus.COMPLETED, vector)
            return "vectors"
            
        elif component_type == "planarizer":
            # Store as mapping from segment keys to 2D points
            result_mapping = dict(zip(segment_keys[:len(results)], results))
            self.planar_coords[save_key] = result_mapping
            
        elif component_type == "clusterer":
            # Store as mapping from segment keys to cluster indices
            result_mapping = dict(zip(segment_keys[:len(results)], results))
            self.cluster_indices[save_key] = result_mapping
            
        return save_key
    
    def _invalidate_downstream(self, segment_keys: list[SegmentKey]) -> None:
        """Mark computations as invalid when segments change"""
        # Invalidate embeddings for changed segments
        for key in segment_keys:
            if key in self.vectors:
                self.vectors.invalidate(key)
        
        # Clear all planarizations and clusterings (they depend on all data)
        self.planar_coords.clear()
        self.cluster_indices.clear()
    
    def wait_for_embeddings(self, segment_keys: Optional[list[SegmentKey]] = None,
                           timeout: float = 10.0, poll_interval: float = 0.1) -> bool:
        """Wait for embeddings to complete.
        
        Args:
            segment_keys: Keys to wait for (if None, waits for all)
            timeout: Maximum time to wait in seconds
            poll_interval: Time between status checks
            
        Returns:
            True if all embeddings completed, False if timeout
        """
        if segment_keys is None:
            segment_keys = list(self.segments.keys())
            
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            if all(self.vectors.is_valid(key) for key in segment_keys):
                return True
            time.sleep(poll_interval)
        return False
    
    @property
    def embedding_status(self) -> dict[str, int]:
        """Get counts of embedding statuses"""
        from collections import Counter
        statuses = [status for status, _ in self.vectors.values()]
        return dict(Counter(statuses))
    
    @property
    def valid_vectors(self) -> VectorMapping:
        """Get all valid computed vectors"""
        return {k: v for k, (s, v) in self.vectors.items() 
                if s == ComputeStatus.COMPLETED}
    
    def get_vectors(self, segment_keys: Optional[list[SegmentKey]] = None) -> list[Vector]:
        """Get vectors for specified segments (or all if None)"""
        if segment_keys is None:
            segment_keys = list(self.segments.keys())
        return [self.vectors[key][1] for key in segment_keys 
                if self.vectors.is_valid(key)]