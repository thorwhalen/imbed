"""Integration tests for imbed_project module."""

"""Integration tests for imbed_project module."""

import pytest
import time
import tempfile
import shutil
from pathlib import Path

from imbed.imbed_project import Project, Projects
from au import ComputationStatus as AuComputationStatus


# Test fixtures and helpers
def simple_embedder(segments):
    """Simple embedder for testing - handles mapping input"""
    if isinstance(segments, dict):
        return {k: [len(v), v.count(' '), v.count('.')] for k, v in segments.items()}
    else:
        return [[len(s), s.count(' '), s.count('.')] for s in segments]


def slow_embedder(segments):
    """Embedder that simulates slow computation"""
    import time

    time.sleep(0.5)  # Simulate work
    return simple_embedder(segments)


def simple_planarizer(vectors):
    """Simple planarizer that takes first 2 dimensions"""
    return [(float(v[0]), float(v[1]) if len(v) > 1 else 0.0) for v in vectors]


def simple_clusterer(vectors):
    """Simple clusterer that assigns alternating clusters"""
    return [i % 2 for i in range(len(list(vectors)))]


@pytest.fixture
def temp_dir():
    """Create a temporary directory for async computations"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def basic_project(temp_dir):
    """Create a basic project with test components"""
    return Project(
        _id="test_proj",
        segments={},
        vectors={},  # Simple dict now!
        planar_coords={},
        cluster_indices={},
        embedders={
            'default': simple_embedder,
            'simple': simple_embedder,
            'slow': slow_embedder,
        },
        planarizers={'default': simple_planarizer, 'simple': simple_planarizer},
        clusterers={'default': simple_clusterer, 'simple': simple_clusterer},
        _async_embeddings=False,  # Start with sync mode for most tests
        _async_base_path=temp_dir,
    )


@pytest.fixture
def async_project(temp_dir):
    """Create a project with async embeddings enabled"""
    return Project(
        _id="async_proj",
        segments={},
        vectors={},
        planar_coords={},
        cluster_indices={},
        embedders={'default': simple_embedder, 'slow': slow_embedder},
        planarizers={'default': simple_planarizer},
        clusterers={'default': simple_clusterer},
        _async_embeddings=True,  # Enable async
        _async_base_path=temp_dir,
    )


class TestProjectBasicWorkflow:
    """Test the basic workflow of adding segments and computing embeddings"""

    def test_add_segments_sync_mode(self, basic_project):
        """Test adding segments with synchronous embedding"""
        # Add segments
        segments = {
            "doc1_s1": "The cat sat on the mat",
            "doc1_s2": "Dogs love to play fetch",
            "doc2_s1": "Birds fly south in winter",
        }
        segment_keys = basic_project.add_segments(segments)

        # Check segments were added
        assert len(segment_keys) == 3
        assert all(key in basic_project.segments for key in segment_keys)

        # Check embeddings were computed immediately (sync mode)
        assert all(key in basic_project.vectors for key in segment_keys)

        # Verify embedding values
        for key in segment_keys:
            vector = basic_project.vectors[key]
            assert isinstance(vector, list)
            assert len(vector) == 3  # Our simple embedder returns 3 values

    def test_add_segments_async_mode(self, async_project):
        """Test adding segments with asynchronous embedding"""
        # Add segments
        segments = {"s1": "Hello world", "s2": "Testing async"}
        segment_keys = async_project.add_segments(segments)

        # Check segments were added
        assert all(key in async_project.segments for key in segment_keys)

        # Embeddings should NOT be immediately available
        assert not all(key in async_project.vectors for key in segment_keys)

        # Check that computation is tracked
        active = async_project.list_active_computations()
        assert len(active) > 0

        # Wait for embeddings
        success = async_project.wait_for_embeddings(timeout=5.0)
        assert success

        # Now embeddings should be available
        assert all(key in async_project.vectors for key in segment_keys)

        # Verify values
        assert async_project.vectors["s1"] == [11, 1, 0]  # "Hello world"
        assert async_project.vectors["s2"] == [13, 1, 0]  # "Testing async"

    def test_embedding_status_tracking(self, async_project):
        """Test tracking of embedding statuses in async mode"""
        # Add some segments
        async_project.add_segments({"s1": "First segment", "s2": "Second segment"})

        # Check status immediately
        status = async_project.embedding_status
        assert status["missing"] >= 0  # Some might be missing
        assert status["computing"] >= 0  # Some might be computing

        # Wait for completion
        async_project.wait_for_embeddings(timeout=5.0)

        # Check final status
        status = async_project.embedding_status
        assert status["present"] == 2
        assert status["missing"] == 0
        assert status["computing"] == 0

    def test_toggle_async_mode(self, basic_project):
        """Test switching between sync and async modes"""
        # Start in sync mode
        assert not basic_project._async_embeddings

        # Add segments synchronously
        basic_project.add_segments({"sync": "Sync segment"})
        assert "sync" in basic_project.vectors

        # Switch to async mode
        basic_project.set_async_mode(True)

        # Add more segments asynchronously
        basic_project.add_segments({"async": "Async segment"})

        # async segment should not be immediately available
        assert "async" not in basic_project.vectors

        # Wait for it
        success = basic_project.wait_for_embeddings(["async"], timeout=5.0)
        assert success
        assert "async" in basic_project.vectors


class TestAsyncComputation:
    """Test async computation features"""

    def test_slow_embedder_async(self, async_project):
        """Test async computation with slow embedder"""
        # Use slow embedder
        async_project.default_embedder = "slow"

        # Add segments
        start_time = time.time()
        async_project.add_segments({"s1": "Segment one", "s2": "Segment two"})
        add_time = time.time() - start_time

        # Should return quickly (not wait for slow embedder)
        assert add_time < 0.3  # Much less than the 0.5s sleep

        # Embeddings not ready yet
        assert len(async_project.vectors) == 0

        # Wait for completion
        success = async_project.wait_for_embeddings(timeout=5.0)
        assert success

        # Check embeddings are correct
        assert len(async_project.vectors) == 2

    def test_computation_status_tracking(self, async_project):
        """Test tracking computation status"""
        # Add segments
        async_project.add_segments({"test": "Test segment"})

        # Get active computations
        active = async_project.list_active_computations()
        assert len(active) == 1

        # Check status
        comp_id = active[0]
        status = async_project.get_computation_status(comp_id)
        assert status in [AuComputationStatus.PENDING, AuComputationStatus.RUNNING]

        # Wait for completion
        async_project.wait_for_embeddings(timeout=5.0)

        # Should be cleaned up from active list
        active = async_project.list_active_computations()
        assert len(active) == 0

    def test_multiple_async_batches(self, async_project):
        """Test multiple async computations"""
        # Add first batch
        async_project.add_segments({"a1": "First A", "a2": "Second A"})

        # Add second batch immediately
        async_project.add_segments({"b1": "First B", "b2": "Second B"})

        # Should have multiple active computations
        active = async_project.list_active_computations()
        assert len(active) >= 1  # Might be 1 or 2 depending on timing

        # Wait for all
        success = async_project.wait_for_embeddings(timeout=5.0)
        assert success

        # All should be present
        assert len(async_project.vectors) == 4
        assert all(k in async_project.vectors for k in ["a1", "a2", "b1", "b2"])

    def test_async_computation_error_handling(self, async_project):
        """Test handling of errors in async computation"""

        # Create an embedder that fails
        def failing_embedder(segments):
            raise ValueError("Intentional failure")

        async_project.embedders["failing"] = failing_embedder
        async_project.default_embedder = "failing"

        # Add segments
        async_project.add_segments({"test": "Will fail"})

        # Wait a bit
        time.sleep(1.0)

        # Should not have the embedding
        assert "test" not in async_project.vectors

        # The computation should no longer be active
        # (it failed and was cleaned up)
        active = async_project.list_active_computations()
        # Active list gets cleaned on access
        assert len(active) == 0


class TestProjectComputation:
    """Test the generic computation interface"""

    def test_compute_with_async_override(self, basic_project):
        """Test compute with explicit async mode override"""
        # Project is in sync mode
        assert not basic_project._async_embeddings

        # Add segments first
        segments = {"s1": "Test segment"}
        basic_project.add_segments(segments)

        # Force async computation of embeddings
        save_key = basic_project.compute(
            "embedder", "simple", data={"s2": "Another segment"}, async_mode=True
        )

        # Should return immediately with a save key
        assert save_key.startswith("simple_")

        # s2 should not be immediately available
        assert "s2" not in basic_project.vectors

        # But s1 should be (from sync add_segments)
        assert "s1" in basic_project.vectors

        # Wait for async computation
        time.sleep(1.0)  # Give it time

        # Now s2 should be available
        assert "s2" in basic_project.vectors

    def test_compute_planarization_sync(self, basic_project):
        """Test computing planarization (always sync currently)"""
        # Add segments and compute embeddings
        segments = {
            "s1": "Hello world",
            "s2": "Python programming",
            "s3": "Machine learning",
        }
        basic_project.add_segments(segments)

        # Compute planarization
        save_key = basic_project.compute("planarizer", "simple", save_key="test_2d")

        # Check results (should be immediate)
        assert save_key == "test_2d"
        assert save_key in basic_project.planar_coords
        coords = basic_project.planar_coords[save_key]

        # Verify structure
        assert len(coords) == 3
        for key in segments:
            assert key in coords
            assert len(coords[key]) == 2


class TestProjectInvalidation:
    """Test the invalidation cascade when segments change"""

    def test_invalidation_removes_vectors(self, basic_project):
        """Test that adding segments removes old vectors"""
        # Initial segments
        segments1 = {"s1": "First", "s2": "Second"}
        basic_project.add_segments(segments1)

        # Verify embeddings exist
        assert "s1" in basic_project.vectors
        assert "s2" in basic_project.vectors

        # Compute derived data
        basic_project.compute("planarizer", "simple", save_key="coords_v1")

        # Modify s1
        basic_project.add_segments({"s1": "Modified first"})

        # s1 should have new embedding
        assert basic_project.vectors["s1"] == [14, 1, 0]  # "Modified first"

        # s2 should still have old embedding
        assert basic_project.vectors["s2"] == [6, 0, 0]  # "Second"

        # But planar coords should be cleared
        assert len(basic_project.planar_coords) == 0

    def test_invalidation_in_async_mode(self, async_project):
        """Test invalidation works with async embeddings"""
        # Add initial segments
        async_project.add_segments({"s1": "First"})
        async_project.wait_for_embeddings(timeout=5.0)

        # Compute derived data
        async_project.compute("clusterer", "default", save_key="clusters_v1")
        assert "clusters_v1" in async_project.cluster_indices

        # Add new segments
        async_project.add_segments({"s2": "Second"})

        # Clusters should be cleared
        assert len(async_project.cluster_indices) == 0

        # Wait for new embeddings
        async_project.wait_for_embeddings(timeout=5.0)

        # Both embeddings should be present
        assert len(async_project.vectors) == 2


class TestProjects:
    """Test the Projects container"""

    def test_projects_with_async_config(self, temp_dir):
        """Test creating projects with async configuration"""
        projects = Projects()

        # Create project with async enabled
        p = projects.create_project(
            project_id="async_test",
            embedders={'default': simple_embedder},
            async_embeddings=True,
            async_base_path=temp_dir,
        )

        assert p._id == "async_test"
        assert p._async_embeddings is True
        assert p._async_base_path == temp_dir

        # Add segments and verify async behavior
        p.add_segments({"test": "Test segment"})

        # Should not be immediately available
        assert "test" not in p.vectors

        # Wait for it
        success = p.wait_for_embeddings(timeout=5.0)
        assert success
        assert "test" in p.vectors


class TestCleanup:
    """Test cleanup functionality"""

    def test_cleanup_async_storage(self, async_project):
        """Test cleaning up expired async results"""
        # This is a basic test - full cleanup testing would require
        # manipulating TTL and time, which is complex

        # Add some segments
        async_project.add_segments({"test": "Test"})
        async_project.wait_for_embeddings(timeout=5.0)

        # Try cleanup (nothing should be expired yet)
        cleaned = async_project.cleanup_async_storage()
        assert cleaned == 0  # Nothing expired yet

        # Note: Testing actual expiration would require either:
        # 1. Mocking time functions
        # 2. Using very short TTL and sleeping
        # 3. Manually manipulating au's storage
        # For now, we just verify the method exists and returns 0


class TestAdvancedFeatures:
    """Test advanced project features"""

    def test_embedder_receives_mapping(self, basic_project):
        """Test that embedder receives segments as a mapping"""
        # Track what the embedder receives
        received_input = None

        def tracking_embedder(segments):
            nonlocal received_input
            received_input = segments
            return simple_embedder(segments)

        basic_project.embedders['default'] = tracking_embedder

        # Add segments
        segments = {"s1": "Hello", "s2": "World"}
        basic_project.add_segments(segments)

        # Verify embedder received the mapping
        assert isinstance(received_input, dict)
        assert received_input == segments

    def test_valid_vectors_property(self, basic_project):
        """Test the valid_vectors property"""
        # Add segments
        basic_project.add_segments({"s1": "Text 1", "s2": "Text 2"})

        # Get valid vectors
        valid = basic_project.valid_vectors
        assert len(valid) == 2
        assert "s1" in valid
        assert "s2" in valid

        # Modify the returned dict shouldn't affect internal state
        valid["s3"] = [1, 2, 3]
        assert "s3" not in basic_project.vectors

    def test_get_vectors_helper(self, basic_project):
        """Test the get_vectors helper method"""
        # Add segments
        segments = {f"s{i}": f"Text {i}" for i in range(3)}
        basic_project.add_segments(segments)

        # Get all vectors
        all_vectors = basic_project.get_vectors()
        assert len(all_vectors) == 3

        # Get specific vectors
        subset = basic_project.get_vectors(["s0", "s2"])
        assert len(subset) == 2

        # Missing keys are skipped
        subset = basic_project.get_vectors(["s0", "s99"])
        assert len(subset) == 1

    def test_compute_with_default_data(self, basic_project):
        """Test compute without providing data explicitly"""
        # Add segments
        basic_project.add_segments({"s1": "Hello", "s2": "World"})

        # Compute without providing data - should use vectors
        save_key = basic_project.compute("planarizer", "simple")

        coords = basic_project.planar_coords[save_key]
        assert len(coords) == 2

    def test_multiple_planarizations(self, basic_project):
        """Test saving multiple planarizations"""
        # Setup
        segments = {f"s{i}": f"Segment {i}" for i in range(4)}
        basic_project.add_segments(segments)

        # Compute multiple planarizations
        key1 = basic_project.compute("planarizer", "simple", save_key="proj_v1")
        key2 = basic_project.compute("planarizer", "simple", save_key="proj_v2")

        # Both should exist
        assert "proj_v1" in basic_project.planar_coords
        assert "proj_v2" in basic_project.planar_coords

        # Should have same structure (using same algorithm)
        assert len(basic_project.planar_coords["proj_v1"]) == 4
        assert len(basic_project.planar_coords["proj_v2"]) == 4

    def test_mixed_sync_async_workflow(self, basic_project):
        """Test mixing sync and async operations"""
        # Start with sync
        basic_project.add_segments({"s1": "Sync one"})
        assert "s1" in basic_project.vectors

        # Switch to async
        basic_project.set_async_mode(True)
        basic_project.add_segments({"s2": "Async two"})

        # s1 still there, s2 pending
        assert "s1" in basic_project.vectors
        assert "s2" not in basic_project.vectors

        # Can still compute on available vectors
        save_key = basic_project.compute("planarizer", "simple")
        coords = basic_project.planar_coords[save_key]
        assert len(coords) == 1  # Only s1

        # Wait for s2
        basic_project.wait_for_embeddings(["s2"], timeout=5.0)

        # Compute again with all vectors
        save_key2 = basic_project.compute("planarizer", "simple")
        coords2 = basic_project.planar_coords[save_key2]
        assert len(coords2) == 2  # Both s1 and s2


if __name__ == "__main__":
    pytest.main([__file__])
