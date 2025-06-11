"""Integration tests for imbed_project module."""

import pytest
import time
from imbed.imbed_project import Project, ComputeStore, ComputeStatus


# Test fixtures and helpers
def simple_embedder(text):
    """Simple embedder for testing"""
    return [len(text), text.count(' '), text.count('.')]


def simple_planarizer(vectors):
    """Simple planarizer that takes first 2 dimensions"""
    return [(float(v[0]), float(v[1]) if len(v) > 1 else 0.0) for v in vectors]


def simple_clusterer(vectors):
    """Simple clusterer that assigns alternating clusters"""
    return [i % 2 for i in range(len(list(vectors)))]


@pytest.fixture
def basic_project():
    """Create a basic project with test components"""
    return Project(
        id="test_proj",
        user_id="test_user",
        segments={},
        vectors=ComputeStore(),
        planar_coords={},
        cluster_indices={},
        embedders={
            'default': simple_embedder,
            'simple': simple_embedder
        },
        planarizers={
            'default': simple_planarizer,
            'simple': simple_planarizer
        },
        clusterers={
            'default': simple_clusterer,
            'simple': simple_clusterer
        }
    )


class TestProjectBasicWorkflow:
    """Test the basic workflow of adding segments and computing embeddings"""
    
    def test_add_segments_and_auto_embed(self, basic_project):
        """Test adding segments triggers automatic embedding"""
        # Add segments
        segments = {
            "doc1_s1": "The cat sat on the mat",
            "doc1_s2": "Dogs love to play fetch", 
            "doc2_s1": "Birds fly south in winter"
        }
        segment_keys = basic_project.add_segments(segments)
        
        # Check segments were added
        assert len(segment_keys) == 3
        assert all(key in basic_project.segments for key in segment_keys)
        
        # Check embeddings were computed automatically
        assert all(key in basic_project.vectors for key in segment_keys)
        
        # Verify embedding values
        for key in segment_keys:
            status, vector = basic_project.vectors[key]
            assert status == ComputeStatus.COMPLETED
            assert isinstance(vector, list)
            assert len(vector) == 3  # Our simple embedder returns 3 values
    
    def test_wait_for_embeddings(self, basic_project):
        """Test waiting for embeddings to complete"""
        segments = {"test": "Hello world"}
        basic_project.add_segments(segments)
        
        # Should complete immediately with our simple embedder
        completed = basic_project.wait_for_embeddings(timeout=1.0)
        assert completed
        assert basic_project.vectors.is_valid("test")
    
    def test_embedding_status_tracking(self, basic_project):
        """Test tracking of embedding statuses"""
        # Add some segments
        basic_project.add_segments({
            "s1": "First segment",
            "s2": "Second segment"
        })
        
        # Check status
        status = basic_project.embedding_status
        assert status.get(ComputeStatus.COMPLETED, 0) == 2
        
        # Invalidate one
        basic_project.vectors.invalidate("s1")
        status = basic_project.embedding_status
        assert status.get(ComputeStatus.COMPLETED, 0) == 1
        assert status.get(ComputeStatus.INVALIDATED, 0) == 1


class TestProjectComputation:
    """Test the generic computation interface"""
    
    def test_compute_planarization(self, basic_project):
        """Test computing and saving planar coordinates"""
        # Add segments and wait for embeddings
        segments = {
            "s1": "Hello world",
            "s2": "Python programming",
            "s3": "Machine learning"
        }
        basic_project.add_segments(segments)
        basic_project.wait_for_embeddings()
        
        # Compute planarization
        save_key = basic_project.compute("planarizer", "simple", save_key="test_2d")
        
        # Check results
        assert save_key == "test_2d"
        assert save_key in basic_project.planar_coords
        coords = basic_project.planar_coords[save_key]
        
        # Verify structure
        assert len(coords) == 3
        for key in segments:
            assert key in coords
            assert len(coords[key]) == 2
            assert all(isinstance(x, float) for x in coords[key])
    
    def test_compute_clustering(self, basic_project):
        """Test computing and saving cluster assignments"""
        # Add segments
        segments = {f"s{i}": f"Segment {i}" for i in range(5)}
        basic_project.add_segments(segments)
        basic_project.wait_for_embeddings()
        
        # Compute clustering with custom save key
        save_key = basic_project.compute("clusterer", "simple", 
                                        save_key="my_clusters")
        
        # Check results
        assert save_key == "my_clusters"
        assert save_key in basic_project.cluster_indices
        clusters = basic_project.cluster_indices[save_key]
        
        # Verify structure
        assert len(clusters) == 5
        for key in segments:
            assert key in clusters
            assert isinstance(clusters[key], int)
            assert clusters[key] in [0, 1]  # Our simple clusterer uses 2 clusters
    
    def test_compute_without_save_key(self, basic_project):
        """Test computation with auto-generated save keys"""
        basic_project.add_segments({"test": "Test segment"})
        basic_project.wait_for_embeddings()
        
        # Compute without specifying save_key
        save_key = basic_project.compute("planarizer", "simple")
        
        # Should generate a key with timestamp
        assert save_key.startswith("simple_")
        assert save_key in basic_project.planar_coords


class TestProjectInvalidation:
    """Test the invalidation cascade when segments change"""
    
    def test_invalidation_cascade(self, basic_project):
        """Test that adding segments invalidates downstream computations"""
        # Initial segments
        segments1 = {"s1": "First", "s2": "Second"}
        basic_project.add_segments(segments1)
        basic_project.wait_for_embeddings()
        
        # Compute derived data
        basic_project.compute("planarizer", "simple", save_key="coords_v1")
        basic_project.compute("clusterer", "simple", save_key="clusters_v1")
        
        # Verify they exist
        assert "coords_v1" in basic_project.planar_coords
        assert "clusters_v1" in basic_project.cluster_indices
        
        # Add new segments
        segments2 = {"s3": "Third", "s4": "Fourth"}
        basic_project.add_segments(segments2)
        
        # Check that planar coords and clusters were cleared
        assert len(basic_project.planar_coords) == 0
        assert len(basic_project.cluster_indices) == 0
        
        # But original embeddings should still be valid
        assert basic_project.vectors.is_valid("s1")
        assert basic_project.vectors.is_valid("s2")
    
    def test_disable_invalidation_cascade(self, basic_project):
        """Test disabling the invalidation cascade"""
        basic_project._invalidation_cascade = False
        
        # Add initial segments and compute
        basic_project.add_segments({"s1": "First"})
        basic_project.wait_for_embeddings()
        basic_project.compute("planarizer", "simple", save_key="coords")
        
        # Add more segments
        basic_project.add_segments({"s2": "Second"})
        
        # Planar coords should NOT be cleared
        assert "coords" in basic_project.planar_coords


class TestComputeStore:
    """Test the ComputeStore functionality"""
    
    def test_compute_store_operations(self):
        """Test ComputeStore status tracking and invalidation"""
        store = ComputeStore()
        
        # Add some data
        store["key1"] = (ComputeStatus.COMPLETED, [1, 2, 3])
        store["key2"] = (ComputeStatus.PENDING, None)
        store["key3"] = (ComputeStatus.COMPLETED, [4, 5, 6])
        
        # Test is_valid
        assert store.is_valid("key1")
        assert not store.is_valid("key2")
        assert store.is_valid("key3")
        
        # Test get_value
        assert store.get_value("key1") == [1, 2, 3]
        assert store.get_value("key2") is None
        
        # Test invalidate
        store.invalidate("key1")
        assert not store.is_valid("key1")
        assert store["key1"][0] == ComputeStatus.INVALIDATED
        assert store["key1"][1] == [1, 2, 3]  # Value preserved
        
        # Test invalidate_all
        store.invalidate_all()
        assert not store.is_valid("key3")
        assert store["key3"][0] == ComputeStatus.INVALIDATED


class TestProjectAdvancedFeatures:
    """Test advanced project features"""
    
    def test_get_vectors_helper(self, basic_project):
        """Test the get_vectors helper method"""
        # Add segments
        segments = {f"s{i}": f"Text {i}" for i in range(3)}
        basic_project.add_segments(segments)
        basic_project.wait_for_embeddings()
        
        # Get all vectors
        all_vectors = basic_project.get_vectors()
        assert len(all_vectors) == 3
        
        # Get specific vectors
        subset = basic_project.get_vectors(["s0", "s2"])
        assert len(subset) == 2
    
    def test_valid_vectors_property(self, basic_project):
        """Test the valid_vectors property"""
        # Add segments
        basic_project.add_segments({"s1": "Text 1", "s2": "Text 2"})
        basic_project.wait_for_embeddings()
        
        # Invalidate one
        basic_project.vectors.invalidate("s1")
        
        # Check valid_vectors only returns completed ones
        valid = basic_project.valid_vectors
        assert len(valid) == 1
        assert "s2" in valid
        assert "s1" not in valid
    
    def test_compute_with_default_data(self, basic_project):
        """Test compute without providing data explicitly"""
        # Add segments


if __name__ == "__main__":
    pytest.main([__file__])