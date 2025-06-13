"""Integration tests for imbed_project module."""

import pytest
from imbed.imbed_project import (
    Project,
    Projects,
)


# Test fixtures and helpers
def simple_embedder(segments):
    """Simple embedder for testing - handles mapping input"""
    if isinstance(segments, dict):
        return {k: [len(v), v.count(' '), v.count('.')] for k, v in segments.items()}
    else:
        return [[len(s), s.count(' '), s.count('.')] for s in segments]


def simple_planarizer(vectors):
    """Simple planarizer that takes first 2 dimensions"""
    return {k: (float(v[0]), float(v[1]) if len(v) > 1 else 0.0) for k, v in vectors.items()}


def simple_clusterer(vectors):
    """Simple clusterer that assigns alternating clusters"""
    return {k: i % 2 for i, k in enumerate(vectors)}


@pytest.fixture
def basic_project():
    """Create a basic project with test components"""
    return Project(
        id="test_proj",
        segments={},
        vectors={},
        planar_coords={},
        cluster_indices={},
        embedders={'default': simple_embedder, 'simple': simple_embedder},
        planarizers={'default': simple_planarizer, 'simple': simple_planarizer},
        clusterers={'default': simple_clusterer, 'simple': simple_clusterer},
    )


class TestProjectBasicWorkflow:
    """Test the basic workflow of adding segments and computing embeddings"""

    def test_add_segments_and_auto_embed(self, basic_project):
        segments = {
            "doc1_s1": "The cat sat on the mat",
            "doc1_s2": "Dogs love to play fetch",
            "doc2_s1": "Birds fly south in winter",
        }
        segment_keys = basic_project.add_segments(segments)
        assert len(segment_keys) == 3
        assert all(key in basic_project.segments for key in segment_keys)
        assert basic_project.wait_for_embeddings(timeout=5)
        # Check embeddings were computed automatically
        assert all(key in basic_project.vectors for key in segment_keys)
        # Verify embedding values
        for key in segment_keys:
            vector = basic_project.vectors[key]
            assert isinstance(vector, list)
            assert len(vector) == 3  # Our simple embedder returns 3 values

    def test_embedder_receives_mapping(self, basic_project):
        segments = {"s1": "Hello", "s2": "World"}
        basic_project.add_segments(segments)
        basic_project.wait_for_embeddings(timeout=5)
        # The output vectors should correspond to the input segments
        for k, v in segments.items():
            assert k in basic_project.vectors
            assert basic_project.vectors[k][0] == len(v)

    def test_wait_for_embeddings(self, basic_project):
        segments = {"test": "Hello world"}
        basic_project.add_segments(segments)
        completed = basic_project.wait_for_embeddings(timeout=5)
        assert completed
        assert "test" in basic_project.vectors

    def test_embedding_status_tracking(self, basic_project):
        # Not applicable: status tracking is now handled by AU, not exposed.
        # Instead, just check that embeddings are present after wait.
        basic_project.add_segments({"s1": "First segment", "s2": "Second segment"})
        assert basic_project.wait_for_embeddings(timeout=5)
        assert "s1" in basic_project.vectors
        assert "s2" in basic_project.vectors


class TestProjectComputation:
    """Test the generic computation interface"""

    def test_compute_planarization(self, basic_project):
        segments = {
            "s1": "Hello world",
            "s2": "Python programming",
            "s3": "Machine learning",
        }
        basic_project.add_segments(segments)
        basic_project.wait_for_embeddings(timeout=5)
        save_key = basic_project.compute("planarizer", "simple", save_key="test_2d")
        assert save_key == "test_2d"
        assert save_key in basic_project.planar_coords
        coords = basic_project.planar_coords[save_key]
        assert len(coords) == 3
        for key in segments:
            assert key in coords
            assert len(coords[key]) == 2
            assert all(isinstance(x, float) for x in coords[key])

    def test_compute_clustering(self, basic_project):
        segments = {f"s{i}": f"Segment {i}" for i in range(5)}
        basic_project.add_segments(segments)
        basic_project.wait_for_embeddings(timeout=5)
        save_key = basic_project.compute("clusterer", "simple", save_key="my_clusters")
        assert save_key == "my_clusters"
        assert save_key in basic_project.cluster_indices
        clusters = basic_project.cluster_indices[save_key]
        assert len(clusters) == 5
        for key in segments:
            assert key in clusters
            assert isinstance(clusters[key], int)
            assert clusters[key] in [0, 1]

    def test_compute_without_save_key(self, basic_project):
        basic_project.add_segments({"test": "Test segment"})
        basic_project.wait_for_embeddings(timeout=5)
        save_key = basic_project.compute("planarizer", "simple")
        assert save_key.startswith("simple_")
        assert save_key in basic_project.planar_coords


class TestProjectInvalidation:
    """Test the invalidation cascade when segments change"""

    def test_invalidation_cascade(self, basic_project):
        segments1 = {"s1": "First", "s2": "Second"}
        basic_project.add_segments(segments1)
        basic_project.wait_for_embeddings(timeout=5)
        basic_project.compute("planarizer", "simple", save_key="coords_v1")
        basic_project.compute("clusterer", "simple", save_key="clusters_v1")
        assert "coords_v1" in basic_project.planar_coords
        assert "clusters_v1" in basic_project.cluster_indices
        segments2 = {"s3": "Third", "s4": "Fourth"}
        basic_project.add_segments(segments2)
        # Planar coords and clusters should be cleared
        assert len(basic_project.planar_coords) == 0
        assert len(basic_project.cluster_indices) == 0
        # Embeddings for old segments should still be present
        assert "s1" in basic_project.vectors
        assert "s2" in basic_project.vectors

    def test_disable_invalidation_cascade(self, basic_project):
        basic_project._invalidation_cascade = False
        basic_project.add_segments({"s1": "First"})
        basic_project.wait_for_embeddings(timeout=5)
        basic_project.compute("planarizer", "simple", save_key="coords")
        basic_project.add_segments({"s2": "Second"})
        assert "coords" in basic_project.planar_coords


class TestProjects:
    """Test the Projects container"""

    def test_projects_mapping_interface(self):
        projects = Projects()
        p1 = Project(
            id="proj1",
            segments={},
            vectors={},
            planar_coords={},
            cluster_indices={},
            embedders={'default': simple_embedder},
            planarizers={'default': simple_planarizer},
            clusterers={'default': simple_clusterer},
        )
        projects["proj1"] = p1
        assert "proj1" in projects
        assert len(projects) == 1
        assert list(projects) == ["proj1"]
        assert projects["proj1"] is p1
        del projects["proj1"]
        assert "proj1" not in projects
        assert len(projects) == 0

    def test_projects_validation(self):
        projects = Projects()
        with pytest.raises(TypeError):
            projects["bad"] = "not a project"
        p = Project(
            id="correct_id",
            segments={},
            vectors={},
            planar_coords={},
            cluster_indices={},
            embedders={},
            planarizers={},
            clusterers={},
        )
        with pytest.raises(ValueError, match="doesn't match key"):
            projects["wrong_id"] = p

    def test_projects_create_helper(self):
        projects = Projects()
        p = projects.create_project("test_id")
        assert p.id == "test_id"
        assert "test_id" in projects
        assert projects["test_id"] is p
        assert isinstance(p.segments, dict)
        assert isinstance(p.vectors, dict)
        assert isinstance(p.planar_coords, dict)
        assert isinstance(p.cluster_indices, dict)

    def test_projects_with_custom_backend(self):
        custom_backend = {}
        projects = Projects(lambda: custom_backend)
        p = Project(
            id="test",
            segments={},
            vectors={},
            planar_coords={},
            cluster_indices={},
            embedders={},
            planarizers={},
            clusterers={},
        )
        projects["test"] = p
        assert "test" in custom_backend
        assert custom_backend["test"] is p


class TestProjectAdvancedFeatures:
    """Test advanced project features"""

    def test_get_vectors_helper(self, basic_project):
        segments = {f"s{i}": f"Text {i}" for i in range(3)}
        basic_project.add_segments(segments)
        basic_project.wait_for_embeddings(timeout=5)
        all_vectors = basic_project.get_vectors()
        assert len(all_vectors) == 3
        subset = basic_project.get_vectors(["s0", "s2"])
        assert len(subset) == 2

    def test_valid_vectors_property(self, basic_project):
        basic_project.add_segments({"s1": "Text 1", "s2": "Text 2"})
        basic_project.wait_for_embeddings(timeout=5)
        # valid_vectors should return all vectors
        valid = basic_project.valid_vectors
        assert len(valid) == 2
        assert "s2" in valid
        assert "s1" in valid

    def test_compute_with_default_data(self, basic_project):
        basic_project.add_segments({"s1": "Hello", "s2": "World"})
        basic_project.wait_for_embeddings(timeout=5)
        save_key = basic_project.compute("planarizer", "simple")
        coords = basic_project.planar_coords[save_key]
        assert len(coords) == 2

    def test_multiple_planarizations(self, basic_project):
        segments = {f"s{i}": f"Segment {i}" for i in range(4)}
        basic_project.add_segments(segments)
        basic_project.wait_for_embeddings(timeout=5)
        key1 = basic_project.compute("planarizer", "simple", save_key="proj_v1")
        key2 = basic_project.compute("planarizer", "simple", save_key="proj_v2")
        assert "proj_v1" in basic_project.planar_coords
        assert "proj_v2" in basic_project.planar_coords
        assert len(basic_project.planar_coords["proj_v1"]) == 4
        assert len(basic_project.planar_coords["proj_v2"]) == 4


if __name__ == "__main__":
    pytest.main([__file__])
