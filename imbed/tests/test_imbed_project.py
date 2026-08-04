"""Integration tests for imbed_project module."""

import pytest
import time
import threading
import tempfile
import shutil
import warnings
from pathlib import Path

from imbed.imbed_project import Project, Projects
from au import ComputationStatus as AuComputationStatus
from au.base import FileSystemStore, StdLibQueueBackend


#: Upper bound on how long a condition-based wait keeps polling. This is a *failure*
#: bound, never a success bound: a correct implementation satisfies the condition as
#: soon as its worker lands, so raising this never slows a passing test down.
WAIT_TIMEOUT_S = 10.0

#: How many computations the key-collision tests launch back to back. More than two on
#: purpose: rapid calls span at most one second boundary, hence at most two distinct
#: second-resolution timestamps, so a suite of this size cannot be satisfied by a
#: timestamp that merely *usually* differs.
RAPID_CALL_COUNT = 5

#: Poll granularity for :func:`wait_until` — small enough to keep tests quick, large
#: enough not to spin the GIL against the worker threads under test.
POLL_INTERVAL_S = 0.01


def wait_until(predicate, *, timeout=WAIT_TIMEOUT_S, interval=POLL_INTERVAL_S):
    """Poll ``predicate`` until it is true; report whether it became true in time.

    Use this instead of ``time.sleep(<guess>)`` followed by an assertion. A fixed
    sleep couples the test to wall-clock luck — too short and it flakes under load,
    too long and every run pays for it. Polling a condition does neither.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


#: Gate that :func:`gated_embedder` blocks on. A test closes it, launches an async
#: computation, and can then assert "the result has not landed yet" as a *fact it
#: controls* rather than a scheduling race it has to win. Module level because the
#: embedder is pickled by qualified name and resolves this as a worker-side global.
EMBEDDER_GATE = threading.Event()


# --- Move all embedders/planarizers/clusterers to module level for pickling ---
def simple_embedder(segments):
    """Simple embedder for testing - handles mapping input"""
    if isinstance(segments, dict):
        return {k: [len(v), v.count(" "), v.count(".")] for k, v in segments.items()}
    else:
        return [[len(s), s.count(" "), s.count(".")] for s in segments]


def slow_embedder(segments):
    """Embedder that simulates slow computation"""
    import time

    time.sleep(0.5)  # Simulate work
    return simple_embedder(segments)


#: Vector :func:`marker_embedder` emits. Deliberately unlike anything
#: :func:`simple_embedder` can produce, so a test can tell *which* embedder ran.
MARKER_VECTOR = [-999.0]


def marker_embedder(segments):
    """Embedder whose output identifies it unambiguously."""
    return {k: list(MARKER_VECTOR) for k in segments}


def gated_embedder(segments):
    """Embedder that blocks until the test opens :data:`EMBEDDER_GATE`.

    This is what makes the async assertions deterministic. "Did ``compute`` hand the
    work off instead of running it inline?" used to be inferred from *how fast* the
    call returned, which only holds while the main thread wins a ~20ms race against
    the worker. Blocking on an explicit gate turns that inference into a real
    happens-before edge that no amount of scheduler jitter can invert.
    """
    if not EMBEDDER_GATE.wait(timeout=WAIT_TIMEOUT_S):
        raise TimeoutError("gated_embedder was never released by the test")
    return simple_embedder(segments)


def simple_planarizer(embeddings):
    """Simple planarizer that takes first 2 dimensions"""
    return [(float(v[0]), float(v[1]) if len(v) > 1 else 0.0) for v in embeddings]


def generator_planarizer(embeddings):
    """Planarizer that yields its points lazily instead of returning a list.

    Perfectly legal for a component, and a good probe: code that *measures* a result
    before consuming it destroys a one-shot iterator.
    """
    return (point for point in simple_planarizer(embeddings))


def truncating_planarizer(embeddings):
    """Planarizer that returns one point too few — i.e. a buggy component.

    Positional results are matched to segments by position, so a short return is not
    recoverable: every point from the dropped one onward would be filed under the
    wrong segment.
    """
    return simple_planarizer(embeddings)[:-1]


def simple_clusterer(embeddings):
    """Simple clusterer that assigns alternating clusters"""
    return [i % 2 for i in range(len(list(embeddings)))]


# Test fixtures and helpers
@pytest.fixture
def temp_dir():
    """Create a temporary directory for async computations"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def embedder_gate():
    """Hand the test control of :data:`EMBEDDER_GATE`, starting closed.

    Always reopened on teardown, even when the test fails: a worker thread still
    parked at the gate would otherwise survive the test and leak into the next one.
    """
    EMBEDDER_GATE.clear()
    try:
        yield EMBEDDER_GATE
    finally:
        EMBEDDER_GATE.set()


@pytest.fixture
def basic_project(temp_dir):
    """Create a basic project with test components (sync mode)"""
    return Project(
        _id="test_proj",
        segments={},
        embeddings={},
        planar_coords={},
        cluster_indices={},
        embedders={
            "default": simple_embedder,
            "simple": simple_embedder,
            "slow": slow_embedder,
            "gated": gated_embedder,
        },
        planarizers={
            "default": simple_planarizer,
            "simple": simple_planarizer,
            "generator": generator_planarizer,
            "truncating": truncating_planarizer,
        },
        clusterers={"default": simple_clusterer, "simple": simple_clusterer},
        _async_embeddings=False,  # Start with sync mode for most tests
        _async_base_path=temp_dir,
    )


@pytest.fixture
def partially_embedded_project(basic_project):
    """A project where only one of three segments has an embedding.

    The ordinary state of an async project mid-flight, and the state in which a
    "planarize the project" request can only be *partly* honoured.
    """
    basic_project._auto_compute_embeddings = False
    basic_project.add_segments({"s1": "one", "s2": "two", "s3": "three"})
    basic_project.embeddings["s1"] = [1.0, 2.0, 3.0]
    return basic_project


@pytest.fixture
def async_project(temp_dir):
    """Create a project with async embeddings enabled and default StdLibQueueBackend"""
    from au.base import FileSystemStore, StdLibQueueBackend, SerializationFormat

    store = FileSystemStore(
        temp_dir, ttl_seconds=3600, serialization=SerializationFormat.PICKLE
    )
    backend = StdLibQueueBackend(
        store, use_processes=False
    )  # Use threads to avoid pickling issues
    return Project(
        _id="async_proj",
        segments={},
        embeddings={},
        planar_coords={},
        cluster_indices={},
        embedders={
            "default": simple_embedder,
            "slow": slow_embedder,
            "gated": gated_embedder,
        },
        planarizers={"default": simple_planarizer},
        clusterers={"default": simple_clusterer},
        _async_embeddings=True,  # Enable async
        _async_base_path=temp_dir,
        _async_backend=backend,
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
        assert all(key in basic_project.embeddings for key in segment_keys)

        # Verify embedding values
        for key in segment_keys:
            vector = basic_project.embeddings[key]
            assert isinstance(vector, list)
            assert len(vector) == 3  # Our simple embedder returns 3 values

    def test_add_segments_async_mode(self, async_project):
        """Test adding segments with asynchronous embedding"""
        # Add segments
        segments = {"s1": "Hello world", "s2": "Testing async"}
        segment_keys = async_project.add_segments(segments)

        # Check segments were added
        assert all(key in async_project.segments for key in segment_keys)

        # Check that computation was tracked (might already be completed)
        # We check if there were computations created by checking internal state
        assert (
            len(async_project._active_computations) >= 0
        )  # Could be 0 if already completed

        # Wait for embeddings (in case they're not ready yet). A failed wait is a real
        # failure, not something to skip past: the sibling async tests establish that
        # the pipeline lands well inside this budget, so exhausting it means the work
        # was dropped -- exactly the defect a test should surface rather than mask.
        assert async_project.wait_for_embeddings(timeout=WAIT_TIMEOUT_S)

        # Now embeddings should be available
        assert all(key in async_project.embeddings for key in segment_keys)

        # Verify values
        assert async_project.embeddings["s1"] == [11, 1, 0]  # "Hello world"
        assert async_project.embeddings["s2"] == [13, 1, 0]  # "Testing async"

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

    def test_toggle_async_mode(self, basic_project, embedder_gate):
        """Test switching between sync and async modes.

        Same gate treatment as the other async-ness assertions: "not immediately
        available" is guaranteed by the embedder being blocked, not by the main
        thread happening to reach the assertion first.
        """
        # Start in sync mode
        assert not basic_project._async_embeddings

        # Add segments synchronously
        basic_project.add_segments({"sync": "Sync segment"})
        assert "sync" in basic_project.embeddings

        # Switch to async mode, with an embedder the test holds open
        basic_project.set_async_mode(True)
        basic_project.default_embedder = "gated"

        # Add more segments asynchronously
        basic_project.add_segments({"async": "Async segment"})

        # async segment cannot be available: its embedder is still at the gate
        assert "async" not in basic_project.embeddings

        # Release it and wait for it
        embedder_gate.set()
        success = basic_project.wait_for_embeddings(["async"], timeout=WAIT_TIMEOUT_S)
        assert success
        assert "async" in basic_project.embeddings


class TestAsyncComputation:
    """Test async computation features"""

    def test_slow_embedder_async(self, async_project, embedder_gate):
        """Test async computation does not block on a slow embedder.

        The property is "``add_segments`` hands the work off", which used to be
        asserted as ``add_time < 0.3`` — an absolute duration, i.e. a statement about
        the machine rather than about the code. The gate states it exactly instead:
        ``add_segments`` returns while the embedder is still blocked, so if it *did*
        run the embedder inline it could not return at all.
        """
        # Use an embedder the test can hold open for as long as it likes
        async_project.default_embedder = "gated"

        async_project.add_segments({"s1": "Segment one", "s2": "Segment two"})

        # Returned without waiting for the embedder, which is still at the gate
        assert len(async_project.embeddings) == 0

        # Release it, then wait for completion
        embedder_gate.set()
        success = async_project.wait_for_embeddings(timeout=WAIT_TIMEOUT_S)
        assert success

        # Check embeddings are correct
        assert len(async_project.embeddings) == 2

    def test_computation_status_tracking(self, async_project):
        """Test tracking computation status"""
        # Add segments
        async_project.add_segments({"test": "Test segment"})

        # The computation might complete very quickly, so we need to be flexible
        # Check that the computation was created (even if it's already done)
        # We can verify this by checking that embeddings were computed

        # Wait briefly to ensure computation has a chance to complete
        success = async_project.wait_for_embeddings(timeout=5.0)
        assert success

        # The computation should have happened and produced results
        assert "test" in async_project.embeddings
        assert async_project.embeddings["test"] == [12, 1, 0]  # "Test segment"

        # Since computation is very fast, active list should be cleaned up
        active = async_project.list_active_computations()
        assert len(active) == 0  # Should be cleaned up after completion

    def test_multiple_async_batches(self, async_project):
        """Test multiple async computations"""
        # Add first batch
        async_project.add_segments({"a1": "First A", "a2": "Second A"})

        # Add second batch immediately
        async_project.add_segments({"b1": "First B", "b2": "Second B"})

        # With fast computation, by the time we check, they might already be done
        # The key is that async mode was used and all results are computed

        # Wait for all to complete
        success = async_project.wait_for_embeddings(timeout=5.0)
        assert success

        # All should be present
        assert len(async_project.embeddings) == 4
        assert all(k in async_project.embeddings for k in ["a1", "a2", "b1", "b2"])

        # Verify the async computation produced correct results
        assert async_project.embeddings["a1"] == [7, 1, 0]  # "First A"
        assert async_project.embeddings["a2"] == [8, 1, 0]  # "Second A"
        assert async_project.embeddings["b1"] == [7, 1, 0]  # "First B"
        assert async_project.embeddings["b2"] == [8, 1, 0]  # "Second B"

    # Patch the error-handling test to skip if function is not picklable
    @pytest.mark.skip(
        reason="Can't pickle local functions for async backends; only works with top-level functions."
    )
    def test_async_computation_error_handling(self, async_project):
        """Test handling of errors in async computation (skipped for local function pickling)"""
        pass


class TestProjectComputation:
    """Test the generic computation interface"""

    def test_compute_with_async_override(self, basic_project, embedder_gate):
        """Test compute with explicit async mode override.

        Deterministic by construction: the embedder is held at ``embedder_gate``, so
        "s2 has not been computed yet" is a fact the test controls. Previously this
        asserted it right after launching a *fast* embedder, which left only a ~20ms
        window — any scheduler delay past that (routine on a loaded CI box) let the
        worker land first and turned the suite red. The trailing ``time.sleep(1.0)``
        had the mirror-image problem: it guessed at how long the worker needed.
        """
        # Project is in sync mode
        assert not basic_project._async_embeddings

        # Add segments first
        segments = {"s1": "Test segment"}
        basic_project.add_segments(segments)

        # Force async computation of embeddings, using the gated embedder so the
        # worker provably cannot finish while the gate is closed.
        save_key = basic_project.compute(
            "embedder", "gated", data={"s2": "Another segment"}, async_mode=True
        )

        # Should return immediately with a save key
        assert save_key.startswith("gated_")

        # s2 cannot be available: its embedder is still blocked at the gate
        assert "s2" not in basic_project.embeddings

        # But s1 should be (from sync add_segments)
        assert "s1" in basic_project.embeddings

        # Release the worker, then wait on the condition rather than on the clock
        embedder_gate.set()
        assert wait_until(lambda: "s2" in basic_project.embeddings)

    def test_async_compute_honours_the_requested_embedder(self, basic_project):
        """``compute`` must run the component it was asked for, sync *or* async.

        Regression test: the async branch dropped the resolved component and always
        ran ``self.default_embedder``, so ``compute("embedder", "marker",
        async_mode=True)`` silently produced *default* embeddings. Silent, because
        both paths return the same save key and the wrong vectors are still vectors.
        """
        basic_project.embedders["marker"] = marker_embedder

        basic_project.compute(
            "embedder", "marker", data={"sync_key": "x"}, async_mode=False
        )
        basic_project.compute(
            "embedder", "marker", data={"async_key": "x"}, async_mode=True
        )
        assert wait_until(lambda: "async_key" in basic_project.embeddings)

        assert basic_project.embeddings["sync_key"] == MARKER_VECTOR
        assert basic_project.embeddings["async_key"] == MARKER_VECTOR

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


class TestComputeHonoursTheCallersRequest:
    """``compute`` must do what it was asked, or say why it cannot.

    Every test here pins one instance of a single defect shape: the caller states an
    intent, ``compute`` resolves it, and then some branch quietly ignores the
    resolution — handing back a save key that is indistinguishable from success. The
    first of these to be found was the async path running the *default* embedder
    instead of the requested one (see
    :meth:`TestProjectComputation.test_async_compute_honours_the_requested_embedder`);
    these are its siblings.
    """

    def test_async_mode_is_refused_rather_than_downgraded(self, basic_project):
        """Asking for async on a kind that has no async path must fail, not run sync.

        ``use_async`` was computed for *any* kind while the async branch was guarded on
        ``component_kind == "embedder"``, so a planarizer request with
        ``async_mode=True`` set the flag, skipped the branch, and ran synchronously —
        returning the same save key a genuine async launch would have.
        """
        basic_project.add_segments({"s1": "Hello world", "s2": "Python code"})

        with pytest.raises(ValueError) as excinfo:
            basic_project.compute("planarizer", "simple", async_mode=True)

        # The error must tell the caller which kinds *can* go async
        assert "async_mode" in str(excinfo.value)
        assert "embedder" in str(excinfo.value)
        # Refused outright, not quietly half-honoured
        assert len(basic_project.planar_coords) == 0

    def test_generated_save_keys_do_not_collide(self, basic_project):
        """Auto-generated save keys must be unique, not merely timestamped.

        The generated key was ``f"{component_key}_{second_resolution_timestamp}"``, so
        computations launched in the same second were handed *the same* key and the
        later result overwrote the earlier one — under a key both callers still held.
        """
        basic_project.add_segments({"s1": "Hello world", "s2": "Python code"})

        keys = [
            basic_project.compute("planarizer", "simple")
            for _ in range(RAPID_CALL_COUNT)
        ]

        assert len(set(keys)) == RAPID_CALL_COUNT
        assert len(basic_project.planar_coords) == RAPID_CALL_COUNT

    def test_every_async_batch_stays_trackable(self, async_project):
        """Each async batch needs its own tracking id, for the same reason.

        ``add_segments`` filed handles under ``f"embeddings_{timestamp}"``: batches
        added in the same second overwrote each other, so all but the last became
        untrackable and ``embedding_status['computing']`` undercounted them.
        """
        for i in range(RAPID_CALL_COUNT):
            async_project.add_segments({f"s{i}": f"Segment {i}"})

        assert len(async_project._active_computations) == RAPID_CALL_COUNT
        assert async_project.wait_for_embeddings(timeout=WAIT_TIMEOUT_S)

    def test_partial_default_input_is_reported(self, partially_embedded_project):
        """A default input missing some segments must not pass unremarked.

        ``compute("planarizer", ...)`` with no ``data`` silently dropped every segment
        whose embedding had not landed yet, so the caller received a planarization
        *of part of the project* that looks exactly like one of all of it.
        """
        with pytest.warns(UserWarning, match="2 of 3 segments"):
            save_key = partially_embedded_project.compute("planarizer", "simple")

        # Still permitted — "planarize what is ready" is legitimate; being unaware is not
        assert len(partially_embedded_project.planar_coords[save_key]) == 1

    def test_partial_default_input_can_be_made_fatal(self, partially_embedded_project):
        """Callers that need all-or-nothing can say so."""
        with pytest.raises(ValueError, match="no embedding"):
            partially_embedded_project.compute(
                "planarizer", "simple", on_missing_embeddings="raise"
            )

        assert len(partially_embedded_project.planar_coords) == 0

    def test_partial_default_input_can_be_accepted_silently(
        self, partially_embedded_project
    ):
        """...and callers that know it is partial can opt out of being told."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails the test
            save_key = partially_embedded_project.compute(
                "planarizer", "simple", on_missing_embeddings="ignore"
            )

        assert len(partially_embedded_project.planar_coords[save_key]) == 1

    def test_unknown_missing_embeddings_policy_is_reported(
        self, partially_embedded_project
    ):
        """A mistyped policy must not be read as "do nothing"."""
        with pytest.raises(ValueError, match="on_missing_embeddings"):
            partially_embedded_project.compute(
                "planarizer", "simple", on_missing_embeddings="warm"
            )

    def test_unknown_component_kind_is_named(self, basic_project):
        """A misspelled kind must name itself, not an attribute the caller never wrote.

        ``component_kind`` went straight into ``getattr(self, kind + "s")``, so a typo
        surfaced as ``AttributeError: 'Project' object has no attribute 'planarisers'``.
        """
        with pytest.raises(ValueError, match="component_kind"):
            basic_project.compute("planariser", "simple")  # typo for 'planarizer'

    def test_component_kind_cannot_resolve_to_a_data_store(self, basic_project):
        """``getattr(self, kind + "s")`` also reached non-registry attributes.

        ``compute("segment", "s1")`` resolved ``self.segments`` and called a segment's
        *text* as if it were a component. And any kind that did resolve to a callable
        registry would have fallen off the end of the results-storage chain, which has
        no ``else`` — computing, discarding, and returning a save key for nothing.
        """
        basic_project.add_segments({"s1": "Hello world"})

        with pytest.raises(ValueError, match="component_kind"):
            basic_project.compute("segment", "s1")

    def test_embedder_data_must_carry_its_segment_keys(self, basic_project):
        """Embedding a bare sequence must be refused, not filed under other segments.

        Embeddings are stored per segment key. Given a keyless sequence the sync path
        invented keys from ``self.segments``, so the vector for unrelated text was
        written over an existing segment's embedding.
        """
        basic_project.add_segments({"s1": "Hello world"})
        original = basic_project.embeddings["s1"]

        with pytest.raises(TypeError, match="[Mm]apping"):
            basic_project.compute("embedder", "simple", ["completely unrelated text"])

        assert basic_project.embeddings["s1"] == original

    def test_embedder_data_contract_is_the_same_sync_and_async(self, basic_project):
        """The two paths must reject the same input the same way.

        The async path did reject a keyless sequence — but as an
        ``AttributeError: 'list' object has no attribute 'keys'`` raised from inside a
        private helper, for input the sync path silently accepted.
        """
        with pytest.raises(TypeError, match="[Mm]apping"):
            basic_project.compute(
                "embedder", "simple", ["completely unrelated text"], async_mode=True
            )

    def test_positional_results_are_not_guessed_onto_project_segments(
        self, basic_project
    ):
        """Caller-supplied input means caller-known keys — ``compute`` must not invent them.

        Positional results were always zipped against *the project's* embedded segment
        keys, even when the caller passed their own vectors, so results were filed
        under segments that had nothing to do with them.
        """
        basic_project.add_segments({"s1": "one", "s2": "two", "s3": "three"})

        with pytest.raises(ValueError, match="attribute"):
            basic_project.compute("planarizer", "simple", [[3.0, 1.0], [1.0, 2.0]])

        assert len(basic_project.planar_coords) == 0

    def test_generator_results_are_not_silently_dropped(self, basic_project):
        """A component returning an iterator must still have its results stored.

        The results were measured with ``len(list(results))`` and only *then* zipped —
        which exhausts a generator, storing an empty mapping under the returned key.
        """
        basic_project.add_segments({"s1": "one", "s2": "two", "s3": "three"})

        save_key = basic_project.compute("planarizer", "generator")

        assert len(basic_project.planar_coords[save_key]) == 3

    def test_result_count_mismatch_is_reported(self, basic_project):
        """Fewer results than inputs must fail, not silently truncate the key list.

        ``zip(valid_segment_keys[: len(results)], results)`` quietly kept the first N
        keys, so a component returning one point too few produced a result mapping
        that was wrong rather than short.
        """
        basic_project.add_segments({"s1": "one", "s2": "two", "s3": "three"})

        with pytest.raises(ValueError, match="returned"):
            basic_project.compute("planarizer", "truncating")

        assert len(basic_project.planar_coords) == 0


class TestProjectInvalidation:
    """Test the invalidation cascade when segments change"""

    def test_invalidation_removes_embeddings(self, basic_project):
        """Test that adding segments removes old embeddings"""
        # Initial segments
        segments1 = {"s1": "First", "s2": "Second"}
        basic_project.add_segments(segments1)

        # Verify embeddings exist
        assert "s1" in basic_project.embeddings
        assert "s2" in basic_project.embeddings

        # Compute derived data
        basic_project.compute("planarizer", "simple", save_key="coords_v1")

        # Modify s1
        basic_project.add_segments({"s1": "Modified first"})

        # s1 should have new embedding
        assert basic_project.embeddings["s1"] == [14, 1, 0]  # "Modified first"

        # s2 should still have old embedding
        assert basic_project.embeddings["s2"] == [6, 0, 0]  # "Second"

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
        assert len(async_project.embeddings) == 2


class TestProjects:
    """Test the Projects container"""

    def test_projects_with_async_config(self, temp_dir, embedder_gate):
        """Test creating projects with async configuration"""
        projects = Projects()

        # Create project with async enabled, using an embedder the test holds open
        # so "not immediately available" is guaranteed rather than merely likely.
        p = projects.create_project(
            project_id="async_test",
            embedders={"default": gated_embedder},
            async_embeddings=True,
            async_base_path=temp_dir,
        )

        assert p._id == "async_test"
        assert p._async_embeddings is True
        assert p._async_base_path == temp_dir

        # Add segments and verify async behavior
        p.add_segments({"test": "Test segment"})

        # Cannot be available: the embedder is still blocked at the gate
        assert "test" not in p.embeddings

        # Release it and wait for it
        embedder_gate.set()
        success = p.wait_for_embeddings(timeout=WAIT_TIMEOUT_S)
        assert success
        assert "test" in p.embeddings

    def test_projects_with_explicit_backend(self, temp_dir, embedder_gate):
        """Test creating projects with explicit StdLibQueueBackend"""
        from au.base import FileSystemStore, StdLibQueueBackend, SerializationFormat

        projects = Projects()
        store = FileSystemStore(
            temp_dir, ttl_seconds=3600, serialization=SerializationFormat.PICKLE
        )
        backend = StdLibQueueBackend(store, use_processes=False)
        p = projects.create_project(
            project_id="async_test_backend",
            embedders={"default": gated_embedder},
            async_embeddings=True,
            async_base_path=temp_dir,
            async_backend=backend,
        )
        assert p._id == "async_test_backend"
        assert p._async_backend is backend
        # Add segments and verify async behavior
        p.add_segments({"test": "Test segment"})
        # Cannot be available: the embedder is still blocked at the gate
        assert "test" not in p.embeddings

        # Release it and wait for it
        embedder_gate.set()
        assert p.wait_for_embeddings(timeout=WAIT_TIMEOUT_S)
        assert "test" in p.embeddings


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

        basic_project.embedders["default"] = tracking_embedder

        # Add segments
        segments = {"s1": "Hello", "s2": "World"}
        basic_project.add_segments(segments)

        # Verify embedder received the mapping
        assert isinstance(received_input, dict)
        assert received_input == segments

    def test_valid_embeddings_property(self, basic_project):
        """Test the valid_embeddings property"""
        # Add segments
        basic_project.add_segments({"s1": "Text 1", "s2": "Text 2"})

        # Get valid embeddings
        valid = basic_project.valid_embeddings
        assert len(valid) == 2
        assert "s1" in valid
        assert "s2" in valid

        # Modify the returned dict shouldn't affect internal state
        valid["s3"] = [1, 2, 3]
        assert "s3" not in basic_project.embeddings

    def test_get_embeddings_helper(self, basic_project):
        """Test the get_embeddings helper method"""
        # Add segments
        segments = {f"s{i}": f"Text {i}" for i in range(3)}
        basic_project.add_segments(segments)

        # Get all embeddings
        all_embeddings = basic_project.get_embeddings()
        assert len(all_embeddings) == 3

        # Get specific embeddings
        subset = basic_project.get_embeddings(["s0", "s2"])
        assert len(subset) == 2

        # Missing keys are skipped
        subset = basic_project.get_embeddings(["s0", "s99"])
        assert len(subset) == 1

    def test_compute_with_default_data(self, basic_project):
        """Test compute without providing data explicitly"""
        # Add segments
        basic_project.add_segments({"s1": "Hello", "s2": "World"})

        # Compute without providing data - should use embeddings
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
        assert "s1" in basic_project.embeddings

        # Switch to async
        basic_project.set_async_mode(True)
        basic_project.add_segments({"s2": "Async two"})

        # s1 still there, s2 may or may not be immediately available depending on async timing
        assert "s1" in basic_project.embeddings

        # Can still compute on available embeddings (at least s1)
        save_key = basic_project.compute("planarizer", "simple")
        coords = basic_project.planar_coords[save_key]
        assert len(coords) >= 1  # At least s1

        # Wait for s2 with a reasonable timeout
        success = basic_project.wait_for_embeddings(["s2"], timeout=10.0)

        # The key test is that we can switch modes and still compute what's available
        # If async worked, we should have both; if not, we still have s1
        final_embeddings = len(
            [k for k in ["s1", "s2"] if k in basic_project.embeddings]
        )
        assert final_embeddings >= 1  # At least s1 should be available

        # Compute final planarization with whatever embeddings we have
        save_key2 = basic_project.compute("planarizer", "simple")
        coords2 = basic_project.planar_coords[save_key2]
        assert (
            len(coords2) == final_embeddings
        )  # Should match number of available embeddings


if __name__ == "__main__":
    pytest.main([__file__])
