"""Utils for tests

```python
from functools import partial, cached_property
from dataclasses import dataclass
from typing import Mapping, Callable, MutableMapping

class Imbed:
    docs: Mapping = None
    segments: MutableMapping = None
    embedder: Callable = None

raw_docs = mk_text_store(doc_src_uri)  # the store used will depend on the source and format of where the docs are stored
segments = mk_segments_store(raw_docs, ...)  # will not copy any data over, but will give a key-value view of chunked (split) docs
search_ctrl = mk_search_controller(vectorDB, embedder, ...)
search_ctrl.fit(segments, doc_src_uri, ...)
search_ctrl.save(...)
```

"""

import re
from typing import Iterable, Callable, Any
from imbed.util import pkg_files

# ------------------------------------------------------------------------------
# Search functionality

Query = str
MaxNumResults = int
ResultT = Any
SearchResults = Iterable[ResultT]


def top_results_contain(results: SearchResults, expected: SearchResults) -> bool:
    """
    Check that the top results contain the expected elements.
    That is, the first len(expected) elements of results match the expected set,
    and if there are less results than expected, the only elements in results are
    contained in expected.
    """
    if len(results) < len(expected):
        return set(results) <= set(expected)
    return set(results[: len(expected)]) == set(expected)


def general_test_for_search_function(
    query,
    top_results_expected_to_contain: SearchResults,
    *,
    search_func: Callable[[Query], SearchResults],
    n_top_results=None,
):
    """
    General test function for search functionality.

    Args:
        query: Query string
        top_results_expected_to_contain: Set of expected document keys
        search_func: Search function to test (keyword-only)
        n_top_results: Number of top results to check. If None, defaults to min(len(results), len(top_results_expected_to_contain)) (keyword-only)

    Example use:

    >>> def search_docs_containing(query):
    ...     docs = {'doc1': 'apple pie recipe', 'doc2': 'car maintenance guide', 'doc3': 'apple varieties'}
    ...     return (key for key, text in docs.items() if query in text)
    >>> general_test_for_search_function(
    ...     query='apple',
    ...     top_results_expected_to_contain={'doc1', 'doc3'},
    ...     search_func=search_docs_containing
    ... )
    """
    # Execute search and collect results
    # TODO: Protect from cases where search_func(query) could be a long generator? Example, a max_results limit?
    results = list(search_func(query))

    # Determine the actual number of top results to check
    if n_top_results is None:
        effective_n_top_results = min(
            len(results), len(top_results_expected_to_contain)
        )
    else:
        effective_n_top_results = n_top_results

    # Get the slice of results to check
    top_results_to_check = results[:effective_n_top_results]

    # Generate helpful error message
    error_context = []
    error_context.append(f"Query: '{query}'")
    error_context.append(f"Expected docs: {top_results_expected_to_contain}")
    error_context.append(f"Actual results: {results}")
    error_context.append(
        f"Checking top {effective_n_top_results} results: {top_results_to_check}"
    )

    error_message = "\n".join(error_context)

    # Perform the assertion
    assert top_results_contain(
        top_results_to_check, top_results_expected_to_contain
    ), error_message


# ------------------------------------------------------------------------------
# Test data files
test_data_files = pkg_files / "tests" / "data"


# ------------------------------------------------------------------------------
# Segmenters


def segmenter1(text):
    """
    Segment text into sentences using a period followed by a space as the delimiter.

    >>> list(segmenter1("This is a sentence. This is another."))
    ['This is a sentence.', 'This is another.']
    """
    segments = re.split(r"(?<=\.) ", text)
    return segments


def segmenter2(text, chk_size=4):
    """
    Segment text into fixed-size chunks of words (up to chk_size words per chunk).

    >>> text = 'This, that, and the other! Something more!?!'
    >>> list(segmenter2(text))
    ['This, that, and the', 'other! Something more!?!']
    >>> list(segmenter2(text, chk_size=3))
    ['This, that, and', 'the other! Something', 'more!?!']
    """
    words = text.split()
    for i in range(0, len(words), chk_size):
        yield " ".join(words[i : i + chk_size])


# ------------------------------------------------------------------------------
# Simple Placeholder Semantic features

from imbed.components.vectorization import three_text_features

# ------------------------------------------------------------------------------
# Plane projection


def planar_projector(vectors):
    """
    Project vectors onto a plane of the two first dimensions.

    >>> vectors = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> list(planar_projector(vectors))
    [[1, 2], [4, 5], [7, 8]]

    """
    return (x[:2] for x in vectors)


# ------------------------------------------------------------------------------
# function types

from imbed.base import (
    SingularTextSegmenter,
    SingularPlanarProjector,
)

segmenter1: SingularTextSegmenter
segmenter2: SingularTextSegmenter
planar_projector: SingularPlanarProjector

# ------------------------------------------------------------------------------
# Data for tests.

test_texts = {
    "doc1": "Hello, world!",
    "doc2": "This is a test. This test is only a test.",
    "doc3": "Segmenting text can be simple or complex. This test aims to make it simple. Let's see how it performs.",
}


# ------------------------------------------------------------------------------
# Tests of utils for tests
def test_segmenter1():
    expected_segments = {
        "doc1": ["Hello, world!"],
        "doc2": ["This is a test.", "This test is only a test."],
        "doc3": [
            "Segmenting text can be simple or complex.",
            "This test aims to make it simple.",
            "Let's see how it performs.",
        ],
    }
    for key, text in test_texts.items():
        assert (
            segmenter1(text) == expected_segments[key]
        ), f"Failed for {key}: {segmenter1(text)=}, {expected_segments[key]=}"


def test_segmenter2():
    expected_segments = {
        "doc1": ["Hello, world!"],
        "doc2": ["This is a test.", "This test is only", "a test."],
        "doc3": [
            "Segmenting text can be",
            "simple or complex. This",
            "test aims to make",
            "it simple. Let's see",
            "how it performs.",
        ],
    }
    for key, text in test_texts.items():
        assert (
            list(segmenter2(text, chk_size=4)) == expected_segments[key]
        ), f"Failed for {key}: {list(segmenter2(text, chk_size=4))=}, {expected_segments[key]=}"


def test_three_text_features_segmenter1():
    expected_features = {
        "doc1": [(2, 12, 2)],
        "doc2": [(4, 12, 1), (6, 20, 1)],
        "doc3": [(7, 35, 1), (7, 27, 1), (6, 22, 2)],
    }
    segments = {k: list(segmenter1(v)) for k, v in test_texts.items()}
    for key, segs in segments.items():
        computed_features = [three_text_features(segment) for segment in segs]
        assert (
            computed_features == expected_features[key]
        ), f"Failed for {key} with segmenter1: {computed_features=}, {expected_features[key]=}"

    # # Run tests
    # test_segmenter1()
    # test_segmenter2()
    # test_three_text_features_segmenter1()
    # test_three_text_features_segmenter2()
    for key, text in test_texts.items():
        assert (
            segmenter1(text) == expected_segments[key]
        ), f"Failed for {key}: {segmenter1(text)=}, {expected_segments[key]=}"


def test_segmenter2():
    expected_segments = {
        "doc1": ["Hello, world!"],
        "doc2": ["This is a test.", "This test is only", "a test."],
        "doc3": [
            "Segmenting text can be",
            "simple or complex. This",
            "test aims to make",
            "it simple. Let's see",
            "how it performs.",
        ],
    }
    for key, text in test_texts.items():
        assert (
            list(segmenter2(text, chk_size=4)) == expected_segments[key]
        ), f"Failed for {key}: {list(segmenter2(text, chk_size=4))=}, {expected_segments[key]=}"


def test_three_text_features_segmenter1():
    expected_features = {
        "doc1": [(2, 12, 2)],
        "doc2": [(4, 12, 1), (6, 20, 1)],
        "doc3": [(7, 35, 1), (7, 27, 1), (6, 22, 2)],
    }
    segments = {k: list(segmenter1(v)) for k, v in test_texts.items()}
    for key, segs in segments.items():
        computed_features = [three_text_features(segment) for segment in segs]
        assert (
            computed_features == expected_features[key]
        ), f"Failed for {key} with segmenter1: {computed_features=}, {expected_features[key]=}"


# # Run tests
# test_segmenter1()
# test_segmenter2()
# test_three_text_features_segmenter1()
# test_three_text_features_segmenter2()
# test_segmenter1()
# test_segmenter2()
# test_three_text_features_segmenter1()
# test_three_text_features_segmenter2()
