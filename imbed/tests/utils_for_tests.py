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
from imbed.base import Vector

# ------------------------------------------------------------------------------
# Segmenters


def segmenter1(text):
    """
    Segment text into sentences using a period followed by a space as the delimiter.

    >>> list(segmenter1("This is a sentence. This is another."))
    ['This is a sentence.', 'This is another.']
    """
    segments = re.split(r'(?<=\.) ', text)
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
        yield ' '.join(words[i : i + chk_size])


# ------------------------------------------------------------------------------
# Semantic features


def word_count(text: str) -> int:
    """
    Count the number of words in the text using `\b\w+\b` to match word boundaries.

    >>> word_count("Hello, world!")
    2
    """
    return len(re.findall(r'\b\w+\b', text))


def character_count(text: str) -> int:
    """
    Count the number of non-whitespace characters in the text using `\S` to match any non-whitespace character.

    >>> character_count("Hello, world!")
    12
    """
    return len(re.findall(r'\S', text))


def non_alphanumerics_count(text: str) -> int:
    """
    Count the number of non-alphanumeric, non-space characters in the text using `\W` and excluding spaces.

    >>> non_alphanumerics_count("Hello, world!")
    2
    """
    return len(re.findall(r'[^\w\s]', text))


# A simple 3d feature vector
def simple_semantic_features(text: str) -> Vector:
    """
    Calculate simple (pseudo-)semantic features of the text.

    >>> simple_semantic_features("Hello, world!")
    (2, 12, 2)
    """
    return word_count(text), character_count(text), non_alphanumerics_count(text)


simple_embedding_vectorizer = simple_semantic_features


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
    Vector,
    SingularTextSegmenter,
    SingularSegmentVectorizer,
    SingularPlanarProjector,
)

simple_semantic_features: SingularSegmentVectorizer
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
        'doc1': ['Hello, world!'],
        'doc2': ['This is a test.', 'This test is only a test.'],
        'doc3': [
            'Segmenting text can be simple or complex.',
            'This test aims to make it simple.',
            "Let's see how it performs.",
        ],
    }
    for key, text in test_texts.items():
        assert (
            segmenter1(text) == expected_segments[key]
        ), f"Failed for {key}: {segmenter1(text)=}, {expected_segments[key]=}"


def test_segmenter2():
    expected_segments = {
        'doc1': ['Hello, world!'],
        'doc2': ['This is a test.', 'This test is only', 'a test.'],
        'doc3': [
            'Segmenting text can be',
            'simple or complex. This',
            'test aims to make',
            "it simple. Let's see",
            'how it performs.',
        ],
    }
    for key, text in test_texts.items():
        assert (
            list(segmenter2(text, chk_size=4)) == expected_segments[key]
        ), f"Failed for {key}: {list(segmenter2(text, chk_size=4))=}, {expected_segments[key]=}"


def test_simple_semantic_features_segmenter1():
    expected_features = {
        'doc1': [(2, 12, 2)],
        'doc2': [(4, 12, 1), (6, 20, 1)],
        'doc3': [(7, 35, 1), (7, 27, 1), (6, 22, 2)],
    }
    segments = {k: list(segmenter1(v)) for k, v in test_texts.items()}
    for key, segs in segments.items():
        computed_features = [simple_semantic_features(segment) for segment in segs]
        assert (
            computed_features == expected_features[key]
        ), f"Failed for {key} with segmenter1: {computed_features=}, {expected_features[key]=}"


def test_simple_semantic_features_segmenter2():
    expected_features = {
        'doc1': [(2, 12, 2)],
        'doc2': [(4, 12, 1), (4, 14, 0), (2, 6, 1)],
        'doc3': [(4, 19, 0), (4, 20, 1), (4, 14, 0), (5, 17, 2), (3, 14, 1)],
    }
    segments = {k: list(segmenter2(v, chk_size=4)) for k, v in test_texts.items()}
    for key, segs in segments.items():
        computed_features = [simple_semantic_features(segment) for segment in segs]
        assert (
            computed_features == expected_features[key]
        ), f"Failed for {key} with segmenter2: {computed_features=}, {expected_features[key]=}"


# # Run tests
# test_segmenter1()
# test_segmenter2()
# test_simple_semantic_features_segmenter1()
# test_simple_semantic_features_segmenter2()
