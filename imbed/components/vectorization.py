"""
Vectorization functions for converting text to embeddings
"""

from typing import Iterable, Mapping
from functools import partial
import string
import re

from contextlib import suppress
from imbed.util import get_config
from imbed.imbed_types import Vector, SingularSegmentVectorizer

suppress_import_errors = suppress(ImportError, ModuleNotFoundError)


def constant_vectorizer(segments):
    """Generate basic constant vector for each segment"""
    if isinstance(segments, dict):
        # Return a mapping if input is a mapping
        return {key: [0.1, 0.2, 0.3] for key in segments}
    else:
        # Return a list if input is a sequence
        return [[0.1, 0.2, 0.3] for _ in segments]


# ------------------------------------------------------------------------------
# Simple Placeholder Semantic features

import re


def _word_count(text: str) -> int:
    """
    Count the number of words in the text using `\b\w+\b` to match word boundaries.

    >>> _word_count("Hello, world!")
    2
    """
    return len(re.findall(r"\b\w+\b", text))


def _character_count(text: str) -> int:
    """
    Count the number of non-whitespace characters in the text using `\S` to match any non-whitespace character.

    >>> _character_count("Hello, world!")
    12
    """
    return len(re.findall(r"\S", text))


def _non_alphanumerics_count(text: str) -> int:
    """
    Count the number of non-alphanumeric, non-space characters in the text using `\W` and excluding spaces.

    >>> _non_alphanumerics_count("Hello, world!")
    2
    """
    return len(re.findall(r"[^\w\s]", text))


# A simple 3d feature vector
def three_text_features(text: str) -> Vector:
    """
    Calculate simple (pseudo-)semantic features of the text.
    This is meant to be used as a placeholder vectorizer (a.k.a. embedding function) for
    text segments, for testing mainly.

    >>> three_text_features("Hello, world!")
    (2, 12, 2)
    """
    return _word_count(text), _character_count(text), _non_alphanumerics_count(text)


def simple_text_embedder(text, stopwords=None):
    """
    Extracts a set of lightweight, linguistically significant features from a text segment.

    The function computes several features based on the input text including:
        - Total number of words.
        - Mean and median word lengths.
        - Number and ratio of stopwords.
        - Number of punctuation characters.
        - Total number of characters.
        - Number of sentences and average words per sentence.
        - Lexical diversity (ratio of unique words to total words).
        - Number of numeric tokens.
        - Number of capitalized words.

    Parameters:
        text (str): The text segment to analyze.
        stopwords (set, optional): A set of stopwords for counting. Defaults to an empty set.

    Returns:
        list: A list of numerical features representing the text, in the following order:
            [num_words, mean_word_length, median_word_length, num_stopwords, stopword_ratio,
            num_punctuation, num_characters, num_sentences, avg_words_per_sentence,
            lexical_diversity, num_numeric_tokens, num_capitalized_words]

    Example:
        >>> sample_text = "Hello, world! This is an example text, with 123 numbers and various punctuation marks."
        >>> sample_stopwords = {'this', 'is', 'an', 'with', 'and'}
        >>> simple_text_embedder(sample_text, sample_stopwords)  # doctest: +ELLIPSIS
        [14, 5.21..., 6, 5, 0.35..., 4, 86, 2, 7.0, 1.0, 1, 2]

    """
    # Use an empty set as default stopwords if none provided
    if stopwords is None:
        stopwords = set()

    # Basic tokenization: split text into words using whitespace.
    if not isinstance(text, str):
        if isinstance(text, Mapping):
            return {k: simple_text_embedder(v, stopwords) for k, v in text.items()}
        elif isinstance(text, Iterable):
            return [simple_text_embedder(_text, stopwords) for _text in text]
        raise ValueError("Input text must be a string or list of strings.")

    words = text.split()
    num_words = len(words)

    # Compute word lengths
    word_lengths = [len(word) for word in words]
    mean_word_length = sum(word_lengths) / num_words if num_words > 0 else 0
    median_word_length = sorted(word_lengths)[num_words // 2] if num_words > 0 else 0

    # Count stopwords (case-insensitive)
    num_stopwords = sum(1 for word in words if word.lower() in stopwords)
    stopword_ratio = num_stopwords / num_words if num_words > 0 else 0

    # Count punctuation characters
    num_punctuation = sum(1 for char in text if char in string.punctuation)

    # Total number of characters
    num_characters = len(text)

    # Split text into sentences using a simple regex pattern.
    sentences = re.split(r"[.!?]+", text)
    # Remove empty sentences that may result from trailing punctuation.
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = len(sentences)
    avg_words_per_sentence = num_words / num_sentences if num_sentences > 0 else 0

    # Lexical diversity: ratio of unique words to total words.
    unique_words = set(word.lower() for word in words)
    lexical_diversity = len(unique_words) / num_words if num_words > 0 else 0

    # Count numeric tokens (words that are purely digits)
    num_numeric_tokens = sum(1 for word in words if word.isdigit())

    # Count capitalized words (assuming proper nouns or sentence starts)
    num_capitalized_words = sum(1 for word in words if word[0].isupper())

    # Create the feature vector as a list of numbers.
    feature_vector = [
        num_words,  # Total number of words
        mean_word_length,  # Mean word length
        median_word_length,  # Median word length
        num_stopwords,  # Number of stopwords
        stopword_ratio,  # Ratio of stopwords to total words
        num_punctuation,  # Number of punctuation characters
        num_characters,  # Total number of characters in the text
        num_sentences,  # Number of sentences
        avg_words_per_sentence,  # Average words per sentence
        lexical_diversity,  # Lexical diversity ratio
        num_numeric_tokens,  # Number of numeric tokens
        num_capitalized_words,  # Number of capitalized words
    ]

    return feature_vector


three_text_features: SingularSegmentVectorizer
simple_text_embedder: SingularSegmentVectorizer

embedders = {
    "constant_vectorizer": constant_vectorizer,
    "simple_text_embedder": simple_text_embedder,
}


with suppress_import_errors:
    from oa import embeddings

    embedders.update(
        {
            "text-embedding-3-small": partial(
                embeddings, model="text-embedding-3-small"
            ),
            "text-embedding-3-large": partial(
                embeddings, model="text-embedding-3-large"
            ),
        }
    )


# NOTE: This line must come towards end of module, after all embedders are defined
from imbed.components.components_util import add_default_key

add_default_key(
    embedders,
    default_key=constant_vectorizer,
    enviornment_var="DEFAULT_IMBED_VECTORIZER_KEY",
)
