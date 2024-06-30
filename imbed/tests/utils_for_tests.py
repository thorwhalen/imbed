"""Utils for tests"""

import re

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


def simple_semantic_features(text: str) -> dict:
    """
    Calculate simple (pseudo-)semantic features of the text.

    >>> simple_semantic_features("Hello, world!")
    (2, 12, 2)
    """
    return word_count(text), character_count(text), non_alphanumerics_count(text)


# ------------------------------------------------------------------------------
