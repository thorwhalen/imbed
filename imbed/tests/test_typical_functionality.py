"""Typical functionality tests for imbed."""

from typing import Callable
from imbed.util import Query, SearchResults
from imbed.tests.utils_for_tests import general_test_for_search_function

# ─── Test Documents ────────────────────────────────────────────────────────────
docs = {
    'python': 'Python is a high‑level programming language emphasizing readability and rapid development.',
    'java': 'Java is a class‑based, object‑oriented language designed for portability across platforms.',
    'numpy': 'NumPy provides support for large, multi‑dimensional arrays and matrices, along with a collection of mathematical functions.',
    'pandas': 'Pandas is a Python library offering data structures and operations for manipulating numerical tables and time series.',
    'apple': 'Apple is a fruit that grows on trees and comes in varieties such as Granny Smith, Fuji, and Gala.',
    'banana': 'Banana is a tropical fruit with a soft, sweet interior and a peel that changes from green to yellow when ripe.',
    'microsoft': 'Microsoft develops software products including the Windows operating system, Office suite, and cloud services.',
}

# ─── Semantic Search Examples ─────────────────────────────────────────────────


def test_search_func(
    search_func: Callable[[Query], SearchResults],
):
    """
    Test the search function with multiple queries using the general test framework.
    """
    test_cases = [
        {
            'query': 'object‑oriented programming',
            'expected_docs': {'java', 'python', 'numpy'},
            'description': 'programming language search',
        },
        {
            'query': 'tropical fruit',
            'expected_docs': {'banana', 'apple'},
            'description': 'fruit category search',
        },
    ]

    general_test_for_search_function(
        test_cases=test_cases,
        search_func=search_func,
        docs=docs,
        test_name='search_func_tests',
    )


# ─── Retrieval‑Augmented Generation Example ────────────────────────────────────


def test_find_docs_to_answer_question(
    find_docs_to_answer_question: Callable[[Query], SearchResults],
):
    """
    Test the function that finds documents relevant to a question.
    """
    general_test_for_search_function(
        query='Which documents describe a fruit that is sweet and easy to eat?',
        expected_docs={'apple', 'banana'},
        search_func=find_docs_to_answer_question,
        docs=docs,
        test_name='question_answering_test',
    )


# ─── test these test functions with a docs_to_search_func factory function ──────


def test_search_func_factory(
    search_func_factory: Callable[[dict], Callable[[Query], SearchResults]],
):
    """
    Test the search function factory with a set of documents.
    """
    search_func = search_func_factory(docs)

    # Run the search function tests
    test_search_func(search_func)
    test_find_docs_to_answer_question(search_func)
