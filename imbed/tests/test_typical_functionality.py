"""Typical functionality tests for imbed."""

from typing import Callable
from imbed.tests.utils_for_tests import (
    general_test_for_search_function,
    Query,
    SearchResults,
)

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

def check_search_func(
    search_func: Callable[[Query], SearchResults],
):
    """
    Test the search function with multiple queries using the general test framework.
    """
    # Test case 1: programming language search
    general_test_for_search_function(
        query='object‑oriented programming',
        top_results_expected_to_contain={'java', 'python', 'numpy'},
        search_func=search_func,
    )

    # Test case 2: fruit category search
    general_test_for_search_function(
        query='tropical fruit',
        top_results_expected_to_contain={'banana', 'apple'},
        search_func=search_func,
    )


# ─── Retrieval‑Augmented Generation Example ────────────────────────────────────


def check_find_docs_to_answer_question(
    find_docs_to_answer_question: Callable[[Query], SearchResults],
):
    """
    Test the function that finds documents relevant to a question.
    """
    general_test_for_search_function(
        query='Which documents describe a fruit that is sweet and easy to eat?',
        top_results_expected_to_contain={'apple', 'banana'},
        search_func=find_docs_to_answer_question,
    )


# ─── test these test functions with a docs_to_search_func factory function ──────


def check_search_func_factory(
    search_func_factory: Callable[[dict], Callable[[Query], SearchResults]],
):
    """
    Test the search function factory with a set of documents.
    """
    search_func = search_func_factory(docs)

    # Run the search function tests
    check_search_func(search_func)
    check_find_docs_to_answer_question(search_func)
