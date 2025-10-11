"""
Simplified interface for computing embeddings in bulk using OpenAI's batch API.

This module provides a clean, reusable interface for generating embeddings from
text segments using OpenAI's batch API, handling the async nature of the API
and providing status monitoring, error handling, and result aggregation.
"""

from warnings import warn

warn(f"oa.batch_embeddings moved to oa.batch_embeddings", DeprecationWarning)

from oa.batch_embeddings import *
