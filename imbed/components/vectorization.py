"""
Vectorization functions for converting text to embeddings
"""

from functools import partial

from contextlib import suppress
from imbed.util import get_config


suppress_import_errors = suppress(ImportError, ModuleNotFoundError)


async def constant_vectorizer(segments):
    """Generate basic 2D projections from embeddings"""
    return [0.1, 0.2, 0.3]


embedders = {
    "constant_vectorizer": constant_vectorizer,
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
