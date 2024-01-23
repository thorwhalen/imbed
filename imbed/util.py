"""Utils for imbed package."""

import os
import importlib.resources
from functools import partial, cached_property
from typing import Mapping, Callable, Optional, TypeVar, KT, Iterable, Any
from config2py import get_app_data_folder
from graze import (
    graze as _graze,
    Graze as _Graze,
    GrazeReturningFilepaths as _GrazeReturningFilepaths,
)

import re
import numpy as np

MappingFactory = Callable[..., Mapping]

package_name = 'imbed'

DFLT_DATA_DIR = get_app_data_folder(package_name, ensure_exists=True)
GRAZE_DATA_DIR = get_app_data_folder(
    os.path.join(package_name, 'graze'), ensure_exists=True
)
graze_kwargs = dict(
    rootdir=GRAZE_DATA_DIR,
    preget=_graze.preget_print_downloading_message_with_size,
)
graze = partial(_graze, **graze_kwargs)
grazed_path = partial(graze, return_filepaths=True)
Graze = partial(_Graze, **graze_kwargs)
GrazeReturningFilepaths = partial(_GrazeReturningFilepaths, **graze_kwargs)


def get_app_folder(name, *, ensure_exists=True):
    return get_app_data_folder(f'imbed/{name}', ensure_exists=ensure_exists)


non_alphanumeric_re = re.compile(r'\W+')


def lower_alphanumeric(text):
    return non_alphanumeric_re.sub(' ', text).strip().lower()


def hash_text(text):
    """Return a hash of the text, ignoring punctuation and capitalization.

    >>> (assert hash_text('Hello, world!')
    ...     ==  hash_text('hello world')
    ...     == '5eb63bbbe01eeed093cb22bb8f5acdc3'
    ... )

    """
    from hashlib import md5

    normalized_text = lower_alphanumeric(text)
    return md5(normalized_text.encode()).hexdigest()


def lenient_bytes_decoder(bytes_: bytes):
    if isinstance(bytes_, bytes):
        return bytes_.decode('utf-8', 'replace')
    return bytes_


def clog(condition, *args, log_func=print, **kwargs):
    """Conditional log

    >>> clog(False, "logging this")
    >>> clog(True, "logging this")
    logging this

    """
    if not args and not kwargs:
        import functools

        return functools.partial(clog, condition, log_func=log_func)
    if condition:
        log_func(*args, **kwargs)


# mdat utils


def is_submodule_path(path):
    path = str(path)
    return path.endswith('.py')


def module_name(path):
    name, ext = os.path.splitext(os.path.basename(path))
    return name


def submodules_of(pkg, include_init=True):
    f = importlib.resources.files(pkg)
    g = map(module_name, filter(is_submodule_path, f.iterdir()))
    if include_init:
        return g
    else:
        return filter(lambda name: name != '__init__', g)


EmbeddingKey = TypeVar('EmbeddingKey')
Metadata = Any
MetaFunc = Callable[[EmbeddingKey], Metadata]


class Embeddings:
    def __init__(
        self,
        embeddings,
        keys: Optional[Iterable[EmbeddingKey]] = None,
        *,
        meta: Optional[MetaFunc] = None,
        max_query_hits: int = 5,
    ):
        self.embeddings = np.array(embeddings)
        if keys is None:
            keys = range(len(embeddings))
        self.keys = keys
        self._meta = meta

    @classmethod
    def from_mapping(cls, mapping: Mapping[EmbeddingKey, object], *, meta):
        return cls(mapping.values(), mapping.keys())

    @classmethod
    def from_dataframe(cls, df, *, meta, embedding_col='embedding', key_col=None):
        if key_col is None:
            return cls(df[embedding_col], meta=meta)
        else:
            return cls(df[embedding_col], keys=df[key_col], meta=meta)

    def search(self, query_embedding, n=None):
        """Return the n closest embeddings to the query embedding."""
        from sklearn.metrics.pairwise import cosine_similarity

        n = n or self.max_query_hits
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), self.embeddings
        )
        return sorted(
            zip(self.keys, similarities[0]), key=lambda x: x[1], reverse=True
        )[:n]
