"""Utils for imbed package."""

import os
import importlib.resources
from functools import partial, wraps
from itertools import islice
from typing import (
    Mapping,
    Callable,
    Optional,
    TypeVar,
    KT,
    Iterable,
    Any,
    Literal,
    Union,
    Coroutine,
    ParamSpec,
)
import asyncio

from config2py import get_app_data_folder, process_path, simple_config_getter
from lkj import clog as clog, print_with_timestamp, log_calls as _log_calls

from graze import (
    graze as _graze,
    Graze as _Graze,
    GrazeReturningFilepaths as _GrazeReturningFilepaths,
)

import re
import numpy as np


mk_factory = partial(
    partial, partial
)  # see https://medium.com/@thorwhalen1/partial-partial-partial-f90396901362

fullpath_factory = mk_factory(os.path.join)

MappingFactory = Callable[..., Mapping]

package_name = "imbed"
app_data_folder = os.environ.get(
    "IMBED_APP_DATA_FOLDER",
    get_app_data_folder(package_name, ensure_exists=True),
)

DFLT_DATA_DIR = process_path(app_data_folder, ensure_dir_exists=True)
GRAZE_DATA_DIR = process_path(DFLT_DATA_DIR, "graze", ensure_dir_exists=True)
DFLT_SAVES_DIR = process_path(DFLT_DATA_DIR, "saves", ensure_dir_exists=True)
DFLT_CONFIG_DIR = process_path(DFLT_DATA_DIR, "config", ensure_dir_exists=True)
DFLT_BATCHES_DIR = process_path(DFLT_DATA_DIR, "batches", ensure_dir_exists=True)


saves_join = fullpath_factory(DFLT_SAVES_DIR)
get_config = simple_config_getter(DFLT_CONFIG_DIR)

graze_kwargs = dict(
    rootdir=GRAZE_DATA_DIR,
    key_ingress=_graze.key_ingress_print_downloading_message_with_size,
)
graze = partial(_graze, **graze_kwargs)
grazed_path = partial(graze, return_filepaths=True)
Graze = partial(_Graze, **graze_kwargs)
GrazeReturningFilepaths = partial(_GrazeReturningFilepaths, **graze_kwargs)


non_alphanumeric_re = re.compile(r"\W+")


def dict_slice(d: Mapping, *args) -> dict:
    return dict(islice(d.items(), *args))


def identity(x):
    return x


def lower_alphanumeric(text):
    return non_alphanumeric_re.sub(" ", text).strip().lower()


def hash_text(text):
    """Return a hash of the text, ignoring punctuation and capitalization.

    >>> hash_text('Hello, world!')
    '5eb63bbbe01eeed093cb22bb8f5acdc3'
    >>> hash_text('hello world')
    '5eb63bbbe01eeed093cb22bb8f5acdc3'
    >>> hash_text('Hello, world!') == hash_text('hello world')
    True

    """
    from hashlib import md5

    normalized_text = lower_alphanumeric(text)
    return md5(normalized_text.encode()).hexdigest()


def lenient_bytes_decoder(bytes_: bytes):
    if isinstance(bytes_, bytes):
        return bytes_.decode("utf-8", "replace")
    return bytes_


# decorator that logs calls
log_calls = _log_calls(
    logger=print_with_timestamp,
)

# decorator that logs calls of methods if the instance verbose flat is set
log_method_calls = _log_calls(
    logger=print_with_timestamp,
    log_condition=partial(_log_calls.instance_flag_is_set, flag_attr="verbose"),
)


def async_sync_wrapper(func):
    """
    A decorator that adds an async and a sync version of a function.
    """

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    func.async_version = async_wrapper
    func.sync_version = sync_wrapper
    return func


P = ParamSpec("P")
R = TypeVar("R")


async def async_call(
    func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
) -> Coroutine[Any, Any, R]:
    """
    Calls a function, awaiting it if it's asynchronous, and running it in a thread if it's synchronous.

    Args:
        func: The function to call.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the function call.
    """
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    else:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)


# --------------------------------------------------------------------------------------
# mdat utils


def is_submodule_path(path):
    path = str(path)
    return path.endswith(".py")


def module_name(path):
    name, ext = os.path.splitext(os.path.basename(path))
    return name


def submodules_of(pkg, include_init=True):
    f = importlib.resources.files(pkg)
    g = map(module_name, filter(is_submodule_path, f.iterdir()))
    if include_init:
        return g
    else:
        return filter(lambda name: name != "__init__", g)


EmbeddingKey = TypeVar("EmbeddingKey")
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
    def from_dataframe(cls, df, *, meta, embedding_col="embedding", key_col=None):
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


def cosine_similarity(u, v, *, cartesian_product=False):
    """
    Computes the cosine similarity between two vectors or arrays of vectors.

    If both inputs are 1D vectors, returns a float.
    If one or both inputs are 2D arrays, returns either a 1D array (row-wise)
    or a 2D array (cartesian product of rows) depending on the cartesian_product flag.

    Behavior for row-wise (cartesian_product=False):
      - If both arrays have the same number of rows, compares row i of u to row i of v.
      - If one array has only 1 row, it is broadcast against each row of the other array.
        (Returns a 1D array of length k, where k is the number of rows in the multi-row array.)

    Args:
        u (array-like): A single vector (1D) or a 2D array (k1 x n),
                        where each row is a separate vector.
        v (array-like): A single vector (1D) or a 2D array (k2 x n).
        cartesian_product (bool, optional):
            - If False (default), the function compares rows in a one-to-one fashion (u[i] vs. v[i]),
              **except** if one array has exactly 1 row and the other has multiple rows, in which case
              that single row is broadcast to all rows of the other array.
            - If True, computes the similarity for every combination of rows
              (results in a 2D array of shape (k1, k2)).

    Returns:
        float or np.ndarray:
            - A float if both u and v are 1D vectors.
            - A 1D numpy array if either u or v is 2D and cartesian_product=False.
            - A 2D numpy array if cartesian_product=True.

    Raises:
        ValueError:
            - If the number of columns in u and v do not match.
            - If cartesian_product=False, both arrays have multiple rows but differ in row count.

    Examples
    --------

    `See here for an explanation of the cases <https://github.com/thorwhalen/imbed/discussions/9#discussioncomment-11968528>`_.

    `See here for a performance comparison of numpy (this function) versus scipy <https://github.com/thorwhalen/imbed/discussions/9#discussioncomment-11971474>`_.

    Case 1: Both are single 1D vectors

    >>> u1d = [2, 0]
    >>> v1d = [2, 0]
    >>> float(cosine_similarity(u1d, v1d))
    1.0

    Case 2: Single 1D vector vs. a 2D array (row-wise broadcast)

    >>> import numpy as np
    >>> M1 = np.array([
    ...     [2, 0],
    ...     [0, 2],
    ...     [2, 2]
    ... ])
    >>> cosine_similarity(u1d, M1)  # doctest: +ELLIPSIS
    array([1.        , 0.        , 0.70710678...])

    Case 3: Two 2D arrays of different row lengths, cartesian_product=False (raises ValueError)

    >>> M2_different = np.array([
    ...     [0, 2],
    ...     [2, 2]
    ... ])
    >>> # Expect a ValueError because M1 has 3 rows and M2_different has 2 rows
    >>> cosine_similarity(M1, M2_different, cartesian_product=False)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: For row-wise comparison, u and v must have the same number of rows...

    Case 4: Two 2D arrays of the same number of rows, cartesian_product=False

    >>> M2 = np.array([
    ...     [0, 2],
    ...     [2, 0],
    ...     [2, 2]
    ... ])
    >>> cosine_similarity(M1, M2, cartesian_product=False)
    array([0., 0., 1.])

    Case 5: Two 2D arrays of the same size, `cartesian_product=True`
    (computes every combination of rows => 3 x 3)

    >>> res5 = cosine_similarity(M1, M2, cartesian_product=True)
    >>> np.round(res5, 3)  # doctest: +NORMALIZE_WHITESPACE
    array([[0.   , 1.   , 0.707],
           [1.   , 0.   , 0.707],
           [0.707, 0.707, 1.   ]])
    """
    # Convert inputs to numpy arrays
    u = np.asarray(u)
    v = np.asarray(v)

    # --------------- CASE 1: Both are single 1D vectors ---------------
    if u.ndim == 1 and v.ndim == 1:
        if u.shape[0] != v.shape[0]:
            raise ValueError("Vectors u and v must have the same dimension.")
        dot_uv = np.dot(u, v)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        return dot_uv / (norm_u * norm_v)

    # --------------- CASE 2: At least one is 2D; ensure both are 2D ---------------
    if u.ndim == 1:  # shape (n,) -> (1, n)
        u = u[np.newaxis, :]
    if v.ndim == 1:  # shape (n,) -> (1, n)
        v = v[np.newaxis, :]

    k1, n1 = u.shape
    k2, n2 = v.shape

    # Check that columns (vector dimension) match
    if n1 != n2:
        raise ValueError(
            f"Inconsistent dimensions: u has {n1} columns, v has {n2} columns."
        )

    # --------------- CARTESIAN PRODUCT ---------------
    if cartesian_product:
        # (k1 x k2) dot products
        dot_uv = u @ v.T  # shape (k1, k2)
        norm_u = np.linalg.norm(u, axis=1)  # shape (k1,)
        norm_v = np.linalg.norm(v, axis=1)  # shape (k2,)
        # Outer product of norms => shape (k1, k2)
        denom = np.outer(norm_u, norm_v)
        return dot_uv / denom

    # --------------- ROW-WISE (NOT CARTESIAN) ---------------
    # 1) If one array has a single row (k=1), broadcast it against each row of the other
    if k1 == 1 and k2 > 1:
        # Broadcast u's single row against each row in v
        dot_uv = np.sum(u[0] * v, axis=1)  # shape (k2,)
        norm_u = np.linalg.norm(u[0])  # scalar
        norm_v = np.linalg.norm(v, axis=1)  # shape (k2,)
        return dot_uv / (norm_u * norm_v)

    if k2 == 1 and k1 > 1:
        # Broadcast v's single row against each row in u
        dot_uv = np.sum(u * v[0], axis=1)  # shape (k1,)
        norm_u = np.linalg.norm(u, axis=1)  # shape (k1,)
        norm_v = np.linalg.norm(v[0])  # scalar
        return dot_uv / (norm_u * norm_v)

    # 2) Otherwise, require the same number of rows
    if k1 != k2:
        raise ValueError(
            f"For row-wise comparison, u and v must have the same number of rows. "
            f"(u has {k1}, v has {k2})"
        )
    dot_uv = np.sum(u * v, axis=1)  # shape (k1,)
    norm_u = np.linalg.norm(u, axis=1)
    norm_v = np.linalg.norm(v, axis=1)
    return dot_uv / (norm_u * norm_v)


def transpose_iterable(iterable):
    """
    This is useful to do things like:

    >>> xy_values = [(1, 2), (3, 4), (5, 6)]
    >>> x_values, y_values = transpose_iterable(xy_values)
    >>> x_values
    (1, 3, 5)
    >>> y_values
    (2, 4, 6)

    Note that transpose_iterable is an [involution](https://en.wikipedia.org/wiki/Involution_(mathematics))
    (if we disregard types).

    >>> list((x_values, y_values))
    [(1, 3, 5), (2, 4, 6)]

    """
    return zip(*iterable)


# umap utils ---------------------------------------------------------------------------

from typing import Mapping, Dict, KT, Tuple, Sequence, Optional
from imbed.imbed_types import (
    EmbeddingsDict,
    EmbeddingType,
    PlanarEmbedding,
    PlanarEmbeddingsDict,
)


def ensure_embedding_dict(embeddings: EmbeddingsDict) -> EmbeddingsDict:
    """
    Ensure that the embeddings are in the correct format.

    :param embeddings: a dict of embeddings
    :return: a dict of embeddings

    """
    if isinstance(embeddings, pd.DataFrame):
        raise TypeError(
            "Expected a Mapping, but got a DataFrame. "
            "Convert this DataFrame to a Mapping of embeddings first."
        )
    elif isinstance(embeddings, pd.Series):
        embeddings = embeddings.to_dict()
    elif isinstance(embeddings, (Sequence, np.ndarray)):
        embeddings = dict(enumerate(embeddings))
    else:
        # Make sure kd_embeddings is a Mapping with embedding values
        assert isinstance(
            embeddings, Mapping
        ), f"Expected a Mapping, but got {type(embeddings)}: {embeddings}"
        first_embedding = next(iter(embeddings.values()))
        if isinstance(first_embedding, np.ndarray):
            if first_embedding.ndim != 1:
                raise ValueError(
                    f"Expected kd_embeddings to be a Mapping with unidimensional values, "
                    f"but got {first_embedding.ndim} dimensions: {first_embedding}"
                )
        elif not isinstance(first_embedding, Sequence):
            raise ValueError(
                f"Expected kd_embeddings to be a Mapping with Sequence values, "
                f"but got {type(first_embedding)}: {first_embedding}"
            )

    return embeddings


PlanarEmbeddingKind = Literal["umap", "ncvis", "tsne", "pca"]
PlanarEmbeddingFunc = Callable[[Iterable[EmbeddingType]], Iterable[PlanarEmbedding]]
DFLT_PLANAR_EMBEDDING_KIND = "umap"


def planar_embeddings_func(
    embeddings_func: Optional[Union[PlanarEmbeddingKind]] = DFLT_PLANAR_EMBEDDING_KIND,
    *,
    distance_metric="cosine",
) -> PlanarEmbeddingFunc:
    if callable(embeddings_func):
        return embeddings_func
    elif isinstance(embeddings_func, str):
        if embeddings_func == "umap":
            import umap  # pip install umap-learn

            return umap.UMAP(n_components=2, metric=distance_metric).fit_transform
        elif embeddings_func == "tsne":
            from sklearn.manifold import TSNE

            return TSNE(n_components=2, metric=distance_metric).fit_transform
        elif embeddings_func == "pca":
            # Note: Here we don't simply apply PCA, but normalize it first to make
            # it appropriate for cosine similarity
            from sklearn.preprocessing import normalize, FunctionTransformer
            from sklearn.decomposition import PCA
            from sklearn.pipeline import Pipeline

            l2_normalization = FunctionTransformer(
                lambda X: normalize(X, norm="l2"), validate=True
            )

            return Pipeline(
                [("normalize", l2_normalization), ("pca", PCA(n_components=2))]
            ).fit_transform
        elif embeddings_func == "ncvis":
            import ncvis  # To install, see https://github.com/cosmograph-org/priv_cosmo/discussions/1#discussioncomment-9579428

            return ncvis.NCVis(d=2, distance=distance_metric).fit_transform
        else:
            raise ValueError(f"Not a valid planar embedding kind: {embeddings_func}")
    else:
        raise TypeError(f"Not a valid planar embedding type: {embeddings_func}")


PlanarEmbeddingSpec = Union[PlanarEmbeddingKind, PlanarEmbeddingFunc]

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

DFLT_PREPROCESS = make_pipeline(StandardScaler(), PCA()).fit_transform


def planar_embeddings(
    kd_embeddings: EmbeddingsDict,
    *,
    embeddings_func: PlanarEmbeddingSpec = DFLT_PLANAR_EMBEDDING_KIND,
    preprocess=DFLT_PREPROCESS,
) -> PlanarEmbeddingsDict:
    """Takes a mapping of k-dimensional (kd) embeddings and returns a dict of the 2d
    umap embeddings

    :param kd_embeddings: a dict of kd embeddings
    :param embeddings_func: the function to compute the embeddings
    :param preprocessors: a list of preprocessors to apply to the embeddings
    :return: a dict of the 2d umap embeddings


    Example:

    >>> # Make a random array of 7 vectors of dimension 3
    >>> import numpy as np
    >>> kd_embeddings = {i: np.random.rand(3) for i in range(7)}
    >>> xy_pairs = planar_embeddings(kd_embeddings)
    >>> xy_pairs  # doctest: +SKIP
    {0: (0.1, 0.2), 1: (0.3, 0.4), 2: (0.5, 0.6), 3: (0.7, 0.8), 4: (0.9, 0.1), 5: (0.2, 0.3),
    >>> x, y = planar_embeddings.transpose_iterable(xy_pairs.values())
    >>> x  # doctest: +SKIP
    (0.1, 0.3, 0.5, 0.7, 0.9, 0.2)
    >>> y  # doctest: +SKIP
    (0.2, 0.4, 0.6, 0.8, 0.1, 0.3)

    Tip: Should you normalize your features (use preprocessors, the default here)?
        See https://umap-learn.readthedocs.io/en/latest/faq.html?utm_source=chatgpt.com#should-i-normalise-my-features

    Tip: If you need to get big vectors of the x and y coordinates, you can do this:

    ```
    x_values, y_values = zip(*planar_embeddings(kd_embeddings).values())
    ```

    Or even, in case you have a pandas dataframe or dict d:

    ```
    d['x'], d['y'] = zip(*planar_embeddings(d).values())
    ```

    Tip: Use planar_embeddings.transpose_iterable to do this in a readabble way:

    ```
    x_values, y_values = planar_embeddings.transpose_iterable(planar_embeddings(kd_embeddings).values())
    ```

    """
    # get a function to compute the embeddings
    embeddings_func = planar_embeddings_func(embeddings_func)

    # make sure the input embeddings have a mapping interface
    kd_embeddings = ensure_embedding_dict(kd_embeddings)

    get_vector = lambda: np.array(list(kd_embeddings.values()))

    if preprocess:
        embedding_vectors = embeddings_func(preprocess(get_vector()))
    else:
        embedding_vectors = embeddings_func(get_vector())

    return {k: tuple(v) for k, v in zip(kd_embeddings.keys(), embedding_vectors)}


planar_embeddings.transpose_iterable = transpose_iterable  # to have it handy


umap_2d_embeddings = partial(planar_embeddings, embeddings_func="umap")

import pandas as pd


def planar_embeddings_dict_to_df(
    planar_embeddings_kv: PlanarEmbeddingsDict,
    *,
    x_col: str = "x",
    y_col: str = "y",
    index_name: Optional[str] = "id_",
    key_col: Optional[str] = None,
) -> pd.DataFrame:
    """A function that takes a dict of planar embeddings and returns a pandas DataFrame
    of the 2d embeddings

    If key_col is not None, the keys are added as a column in the dataframe.

    :param planar_embeddings_kv: a dict of planar embeddings
    :param x_col: the name of the x column
    :param y_col: the name of the y column
    :param index_name: the name of the index
    :param key_col: if you want to add a column with the index values copied into them
    :return: a pandas DataFrame of the 2d embeddings

    Example:

    >>> planar_embeddings_kv = {1: (0.1, 0.2), 2: (0.3, 0.4)}
    >>> planar_embeddings_dict_to_df(planar_embeddings_kv)  # doctest: +NORMALIZE_WHITESPACE
           x    y
    id_
    1    0.1  0.2
    2    0.3  0.4

    """
    df = pd.DataFrame(
        index=planar_embeddings_kv.keys(),
        data=planar_embeddings_kv.values(),
        columns=[x_col, y_col],
    ).rename_axis(index_name)

    if key_col is not None:
        if key_col is True:
            key_col = index_name  # default key column name is the index name
        df[key_col] = df.index

    return df


two_d_embedding_dict_to_df = planar_embeddings_dict_to_df  # back-compatibility alias


def umap_2d_embeddings_df(
    kd_embeddings: Mapping[KT, Sequence],
    *,
    x_col: str = "x",
    y_col: str = "y",
    index_name: Optional[str] = "id_",
    key_col: Optional[str] = None,
) -> pd.DataFrame:
    """A function that takes a mapping of kd embeddings and returns a pandas DataFrame
    of the 2d umap embeddings"""
    return planar_embeddings_dict_to_df(
        umap_2d_embeddings(kd_embeddings),
        x_col=x_col,
        y_col=y_col,
        index_name=index_name,
        key_col=key_col,
    )


# --------------------------------------------------------------------------------------
# data store utils
#
# A lot of what is defined here are functions that are used to transform data.
# More precisely, encode and decode data depending on it's format, file extension, etc.
# TODO: Merge with codec-matching ("routing"?) functionalities of dol

# TODO: Moved a bunch of stuff to tabled.wrappers. Importing here for back-compatibility
#    but should be removed in the future.
from tabled.wrappers import (
    get_extension,  # Return the extension of a file path
    if_extension_not_present_add_it,  # Add an extension to a file path if it's not already there
    if_extension_present_remove_it,  # Remove an extension from a file path if it's there
    save_df_to_zipped_tsv,  # Save a dataframe to a zipped TSV file
    extension_to_encoder,  # Dictionary mapping extensions to encoder functions
    extension_to_decoder,  # Dictionary mapping extensions to decoder functions
    get_codec_mappings,  # Get the current encoder and decoder mappings
    print_current_mappings,  # Print the current encoder and decoder mappings
    add_extension_codec,  # Add an extension-based encoder and decoder to the extension-code mapping
    extension_based_wrap,  # Add extension-based encoding and decoding to a store,
    auto_decode_bytes,  # Decode bytes to a string if it's a bytes object
)

# TODO: Use dol tools for this.
# Make a codecs for imbed
import json
import pickle
import io
from dol import Pipe, written_bytes


extension_to_encoder = {
    "txt": lambda obj: obj.encode("utf-8"),
    "json": json.dumps,
    "pkl": pickle.dumps,
    "parquet": written_bytes(pd.DataFrame.to_parquet, obj_arg_position_in_writer=0),
    "npy": written_bytes(np.save, obj_arg_position_in_writer=1),
    "csv": written_bytes(pd.DataFrame.to_csv),
    "xlsx": written_bytes(pd.DataFrame.to_excel),
    "tsv": written_bytes(
        partial(pd.DataFrame.to_csv, sep="\t", escapechar="\\", quotechar='"')
    ),
}

extension_to_decoder = {
    "txt": lambda obj: obj.decode("utf-8"),
    "json": json.loads,
    "pkl": pickle.loads,
    "parquet": Pipe(io.BytesIO, pd.read_parquet),
    "npy": Pipe(io.BytesIO, partial(np.load, allow_pickle=True)),
    "csv": Pipe(auto_decode_bytes, io.StringIO, pd.read_csv),
    "xlsx": Pipe(io.BytesIO, pd.read_excel),
    "tsv": Pipe(
        io.BytesIO, partial(pd.read_csv, sep="\t", escapechar="\\", quotechar='"')
    ),
}

from tabled.wrappers import (
    extension_based_encoding as _extension_based_encoding,  # Encode a value based on the extension of the key
    extension_based_decoding as _extension_based_decoding,  # Decode a value based on the extension of the key
)

extension_based_encoding = partial(
    _extension_based_encoding, extension_to_encoder=extension_to_encoder
)
extension_based_decoding = partial(
    _extension_based_decoding, extension_to_decoder=extension_to_decoder
)

# --------------------------------------------------------------------------------------
# Matching utils
#

import re
from typing import List, Dict, Callable, Union, Optional, TypeVar

Role = TypeVar("Role", bound=str)
Field = TypeVar("Field", bound=str)
Regex = TypeVar("Regex", bound=str)


# TODO: Move, or copy, to doodad
def match_aliases(
    fields: List[Field],
    aliases: Dict[
        Role, Union[List[Field], Regex, Callable[[List[Field]], Optional[Field]]]
    ],
) -> Dict[Role, Optional[Field]]:
    """
    Matches the keys of aliases to the given fields,
    using the values of aliases as the matching logic (could be a list of possible
    fields, a regular expression, or a custom matching function.).

    A dictionary

    Args:
        fields (List[Field]): A list of fields
        aliases (Dict[Role, Union[List[Field], Regex, Callable[[List[Field]], Optional[Field]]]]): A dictionary where:
            - Keys are roles (e.g., 'ID', 'Name') we're looking for
            - Values are either:
                - A list of field "aliases" (e.g., ['id', 'user_id']).
                - A string representing a regular expression (e.g., r'user.*id').
                - A function that takes a list of fields and returns a matched field or None.

    Returns:
        Dict[Role, Optional[Field]]: A dictionary mapping each role to the first matching
                                     field found in `fields`, or `None` if no match is
                                     found. Once a column is matched, it is removed
                                     from further matching, so it can't be matched again.


    Example 1: List-based aliases, regex, and custom function matching

    >>> fields = ['user_id', 'full_name', 'created_at', 'email_address']
    >>> aliases = {
    ...     'ID': ['id', 'user_id'],  # List of possible aliases for 'ID'
    ...     'Name': r'.*name',  # Regular expression for 'Name'
    ...     'Date': lambda cols: next((col for col in cols if "date" in col.lower() or "created" in col.lower()), None)  # Custom matching function
    ... }
    >>> match_aliases(fields, aliases)
    {'ID': 'user_id', 'Name': 'full_name', 'Date': 'created_at'}

    # Example 2: Handles conflict resolution by removing matched columns

    >>> fields = ['id', 'full_name', 'id_created', 'email_address']
    >>> aliases = {
    ...     'Primary ID': ['id'],  # List-based alias that should match 'id' first
    ...     'Secondary ID': r'id.*',  # Regex to match anything starting with 'id'
    ...     'Email': lambda cols: next((col for col in cols if 'email' in col.lower()), None)  # Custom function for email
    ... }
    >>> match_aliases(fields, aliases)
    {'Primary ID': 'id', 'Secondary ID': 'id_created', 'Email': 'email_address'}
    """

    def normalize_alias(
        value: Union[List[str], str, Callable[[List[str]], Optional[str]]],
    ) -> Callable[[List[str]], Optional[str]]:
        """Converts the alias to a matching function."""
        if isinstance(value, list):
            # Convert the list into a regular expression
            pattern = "|".join(re.escape(alias) for alias in value)
            return lambda columns: next(
                (col for col in columns if re.fullmatch(pattern, col)), None
            )
        elif isinstance(value, str):
            # Treat the string as a regular expression
            return lambda columns: next(
                (col for col in columns if re.fullmatch(value, col)), None
            )
        elif callable(value):
            # It's already a matching function
            return value
        else:
            raise ValueError("Alias must be a list, string, or callable.")

    # Normalize all alias entries into functions
    alias_functions = {role: normalize_alias(alias) for role, alias in aliases.items()}

    role_to_column = {role: None for role in aliases}  # Initialize result dictionary
    remaining_columns = set(fields)  # Set of columns that haven't been matched yet

    # Process each role and its corresponding matching function
    for role, match_func in alias_functions.items():
        matched_column = match_func(
            list(remaining_columns)
        )  # Apply the matching function to the remaining columns
        if matched_column:
            role_to_column[role] = matched_column
            remaining_columns.remove(
                matched_column
            )  # Remove the matched column from further consideration

    return role_to_column


# --------------------------------------------------------------------------------------
# TODO: Deprecated: Replaced by dol.cache_this
def load_if_saved(
    key=None,
    store_attr="saves",
    save_on_compute=True,
    print_when_loading_from_file=True,
):
    """
    Decorator to load the value from the store if it is saved, otherwise compute it.
    """
    from functools import wraps

    if callable(key):
        # Assume load_if_saved is being called on the method and that the key should
        # be the method name.
        method = key
        key = name_of_obj(method)
        return load_if_saved(key, store_attr, save_on_compute=save_on_compute)

    def _load_if_saved(method):
        wraps(method)

        def _method(self):
            store = getattr(self, store_attr)
            if key in store:
                if print_when_loading_from_file:
                    print(f"Loading {key} from file")
                return store[key]
            else:
                obj = method(self)
                if save_on_compute:
                    store[key] = obj
                return obj

        return _method

    return _load_if_saved


# --------------------------------------------------------------------------------------
# data manipulation

MatrixData = Union[np.ndarray, pd.DataFrame]


def merge_data(
    data_1: MatrixData,
    data_2: MatrixData,
    *,
    merge_on=None,
    data_1_cols: Optional[List[str]] = None,
    data_2_cols: Optional[List[str]] = None,
    column_index_cursor_start: int = 0,
) -> pd.DataFrame:
    """Merges two sources of data, returning a dataframe.

    The sources of data could be numpy arrays or pandas DataFrames.

    If they're both dataframes, the merge_on specification is needed.
    If at least one of them is a numpy array, data_1 and data_2 must have the same
    number of rows and merge_on is ignored, since the merge will simply be the
    concatination of the two datas over the rows (that is, the result will have
    that common number of rows and the number of columns will be added).

    The optional data_1_cols and data_2_cols are used to transform numpy matrices into
    dataframes with the given column names.

    :param data_1: The first source of data.
    :param data_2: The second source of data.
    :param merge_on: The column to merge on, if both data_1 and data_2 are dataframes.
    :param data_1_cols: The column names for the first source of data, if it is a numpy array.
    :param data_2_cols: The column names for the second source of data, if it is a numpy array.

    """
    column_index_cursor = column_index_cursor_start

    # if only one of the data sources is a numpy array, we need to get the
    # row indices of the dataframe data to use when making a dataframe for the array
    data_1_row_indices = list(range(len(data_1)))
    data_2_row_indices = list(range(len(data_2)))
    if isinstance(data_1, pd.DataFrame):
        data_1_row_indices = data_1.index.values
    if isinstance(data_2, pd.DataFrame):
        data_2_row_indices = data_2.index.values

    if isinstance(data_1, np.ndarray):
        if data_1_cols is None:
            data_1_cols = list(range(data_1.shape[1]))
            column_index_cursor += len(data_1_cols)
        data_1 = pd.DataFrame(data_1, columns=data_1_cols, index=data_2_row_indices)

    if isinstance(data_2, np.ndarray):
        assert len(data_2) == len(data_1), (
            f"Data 1 and Data 2 must have the same length. Instead, we got: "
            f"{len(data_1)} and {len(data_2)}"
        )
        if data_2_cols is None:
            data_2_cols = list(
                range(column_index_cursor, column_index_cursor + data_2.shape[1])
            )
        data_2 = pd.DataFrame(data_2, columns=data_2_cols, index=data_1_row_indices)

    if merge_on is not None:
        return data_1.merge(data_2, on=merge_on)
    else:
        return pd.concat([data_1, data_2], axis=1)


def counts(sr: pd.Series) -> pd.Series:
    # return pd.Series(dict(Counter(sr).most_common()))
    return sr.value_counts()


# --------------------------------------------------------------------------------------
# more misc

from typing import Union, MutableMapping, Any
from dol import Files, add_extension
from config2py import process_path
from lkj import print_progress


CacheSpec = Union[str, MutableMapping]


def is_string_with_path_seps(x: Any):
    return isinstance(x, str) and os.path.sep in x


def ensure_cache(cache: CacheSpec) -> MutableMapping:
    if isinstance(cache, str):
        rootdir = process_path(cache, ensure_dir_exists=1)
        return Files(rootdir)
        # if os.path.isdir(cache):
        #     rootdir = process_path(cache, ensure_dir_exists=1)
        #     return Files(rootdir)
        # else:
        #     raise ValueError(f"cache directory {cache} does not exist")
    elif isinstance(cache, MutableMapping):
        return cache
    else:
        raise TypeError(f"cache must be a str or MutableMapping, not {type(cache)}")


def ensure_fullpath(filepath: str, conditional_rootdir: str = "") -> str:
    """Ensures a full path, prepending a rootdir if input is a (slash-less) file name.

    If you pass in a file name, it will be considered relative to the current directory.
    In all other situations, the conditional_rootdir is ignored, and the filepath is
    taken at face value.
    All outputs will be processed to ensure a full path is returned.

    >>> ensure_fullpath('apple/sauce')  # doctest: +ELLIPSIS
    '.../apple/sauce'
    >>> assert (
    ...     ensure_fullpath('apple/sauce')
    ...     == ensure_fullpath('./apple/sauce')
    ...     == ensure_fullpath('apple/sauce', '')
    ... )

    The only time you actually use the rootdir is when you pass in a file name
    that doesn't have slashes in it.

    >>> ensure_fullpath('apple', '/root/dir')
    '/root/dir/apple'

    """
    if not is_string_with_path_seps(filepath):  # then consider it a file name
        # ... and instead of taking the file name to be relative to the current
        # directory, we'll take it to be relative to the conditional_rootdir.
        filepath = process_path(filepath, rootdir=conditional_rootdir)
    # elif conditional_rootdir:
    #     warnings.warn(
    #         f"ignoring rootdir {conditional_rootdir} for full path {filepath}"
    #     )

    return process_path(filepath)


add_extension  # just to avoid unused import warning


# --------------------------------------------------------------------------------------
# graph utils

Node = TypeVar("Node")
Nodes = List[Node]


def fuzzy_induced_graph(
    graph: dict, inducing_node_set: set, min_proportion: float = 1
) -> Iterable[Tuple[int, List[int]]]:
    """
    Keep only those (node, neighbors) pairs where both node and a minimum proportion of
    neighbors are in inducing_node_set.
    """
    for node, neighbors in graph.items():
        if node in inducing_node_set:
            neighbors_in_set = [n for n in neighbors if n in inducing_node_set]
            if len(neighbors_in_set) / len(neighbors) >= min_proportion:
                yield node, neighbors_in_set
