"""Utils for imbed package."""

import os
import importlib.resources
from functools import partial, cached_property
from typing import Mapping, Callable, Optional, TypeVar, KT, Iterable, Any
from config2py import get_app_data_folder, process_path
from graze import (
    graze as _graze,
    Graze as _Graze,
    GrazeReturningFilepaths as _GrazeReturningFilepaths,
)

import re
import numpy as np

MappingFactory = Callable[..., Mapping]

package_name = 'imbed'
app_data_folder = os.environ.get(
    'IMBED_APP_DATA_FOLDER',
    get_app_data_folder(package_name, ensure_exists=True),
)

DFLT_DATA_DIR = process_path(app_data_folder, ensure_dir_exists=True)
GRAZE_DATA_DIR = process_path(DFLT_DATA_DIR, 'graze', ensure_dir_exists=True)
DFLT_SAVES_DIR = process_path(DFLT_DATA_DIR, 'saves', ensure_dir_exists=True)

saves_join = partial(os.path.join, DFLT_SAVES_DIR)

graze_kwargs = dict(
    rootdir=GRAZE_DATA_DIR,
    key_ingress=_graze.key_ingress_print_downloading_message_with_size,
)
graze = partial(_graze, **graze_kwargs)
grazed_path = partial(graze, return_filepaths=True)
Graze = partial(_Graze, **graze_kwargs)
GrazeReturningFilepaths = partial(_GrazeReturningFilepaths, **graze_kwargs)


non_alphanumeric_re = re.compile(r'\W+')


def lower_alphanumeric(text):
    return non_alphanumeric_re.sub(' ', text).strip().lower()


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
        return bytes_.decode('utf-8', 'replace')
    return bytes_


def print_with_timestamp(msg, *, refresh=None, display_time=True, print_func=print):
    """Prints with a timestamp and optional refresh.

    input: message, and possibly args (to be placed in the message string, sprintf-style

    output: Displays the time (HH:MM:SS), and the message

    use: To be able to track processes (and the time they take)

    >>> print_with_timestamp('hello')  # doctest: +SKIP
    (29)12:34:56 - hello

    """
    from datetime import datetime

    def hms_message(msg=''):
        t = datetime.now()
        return '({:02.0f}){:02.0f}:{:02.0f}:{:02.0f} - {}'.format(
            t.day, t.hour, t.minute, t.second, msg
        )

    if display_time:
        msg = hms_message(msg)
    if refresh:
        print_func(msg, end='\r')
    else:
        print_func(msg)


def clog(condition, *args, log_func=print_with_timestamp, **kwargs):
    """Conditional log

    >>> clog(False, "logging this")
    >>> clog(True, "logging this")  # doctest: +SKIP
    (29)12:34:56 - logging this
    >>> clog(True, "logging this", log_func=print)
    logging this

    """
    if not args and not kwargs:
        import functools

        return functools.partial(clog, condition, log_func=log_func)
    if condition:
        return log_func(*args, **kwargs)


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


def cosine_similarity(vec1, vec2):
    from scipy.spatial.distance import cosine

    return 1 - cosine(vec1, vec2)


# umap utils ---------------------------------------------------------------------------

from typing import Mapping, Dict, KT, Tuple, Sequence, Optional

EmbeddingsDict = Mapping[KT, Sequence]
PlanarEmbedding = Tuple[float, float]
PlanarEmbeddingsDict = Dict[KT, PlanarEmbedding]


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


def umap_2d_embeddings(kd_embeddings: EmbeddingsDict) -> PlanarEmbeddingsDict:
    """Takes a mapping of k-dimensional (kd) embeddings and returns a dict of the 2d
    umap embeddings

    :param kd_embeddings: a dict of kd embeddings
    :return: a dict of the 2d umap embeddings

    """
    import umap  # pip install umap-learn

    kd_embeddings = ensure_embedding_dict(kd_embeddings)

    umap_embeddings = umap.UMAP(n_components=2).fit_transform(
        list(kd_embeddings.values())
    )
    return {k: tuple(v) for k, v in zip(kd_embeddings.keys(), umap_embeddings)}


import pandas as pd


def planar_embeddings_dict_to_df(
    planar_embeddings: PlanarEmbeddingsDict,
    *,
    x_col: str = 'x',
    y_col: str = 'y',
    key_col: Optional[str] = None,
) -> pd.DataFrame:
    """A function that takes a dict of planar embeddings and returns a pandas DataFrame
    of the 2d embeddings

    If key_col is not None, the keys are added as a column in the dataframe.

    :param planar_embeddings: a dict of planar embeddings
    :param x_col: the name of the x column
    :param y_col: the name of the y column
    :param key_col: the name of the key column
    :return: a pandas DataFrame of the 2d embeddings

    Example:

    >>> planar_embeddings = {1: (0.1, 0.2), 2: (0.3, 0.4)}
    >>> planar_embeddings_dict_to_df(planar_embeddings)  # doctest: +NORMALIZE_WHITESPACE
         x    y
    1  0.1  0.2
    2  0.3  0.4


    """
    df = pd.DataFrame(planar_embeddings).T.rename(columns={0: x_col, 1: y_col})
    if key_col is not None:
        # return a dataframe with an extra key column containing the keys
        df[key_col] = df.index
        df.reset_index(drop=True, inplace=True)
        df = df[[key_col, x_col, y_col]]
    return df


two_d_embedding_dict_to_df = planar_embeddings_dict_to_df  # back-compatibility alias


def umap_2d_embeddings_df(
    kd_embeddings: Mapping[KT, Sequence],
    *,
    x_col: str = 'x',
    y_col: str = 'y',
    key_col: Optional[str] = None,
) -> pd.DataFrame:
    """A function that takes a mapping of kd embeddings and returns a pandas DataFrame
    of the 2d umap embeddings"""
    return two_d_embedding_dict_to_df(
        umap_2d_embeddings(kd_embeddings),
        x_col=x_col,
        y_col=y_col,
        key_col=key_col,
    )


# --------------------------------------------------------------------------------------
# misc
from functools import partial
import os
import io
from typing import List, Tuple, Dict, Any, Callable, Union, Optional
from collections import Counter
import pandas as pd
import json
import pickle

import numpy as np
from i2 import name_of_obj
from dol import Pipe, wrap_kvs, written_bytes
from dol.zipfiledol import file_or_folder_to_zip_file


def if_extension_not_present_add_it(filepath, extension):
    if not filepath.endswith(extension):
        return filepath + extension
    return filepath


def if_extension_present_remove_it(filepath, extension):
    if filepath.endswith(extension):
        return filepath[: -len(extension)]
    return filepath


def save_df_to_zipped_tsv(df: pd.DataFrame, name: str, sep='\t', index=False, **kwargs):
    """Save a dataframe to a zipped tsv file."""
    name = if_extension_present_remove_it(name, '.zip')
    name = if_extension_present_remove_it(name, '.tsv')
    tsv_filepath = f'{name}.tsv'
    zip_filepath = f'{tsv_filepath}.zip'
    df.to_csv(tsv_filepath, sep=sep, index=index, **kwargs)

    file_or_folder_to_zip_file(tsv_filepath, zip_filepath)


extension_to_decoder = {
    '.txt': lambda obj: obj.decode('utf-8'),
    '.json': json.loads,
    '.pkl': pickle.loads,
    '.parquet': Pipe(io.BytesIO, pd.read_parquet),
    '.npy': Pipe(io.BytesIO, np.load),
    '.csv': Pipe(io.BytesIO, pd.read_csv),
    '.xlsx': Pipe(io.BytesIO, pd.read_excel),
    '.tsv': Pipe(io.BytesIO, partial(pd.read_csv, sep='\t')),
}

extension_to_encoder = {
    '.txt': lambda obj: obj.encode('utf-8'),
    '.json': json.dumps,
    '.pkl': pickle.dumps,
    '.parquet': written_bytes(pd.DataFrame.to_parquet),
    '.npy': written_bytes(np.save),
    '.csv': written_bytes(pd.DataFrame.to_csv),
    '.xlsx': written_bytes(pd.DataFrame.to_excel),
    '.tsv': written_bytes(partial(pd.DataFrame.to_csv, sep='\t')),
}


def extension_based_decoding(k, v):
    ext = '.' + k.split('.')[-1]
    decoder = extension_to_decoder.get(ext, None)
    if decoder is None:
        raise ValueError(f"Unknown extension: {ext}")
    return decoder(v)


def extension_based_encoding(k, v):
    ext = '.' + k.split('.')[-1]
    encoder = extension_to_encoder.get(ext, None)
    if encoder is None:
        raise ValueError(f"Unknown extension: {ext}")
    return encoder(v)


def extension_base_wrap(store):
    return wrap_kvs(
        store,
        postget=extension_based_decoding,
        preset=extension_based_encoding,
    )


def load_if_saved(
    key=None,
    store_attr='saves',
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
