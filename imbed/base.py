"""Base functionality of imbded."""

from functools import partial

# TODO: Take the default from oa
DFLT_EMBEDDING_MODEL = 'text-embedding-3-small'


# ---------------------------------------------------------------------------------
# Typing
from typing import (
    Callable,
    Protocol,
    Iterable,
    Sequence,
    Union,
    KT,
    Mapping,
    Any,
    Optional,
    NewType,
    Tuple,
)

# Domain specific type aliases
# We use the convention that if THING is a type, then THINGs is an iterable of THING,
# and THINGMapping is a mapping from a key to a THING, and THINGSpec is a Union of
# objects that can specify THING explicitly or implicitly (for example, arguments to
# make a THING or the key to retrieve a THING).

Metadata = Any

# Text (also known as a document in some contexts)
Text = NewType('Text', str)
TextKey = NewType('TextKey', KT)
TextSpec = Union[str, TextKey]  # the text itself, or a key to retrieve it
Texts = Iterable[Text]
TextMapping = Mapping[TextKey, Text]

# The metadata of a text
TextMetadata = Metadata
MetadataMapping = Mapping[TextKey, TextMetadata]

# Text is usually segmented before vectorization.
# A Segment could be the whole text, or a part of the text (e.g. sentence, paragraph...)
Segment = NewType('Segment', str)
SegmentKey = NewType('SegmentKey', KT)
Segments = Iterable[Segment]
SingularTextSegmenter = Callable[[Text], Segments]
SegmentMapping = Mapping[SegmentKey, Segment]

# NLP models often require a vector representation of the text segments.
# A vector is a sequence of floats.
# These vectors are also called embeddings.
Vector = Sequence[float]
Vectors = Iterable[Vector]
VectorMapping = Mapping[SegmentKey, Vector]
SingularSegmentVectorizer = Callable[[Segment], Vector]
BatchSegmentVectorizer = Callable[[Segments], Vectors]
SegmentVectorizer = Union[SingularSegmentVectorizer, BatchSegmentVectorizer]

# To visualize the vectors, we often project them to a 2d plane.
PlanarVector = Tuple[float, float]
PlanarVectors = Iterable[PlanarVector]
PlanarVectorMapping = Mapping[SegmentKey, PlanarVector]
SingularPlanarProjector = Callable[[Vector], PlanarVector]
BatchPlanarProjector = Callable[[Vectors], PlanarVectors]
PlanarProjector = Union[SingularPlanarProjector, BatchPlanarProjector]


class Embed(Protocol):
    """A callable that embeds text."""

    def __call__(self, text: Union[Text, Texts]) -> Union[Vector, Vectors]:
        """Embed a single text, or an iterable of texts.
        Note that this embedding could be calculated, or retrieved from a store,
        """


# ---------------------------------------------------------------------------------
# Base data access class for imbeddings data flows (e.g. pipelines)

from dataclasses import dataclass
from dol import KvReader


def identity(x):
    return x


@dataclass
class ComputedValuesMapping(KvReader, Mapping):
    """
    A mapping that returns empty values for all keys.

    Example usage:

    >>> m = ComputedValuesMapping(('apple', 'crumble'), value_of_key=len)
    >>> list(m)
    ['apple', 'crumble']
    >>> m['apple']
    5

    """

    keys_factory: Optional[Callable[[], Iterable[KT]]]
    value_of_key: Callable[[KT], Any] = partial(identity, None)

    def __post_init__(self):
        if not callable(self.keys_factory):
            self.keys_factory = partial(identity, self.keys_factory)

    def __iter__(self):
        return iter(self.keys_factory())

    def __getitem__(self, k):
        return self.value_of_key(k)


class ImbedDaccBase:
    text_to_segments: Callable[[Text], Sequence[Segment]] = identity

    def download_source_data(self, uri: str):
        """Initial download of data from the source"""

    @property
    def texts(self) -> TextMapping:
        """key-value view (i.e. Mapping) of the text data"""

    @property
    def text_metadatas(self) -> MetadataMapping:
        """Mapping of the metadata of the text data.

        The keys of texts and text_metadatas mappings should be the same
        """

    @property
    def text_segments(self) -> SegmentMapping:
        """Mapping of the segments of text data.

        Could be computed on the fly from the text_store and a segmentation algorithm,
        or precomputed and stored in a separate key-value store.

        Preferably, the key of the text store should be able to be computed from key
        of the text_segments store, and even contain the information necessary to
        extract the segment from the corresponding text store value.

        Note that the imbed.segmentation.SegmentMapping class can be used to
        create a mapping between the text store and the text segments store.
        """
        # default is segments are the same as the text
        return self.texts

    @property
    def segment_vectors(self) -> VectorMapping:
        """Mapping of the vectors (embeddings) of the segments of text data.

        The keys of the segment_vectors store should be the same as the keys of the
        text_segments store.

        Could be computed on the fly from the text_segments and a vectorization algorithm,
        or precomputed and stored in a separate key-value store.

        Preferably, the key of the text_segments store should be able to be computed from key
        of the segment_vectors store, and even contain the information necessary to
        extract the segment from the corresponding text segments store value.

        Note that the imbed.vectorization.VectorMapping class can be used to
        create a mapping between the text segments store and the segment_vectors store.
        """

    @property
    def planar_embeddings(self) -> VectorMapping:
        """Mapping of the 2d embeddings of the segments of text data.

        The keys of the planar_embeddings store should be the same as the keys of the
        segment_vectors store.

        Could be computed on the fly from the segment_vectors and a dimensionality reduction algorithm,
        or precomputed and stored in a separate key-value store.

        Preferably, the key of the segment_vectors store should be able to be computed from key
        of the planar_embeddings store, and even contain the information necessary to
        extract the segment from the corresponding segment_vectors store value.

        Note that the imbed.vectorization.VectorMapping class can be used to
        create a mapping between the segment_vectors store and the planar_embeddings store.
        """
        # default is to compute


# ---------------------------------------------------------------------------------
# Base functionality


from functools import cached_property, partial
import os
from dataclasses import dataclass, field, KW_ONLY
from typing import List, Tuple, Dict, Any, Callable, Union, Optional, MutableMapping

import pandas as pd

from dol import Files, mk_dirs_if_missing, add_ipython_key_completions
from imbed.util import extension_base_wrap, DFLT_SAVES_DIR, clog

saves_join = partial(os.path.join, DFLT_SAVES_DIR)


DataSpec = Union[str, Any]


def _ensure_dir_exists(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


def mk_local_store(rootdir: str):
    return extension_base_wrap(
        add_ipython_key_completions(mk_dirs_if_missing(Files(rootdir)))
    )


class LocalSavesMixin:
    # @staticmethod
    # def init_data_loader(init_data_key):
    #     return huggingface_load_dataset(init_data_key)

    @cached_property
    def saves_bytes_store(self):
        return Files(self.saves_dir)

    @cached_property
    def saves(self):
        rootdir = _ensure_dir_exists(self.saves_dir)
        return mk_local_store(rootdir)

    @cached_property
    def embeddings_chunks_store(self):
        rootdir = _ensure_dir_exists(os.path.join(self.saves_dir, 'embeddings_chunks'))
        return mk_local_store(rootdir)


@dataclass
class HugfaceDaccBase(LocalSavesMixin):
    huggingface_data_stub: str
    name: Optional[str] = None
    _: KW_ONLY
    saves_dir: Optional[str] = None
    root_saves_dir: str = DFLT_SAVES_DIR
    verbose: int = 1

    # just for information (haven't found where to ask datasets package this info)
    _huggingface_dowloads_dir = os.environ.get(
        "HF_DATASETS_CACHE", os.path.expanduser('~/.cache/huggingface/datasets')
    )

    def __post_init__(self):
        assert isinstance(
            self.huggingface_data_stub, str
        ), f"{self.huggingface_data_stub=} is not a string"
        assert (
            len(self.huggingface_data_stub.split('/')) == 2
        ), f"{self.huggingface_data_stub=} should have exactly one '/'"
        if self.name is None:
            self.name = self.huggingface_data_stub.split('/')[-1]

        # TODO: Below is reusable. Factor out:
        if self.saves_dir is None:
            self.saves_dir = self._saves_join(self.name)
        self.dataset_dict_loader = partial(
            self.init_data_loader, self.huggingface_data_stub
        )

    def _saves_join(self, *args):
        return os.path.join(self.root_saves_dir, *args)

    @staticmethod
    def init_data_loader(init_data_key):
        from datasets import load_dataset as huggingface_load_dataset

        return huggingface_load_dataset(init_data_key)

    def get_data(self, data_spec: DataSpec, *, assert_type=None):
        if isinstance(data_spec, str):
            # if data_spec is a string, check if it's an attribute or a key of saves
            if hasattr(self, data_spec):
                return getattr(self, data_spec)
            elif data_spec in self.saves:
                return self.saves[data_spec]
        if assert_type:
            assert isinstance(
                data_spec, assert_type
            ), f"{data_spec=} is not {assert_type}"
        # just return the data_spec itself as the data
        return data_spec

    @cached_property
    def dataset_dict(self):
        return self.dataset_dict_loader()

    @property
    def _train_data(self):
        return self.dataset_dict['train']

    @cached_property
    def train_data(self):
        return self._train_data.to_pandas()

    @property
    def _test_data(self):
        return self.dataset_dict['test']

    @cached_property
    def test_data(self):
        return self._test_data.to_pandas()

    @cached_property
    def all_data(self):
        return pd.concat([self.train_data, self.test_data], axis=0, ignore_index=True)


import oa
from oa.base import DFLT_EMBEDDINGS_MODEL

def add_token_info_to_df(
    df,
    segments_col: str,
    *,
    token_count_col='token_count',
    segment_is_valid_col='segment_is_valid',
    model=DFLT_EMBEDDINGS_MODEL
):
    num_tokens = partial(oa.num_tokens, model=model)
    max_input = oa.util.embeddings_models[model]['max_input']

    if token_count_col and token_count_col not in df.columns:
        df[token_count_col] = df[segments_col].apply(num_tokens)
    if segment_is_valid_col and segment_is_valid_col not in df.columns:
        df[segment_is_valid_col] = df[token_count_col] <= max_input

    return df


from imbed.segmentation import fixed_step_chunker
from imbed.util import clog

DFLT_CHK_SIZE = 1000


def batches(df, chk_size=DFLT_CHK_SIZE):
    """
    Yield batches of rows from a DataFrame.

    The yielded batches are lists of (index, row) tuples.

    If chk_size is None, yield the whole DataFrame as a single batch.

    """
    if chk_size is None:
        yield list(df.iterrows())
    else:
        for index_and_row in fixed_step_chunker(
            df.iterrows(),
            chk_size,
            return_tail=True,
        ):
            yield index_and_row


def get_empty_temporary_folder():
    """Returns the path of a new, empty temporary folder."""
    import tempfile

    return tempfile.mkdtemp()


def compute_and_save_embeddings(
    df: pd.DataFrame,
    save_store: Optional[Union[MutableMapping[int, Any], str]] = None,
    *,
    text_col='content',
    embeddings_col='embeddings',
    chk_size=DFLT_CHK_SIZE,  # needs to be under max batch size of 2048
    validate=False,
    overwrite_chunks=False,
    model=DFLT_EMBEDDING_MODEL,
    verbose=1,
    exclude_chk_ids=(),
    include_chk_ids=(),
    progress_log_every: int = 100,
    key_for_chunk_index: Union[Callable[[int], Any], str] = 'embeddings_{:04d}.parquet',
):
    _clog = partial(clog, verbose)
    __clog = partial(clog, verbose >= 2)

    from oa import embeddings as embeddings_

    embeddings = partial(embeddings_, validate=validate, model=model)

    if save_store is None:
        save_store = get_empty_temporary_folder()
        _clog(f"Using a temporary folder for save_store: {save_store}")
    if isinstance(save_store, str) and os.path.isdir(save_store):
        save_store = extension_base_wrap(mk_dirs_if_missing(Files(save_store)))
    assert isinstance(save_store, MutableMapping)

    embeddings = partial(embeddings_, validate=validate, model=model)

    if isinstance(key_for_chunk_index, str):
        key_for_chunk_index_template = key_for_chunk_index
        key_for_chunk_index = key_for_chunk_index_template.format
    elif key_for_chunk_index is None:
        key_for_chunk_index = lambda i: i
    assert callable(key_for_chunk_index)

    def store_chunk(i, chunk):
        key = key_for_chunk_index(i)
        save_store[key] = chunk
        # save_path = os.path.join(save_store.rootdir, key_for_chunk_index(i))
        # chunk.to_parquet(save_path)

    for i, index_and_row in enumerate(batches(df, chk_size)):
        if i in exclude_chk_ids or (include_chk_ids and i not in include_chk_ids):
            # skip this chunk if it is in the exclude list or if the
            # include list is not empty and this chunk is not in it
            __clog(
                f"Skipping {i=} because it is in the exclude list or not in the include list."
            )
            continue
        if not overwrite_chunks and key_for_chunk_index(i) in save_store:
            _clog(f"Skipping {i=} because it is already saved.")
            continue
        # else...
        if i % progress_log_every == 0:
            _clog(f"Processing {i=}")
        try:
            chunk = pd.DataFrame(
                [x[1] for x in index_and_row], index=[x[0] for x in index_and_row]
            )
            vectors = embeddings(chunk[text_col].tolist())
            chunk[embeddings_col] = vectors
            store_chunk(i, chunk)
        except Exception as e:
            _clog(f"--> ERROR: {i=}, {e=}")


def compute_and_save_planar_embeddings(
    embeddings_store,
    save_store=None,
    *,
    verbose=0,
    save_key='planar_embeddings.parquet',
):
    from imbed import umap_2d_embeddings

    # dacc = dacc or mk_dacc()
    _clog = partial(clog, verbose)
    __clog = partial(clog, verbose >= 2)

    # _clog("Getting flat_en_embeddings")
    # dacc.flat_en_embeddings

    # _clog(f"{len(dacc.flat_en_embeddings.shape)=}")
    __clog("Making an embeddings store from it, using flat_end_embeddings keys as keys")
    # embdeddings_store = {
    #     id_: row.embeddings for id_, row in dacc.flat_en_embeddings.iterrows()
    # }

    __clog("Computing the 2d embeddings (the long process)...")
    planar_embeddings = umap_2d_embeddings(embeddings_store)

    __clog(f"Reformatting the {len(planar_embeddings)} embeddings into a DataFrame")
    planar_embeddings = pd.DataFrame(planar_embeddings, index=['x', 'y']).T

    __clog("Saving the planar embeddings to planar_embeddings.parquet'")
    if save_store is not None:
        try:
            save_store[save_key] = planar_embeddings
        except Exception as e:
            _clog(f"Error saving planar embeddings: {e}")

    return planar_embeddings
