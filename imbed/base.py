"""Base functionality of imbded."""

DFLT_EMBEDDING_MODEL = 'text-embedding-3-small'


# ---------------------------------------------------------------------------------
# Typing
from typing import Callable, Protocol, Iterable, Sequence, Union, KT


Text = Union[str, KT]  # the text itself, or a key to retrieve it
Texts = Iterable[Text]
Vector = Sequence[float]
Vectors = Iterable[Vector]


class Embed(Protocol):
    """A callable that embeds text."""

    def __call__(self, text: Union[Text, Texts]) -> Union[Vector, Vectors]:
        """Embed a single text, or an iterable of texts.
        Note that this embedding could be calculated, or retrieved from a store,
        """


# ---------------------------------------------------------------------------------
# Base functionality


from functools import cached_property, partial
import os
from dataclasses import dataclass, field, KW_ONLY
from typing import List, Tuple, Dict, Any, Callable, Union, Optional, MutableMapping

import pandas as pd
from datasets import load_dataset as huggingface_load_dataset

from dol import Files, mk_dirs_if_missing, add_ipython_key_completions
from imbed.util import extension_base_wrap, DFLT_SAVES_DIR


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
    @staticmethod
    def init_data_loader(init_data_key):
        return huggingface_load_dataset(init_data_key)

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
