"""Utils for stores"""

from typing import Callable, Mapping, MutableMapping
import os
from pathlib import Path
import json

from dol import (
    filt_iter,
    Files,
    KeyTemplate,
    Pipe,
    KeyCodecs,
    add_ipython_key_completions,
    mk_dirs_if_missing,
)
from dol import DirReader, wrap_kvs
from dol.filesys import with_relative_paths
from dol.util import not_a_mac_junk_path

Store = MutableMapping[str, Callable]
Mall = Mapping[str, Store]

pjoin = os.path.join

spaces_dirname = 'spaces'
spaces_template = pjoin(spaces_dirname, '{space}')
stores_template = pjoin('stores', '{store_kind}')
space_stores_template = pjoin(spaces_template, stores_template)


def mk_blob_store_for_path(
    path,
    space: str = None,
    *,
    store_kind='miscellenous_stuff',
    path_to_store: Callable = Files,
    rm_mac_junk=True,
    filename_suffix: str = '',
    filename_prefix: str = '',
    auto_make_dirs=True,
    key_autocomplete=True,
):
    _input_kwargs = locals()

    # TODO: Add for local stores only.
    # if not os.path.isdir(path):
    #     raise ValueError(f"path {path} is not a directory")
    if space is None:
        # bind the path, resulting in a function parametrized by space
        _input_kwargs = {
            k: v for k, v in _input_kwargs.items() if k not in {'path', 'space'}
        }
        return partial(mk_blob_store_for_path, path, **_input_kwargs)
    assert space is not None, f"space must be provided"

    store_wraps = []
    if filename_suffix or filename_prefix:
        store_wraps.append(
            KeyCodecs.affixed(prefix=filename_prefix, suffix=filename_suffix)
        )
    if rm_mac_junk:
        store_wraps.append(filt_iter(filt=not_a_mac_junk_path))
    if auto_make_dirs:
        store_wraps.append(mk_dirs_if_missing)
        # if not os.path.isdir(path):
        #     os.makedirs(path, exist_ok=True)
    if key_autocomplete:
        store_wraps.append(add_ipython_key_completions)

    store_wrap = Pipe(*store_wraps)

    space_store_root = pjoin(
        path,
        space_stores_template.format(space=space, store_kind=store_kind),
    )
    store = store_wrap(path_to_store(space_store_root))
    return store


from functools import partial
import dill
from dol import TextFiles, JsonFiles, PickleFiles, wrap_kvs

mk_text_local_store = partial(
    mk_blob_store_for_path, path_to_store=TextFiles, filename_suffix='.txt'
)
mk_json_local_store = partial(
    mk_blob_store_for_path,
    path_to_store=JsonFiles,  # filename_suffix='.json'
)
mk_pickle_local_store = partial(
    mk_blob_store_for_path, path_to_store=PickleFiles, filename_suffix='.pkl'
)

# pickle is builtin, but fickle -- dill can serialize more things (lambdas, etc.)
LocalDillStore = wrap_kvs(Files, data_of_obj=dill.dumps, obj_of_data=dill.loads)
mk_dill_local_store = partial(
    mk_blob_store_for_path,
    path_to_store=LocalDillStore,
    filename_suffix='.dill',
)

# For tables, a DfFiles stores will be able to read/write in many formats
from tabled import DfFiles

mk_table_local_store = partial(mk_blob_store_for_path, path_to_store=DfFiles)
