"""Utils for stores"""

from typing import Callable, Mapping, MutableMapping, Optional
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

spaces_dirname = "spaces"
spaces_template = pjoin(spaces_dirname, "{space}")
stores_template = pjoin("stores", "{store_kind}")
space_stores_template = pjoin(spaces_template, stores_template)


def mk_blob_store_for_path(
    path,
    space: str = None,
    *,
    store_kind="miscellenous_stuff",
    path_to_bytes_store: Callable = Files,
    base_store_wrap: Optional[Callable] = None,
    rm_mac_junk=True,
    filename_suffix: str = "",
    filename_prefix: str = "",
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
            k: v for k, v in _input_kwargs.items() if k not in {"path", "space"}
        }
        return partial(mk_blob_store_for_path, path, **_input_kwargs)
    assert space is not None, f"space must be provided"

    if base_store_wrap is None:
        store_wraps = []
    else:
        store_wraps = [base_store_wrap]
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
    store = store_wrap(path_to_bytes_store(space_store_root))
    return store


from functools import partial
from tabled import extension_based_wrap
import dill, json, pickle
import dol

general_decoder = {
    **extension_based_wrap.dflt_extension_to_decoder,
    "": dill.loads,
    "dill": dill.loads,
    "json": json.loads,
    "pkl": pickle.loads,
    "txt": bytes.decode,
}
general_encoder = {
    **extension_based_wrap.dflt_extension_to_encoder,
    "": dill.dumps,
    "dill": dill.dumps,
    "json": dol.Pipe(json.dumps, str.encode),
    "pkl": pickle.dumps,
    "txt": str.encode,
}

wrap_with_extension_codecs = partial(
    extension_based_wrap,
    extension_to_decoder=general_decoder,
    extension_to_encoder=general_encoder,
)


def extension_based_mall_maker(
    path_to_bytes_store=Files,
    extensions=("txt", "json", "pkl", "dill", ""),
    *,
    blob_store_maker=mk_blob_store_for_path,
    base_store_wrap=wrap_with_extension_codecs,
):
    store_maker_maker = partial(
        blob_store_maker,
        path_to_bytes_store=path_to_bytes_store,
        base_store_wrap=base_store_wrap,
    )
    ext_suffix = lambda ext: f".{ext}" if ext else ""
    return {
        ext: partial(store_maker_maker, filename_suffix=ext_suffix(ext))
        for ext in extensions
    }


# local_store_makers is a dict of store makers of bytes-based stores with various extensions
# Keys are file extensions, values are functions to create (local) stores with those extensions.
local_store_makers = extension_based_mall_maker(
    Files, extensions=("txt", "json", "pkl", "dill", "")
)

# A dict of store makers of bytes-based stores with various extensions
# Keys are file extensions, values are functions to create (local) stores with those extensions.


# For backcompatibility:
mk_text_local_store = local_store_makers["txt"]
mk_json_local_store = local_store_makers["json"]
mk_pickle_local_store = local_store_makers["pkl"]
mk_dill_local_store = local_store_makers["dill"]

# from tabled import DfFiles

# mk_table_local_store = partial(mk_blob_store_for_path, path_to_bytes_store=DfFiles)
mk_table_local_store = local_store_makers[""]
