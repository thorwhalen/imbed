"""Access to ArXiv data.

The data was found at https://alex.macrocosm.so/download

Direct download 
"""

from typing import Mapping, Literal, Iterable, KT, Union
from functools import partial, lru_cache
import io
from itertools import chain
from dol import (
    FilesOfZip,
    remove_mac_junk_from_zip,
    add_ipython_key_completions,
    Pipe,
    cached_keys,
    wrap_kvs,
    KeyCodecs,
    FuncReader,
    KeyTemplate,
)
from dol.kv_codecs import common_prefix_keys_wrap
import pandas as pd
from imbed.util import MappingFactory, grazed_path

# from tabled import ColumnOrientedMapping

_raw_store = Pipe(
    grazed_path,
    FilesOfZip,
    remove_mac_junk_from_zip,
    common_prefix_keys_wrap,
    add_ipython_key_completions,
)

urls = {
    'titles': 'https://drive.google.com/file/d/1Ul5mPePtoPKHZkH5Rm6dWKAO11dG98GN/view?usp=share_link',
    'abstracts': 'https://drive.google.com/file/d/1g3K-wlixFxklTSUQNZKpEgN4WNTFTPIZ/view?usp=share_link',
}
raw_sources = FuncReader({name: partial(_raw_store, url) for name, url in urls.items()})

# --------------------------------------------------------------------------------------
# The data-specific part
# TODO: Ugly and unclean. Maybe just do with wrap_kvs and lambdas?

import pandas as pd

_key_template_kwargs = dict(
    field_patterns=dict(
        kind='\w+', number=r'\d+'
    ),  # the pattern to match is an integer
    from_str_funcs=dict(number=int),  # transform integer string into actual integer
)
_parquet_codec_end_pipe = (
    wrap_kvs(obj_of_data=Pipe(io.BytesIO, pd.read_parquet)),  # get a dataframe
    cached_keys(keys_cache=sorted),  # sort the keys when iterating
)

titles_key_template = KeyTemplate(
    'titles_{number:d}.parquet',  # this is the template/pattern for the keys
    **_key_template_kwargs,
)
titles_parquet_codec = Pipe(
    titles_key_template.filt_iter('str'),  # filter in only keys that match the pattern
    titles_key_template.key_codec('single'),  # just get a single integer as the key
    *_parquet_codec_end_pipe,
)
abstracts_key_template = KeyTemplate(
    'abstracts_{number:d}.parquet',  # this is the template/pattern for the keys
    **_key_template_kwargs,
)
abstracts_parquet_codec = Pipe(
    abstracts_key_template.filt_iter(
        'str'
    ),  # filter in only keys that match the pattern
    abstracts_key_template.key_codec('single'),  # just get a single integer as the key
    *_parquet_codec_end_pipe,
)


def _kind_router(k, v):
    if k.startswith('titles'):
        return titles_parquet_codec(v)
    elif k.startswith('abstracts'):
        return abstracts_parquet_codec(v)
    raise KeyError(f'Invalid key: {k}')


# --------------------------------------------------------------------------------------

sources = wrap_kvs(raw_sources, postget=_kind_router)

arxiv_url_template = "https://arxiv.org/{resource}/{doi}"

ArxivResource = Literal['abs', 'pdf', 'format', 'src', 'cits']

resource_descriptions = {
    "abs": "Main page of article. Contains links to all other relevant information.",
    "pdf": "Direct link to article pdf",
    "format": "Page giving access to other formats",
    "src": "Access to the original source files submitted by the authors.",
    "cits": "Tracks citations of the article across various platforms and databases.",
}


def arxiv_url(doi: str, resource: ArxivResource = 'abs') -> str:
    """Return the url for the given DOI and resource."""
    return arxiv_url_template.format(doi=doi, resource=resource)
