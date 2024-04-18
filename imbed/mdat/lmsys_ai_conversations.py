"""
Lymsys AI Conversations

See paper: https://arxiv.org/pdf/2309.11998.pdf
Data here: https://huggingface.co/papers/2309.11998

"""

from functools import cached_property, partial
import os
import io
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Callable
from collections import Counter
from itertools import chain
import pandas as pd
from tabled import expand_rows, expand_columns
import json
from datasets import load_dataset
import pickle

from i2 import name_of_obj
from dol import Files, Pipe, wrap_kvs, written_bytes, filt_iter
from imbed.util import DFLT_SAVES_DIR

extension_to_decoder = {
    '.txt': lambda obj: obj.decode('utf-8'),
    '.json': json.loads,
    '.parquet': Pipe(io.BytesIO, pd.read_parquet),
    '.pkl': pickle.loads,
}

parquet_bytes = written_bytes(pd.DataFrame.to_parquet)

extension_to_encoder = {
    '.txt': lambda obj: obj.encode('utf-8'),
    '.json': json.dumps,
    '.parquet': parquet_bytes,
    '.pkl': pickle.dumps,
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


def counts(sr: pd.Series) -> pd.Series:
    return pd.Series(dict(Counter(sr).most_common()))


@dataclass
class LmsysDacc:
    name = 'lmsys-chat-1m'

    dataset_dict_loader: Callable = partial(load_dataset, 'lmsys/lmsys-chat-1m')
    saves_dir: str = os.path.join(DFLT_SAVES_DIR, name)

    @cached_property
    def saves_bytes_store(self):
        if not os.path.isdir(self.saves_dir):
            os.mkdir(self.saves_dir)
        return Files(self.saves_dir)

    @cached_property
    def saves(self):
        return extension_base_wrap(self.saves_bytes_store)

    @cached_property
    def dataset_dict(self):
        return self.dataset_dict_loader()

    @property
    def _train_data(self):
        return self.dataset_dict['train']

    @cached_property
    def train_data(self):
        return self._train_data.to_pandas()

    @cached_property
    def conversation_sizes(self):
        _conversation_sizes = self.train_data.conversation.apply(len)
        assert all(
            _conversation_sizes == self.train_data.turn * 2
        ), "Some turns were not twice the conversation size"
        return _conversation_sizes

    @cached_property
    def language_count(self):
        return counts(self.train_data['language'])

    @cached_property
    def model_count(self):
        return counts(self.train_data['model'])

    @cached_property
    def redacted_count(self):
        return counts(self.train_data['redacted'])

    @cached_property
    def role_count(self):
        c = Counter(
            x['role'] for x in chain.from_iterable(self.train_data.conversation)
        )
        return pd.Series(dict(c.most_common()))

    @cached_property
    def en_df(self):
        return self.train_data[self.train_data['language'] == 'English']

    @cached_property
    def flat_en(self):
        t = self.en_df
        t = expand_rows(t, ['conversation', 'openai_moderation'])
        t = expand_columns(t, ['conversation', 'openai_moderation'], key_mapper=None)
        t = expand_columns(t, ['categories'], key_mapper=None)
        t = expand_columns(t, ['category_scores'])
        return t

    @cached_property
    def flat_en_embeddable(self):
        from oa.util import embeddings_models
        from oa import text_is_valid
        import numpy as np

        model = 'text-embedding-3-small'
        max_tokens = embeddings_models[model]['max_input']

        # TODO: Make it source from self.flat_en directly (once persistent caching is working)
        # TODO: Also make it check if dacc.flat_en is loaded or not before, and if not, unload it after using.
        flat_en = self.saves['flat_en.parquet']

        lidx = ~np.array(
            list(
                text_is_valid(
                    flat_en.content,
                    flat_en.num_of_tokens,
                    max_tokens=max_tokens,
                    model=model,
                )
            )
        )
        invalid_conversations = set(flat_en[lidx]['conversation_id'])

        print(f"{len(invalid_conversations)=}")

        df = flat_en[~flat_en.conversation_id.isin(invalid_conversations)]
        return df

    @property
    def flat_en_embeddings_store(self):
        from dol import KeyTemplate, cache_iter

        key_template = KeyTemplate(
            'flat_en_embeddings/{index}.parquet',
            from_str_funcs={'index': int},
            to_str_funcs={'index': "{:04d}".format},
        )
        s = key_template.filt_iter(self.saves)
        s = key_template.key_codec(decoded='single')(s)
        s = cache_iter(s, keys_cache=sorted)
        return s

    @cached_property
    def flat_en_embeddings(self):
        return pd.concat(self.flat_en_embeddings_store.values())


def mk_dacc():
    return LmsysDacc()


# def dataframe_to_embed(dacc=None):
#     from oa.util import embeddings_models
#     from oa import text_is_valid
#     import numpy as np

#     dacc = dacc or mk_dacc()

#     model = 'text-embedding-3-small'
#     max_tokens = embeddings_models[model]['max_input']

#     flat_en = dacc.saves['flat_en.parquet']

#     lidx = ~np.array(
#         list(
#             text_is_valid(
#                 flat_en.content,
#                 flat_en.num_of_tokens,
#                 max_tokens=max_tokens,
#                 model=model,
#             )
#         )
#     )
#     invalid_conversations = set(flat_en[lidx]['conversation_id'])

#     print(f"{len(invalid_conversations)=}")

#     df = flat_en[~flat_en.conversation_id.isin(invalid_conversations)]
#     return df


def compute_and_save_embeddings(
    dacc=None,
    *,
    chk_size=1000,  # needs to be under max batch size of 2048
    validate=False,
    overwrite_chunks=False,
    model='text-embedding-3-small',
):
    dacc = dacc or mk_dacc()
    df = dacc.flat_en_embeddable

    from oa import embeddings as embeddings_
    from imbed import fixed_step_chunker
    import pandas as pd
    from functools import partial
    import os

    embeddings = partial(embeddings_, validate=validate, model=model)

    def key_for_chunk_index(i):
        return f"flat_en_embeddings/{i:04d}.parquet"

    def store_chunk(i, chunk):
        save_path = os.path.join(dacc.saves.rootdir, key_for_chunk_index(i))
        chunk.to_parquet(save_path)

    for i, index_and_row in enumerate(
        fixed_step_chunker(
            df[['conversation_id', 'content', 'num_of_tokens']].iterrows(),
            chk_size,
            return_tail=True,
        )
    ):
        if not overwrite_chunks and key_for_chunk_index(i) in dacc.saves:
            print(f"Skipping {i=} because it is already saved.")
            continue
        # else...
        if i % 100 == 0:
            print(f"Processing {i=}")
        try:
            chunk = pd.DataFrame(
                [x[1] for x in index_and_row], index=[x[0] for x in index_and_row]
            )
            vectors = embeddings(chunk.content.tolist())
            chunk['embeddings'] = vectors
            store_chunk(i, chunk)
        except Exception as e:
            print(f"--> ERROR: {i=}, {e=}")


if __name__ == '__main__':
    print("Making the dacc...")
    dacc = mk_dacc()
    print("Computing embeddings...")
    compute_and_save_embeddings(dacc)
