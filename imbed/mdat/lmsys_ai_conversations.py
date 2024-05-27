"""
Lymsys AI Conversations

See paper: https://arxiv.org/pdf/2309.11998.pdf
Data here: https://huggingface.co/papers/2309.11998

"""

from functools import cached_property, partial
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Callable, Union
from collections import Counter
from itertools import chain

import pandas as pd
from datasets import load_dataset

from dol import Files
from tabled import expand_rows, expand_columns
from imbed.util import saves_join, merge_data, extension_base_wrap, counts

data_name = 'lmsys-chat-1m'
huggingface_data_stub = 'lmsys/lmsys-chat-1m'
data_saves_join = partial(saves_join, data_name)


DataSpec = Union[str, Any]


@dataclass
class Dacc:
    name = data_name

    dataset_dict_loader: Callable = partial(load_dataset, huggingface_data_stub)
    saves_dir: str = data_saves_join()

    @cached_property
    def saves_bytes_store(self):
        if not os.path.isdir(self.saves_dir):
            os.mkdir(self.saves_dir)
        return Files(self.saves_dir)

    @cached_property
    def saves(self):
        return extension_base_wrap(self.saves_bytes_store)

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

    @cached_property
    def flat_en_conversation_grouped_embeddings(self):
        # TODO: This is the kind of thing that the save decorator should take care of
        if 'flat_en_conversation_grouped_embeddings.parquet' in self.saves:
            return self.saves['flat_en_conversation_grouped_embeddings.parquet']
        else:
            g = self.flat_en_embeddings
            g = g.drop(columns=['content'])
            # groupby conversation_id, doing the following with, for each group:
            g = g.groupby('conversation_id')
            g = g.agg({'num_of_tokens': 'sum', 'embeddings': 'mean'})
            self.saves['flat_en_conversation_grouped_embeddings.parquet'] = g
            return g

    def planar_embeddings_with_metadata(
        self,
        planar_embeddings: DataSpec,
        metadata: DataSpec = 'flat_en_embeddable',
        *,
        merge_on=None,
    ):
        planar_embeddings = self.get_data(planar_embeddings)
        metadata = self.get_data(metadata)

        return merge_data(
            metadata, planar_embeddings, merge_on=merge_on, data_2_cols=['x', 'y']
        )

    @property
    def planar_embeddings_of_grouped_conversations(self):
        t = self.saves['planar_embeddings_grouped.pkl']
        t = pd.Series(t)
        t = pd.DataFrame(t).reset_index()
        t = t.rename(columns={'index': 'conversation_id'})
        t['x'] = t[0].apply(lambda x: x[0])
        t['y'] = t[0].apply(lambda x: x[1])
        t = t.drop(0, axis=1)
        t.index.name = 'id'
        # self.saves['planar_embeddings_grouped.tsv'] = t
        return t

    @property
    def meta_for_grouped_conversations(self):
        meta = self.saves['flat_en.parquet']
        nested_columns = list(filter(lambda x: '.' in x, meta.columns))
        meta = meta.drop(
            ['language', 'redacted', 'content', 'role'] + nested_columns, axis=1
        )
        aggregate_specs_1 = {
            'model': lambda x: x[0],
            'turn': lambda x: x[0],
            'flagged': 'mean',
            'num_of_tokens': 'sum',
        }
        remaining_columns = (
            set(meta.columns) - set(aggregate_specs_1.keys()) - {'conversation_id'}
        )
        aggregate_specs = {
            'model': 'first',
            'turn': 'first',
            'flagged': 'mean',
            'num_of_tokens': 'sum',
            **{col: 'mean' for col in remaining_columns},
        }

        # group by conversaion_id and apply aggaggregate_specs to each group
        meta = meta.groupby('conversation_id').agg(aggregate_specs)
        meta = meta.reset_index(drop=False)
        return meta

    @property
    def planar_embeddings_of_grouped_conversations_with_metadata(self):
        e = self.planar_embeddings_of_grouped_conversations
        meta = self.meta_for_grouped_conversations
        df = merge_data(e, meta, merge_on='conversation_id')
        df['id'] = df.conversation_id  # just because cosmograph needs it!
        return df
        # dacc.saves['planar_embeddings_of_grouped_conversations_with_metadata.tsv'] = df


LmsysDacc = Dacc  # backwards compatibility alias


def mk_dacc():
    return Dacc()


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

from imbed.segmentation import fixed_step_chunker
from imbed.util import clog
from imbed.base import batches, DFLT_CHK_SIZE


def compute_and_save_embeddings(
    dacc=None,
    *,
    chk_size=DFLT_CHK_SIZE,  # needs to be under max batch size of 2048
    validate=False,
    overwrite_chunks=False,
    model='text-embedding-3-small',
    verbose=1,
    exclude_chk_ids=(),
    include_chk_ids=(),
):
    _clog = partial(clog, verbose)
    __clog = partial(clog, verbose >= 2)

    dacc = dacc or mk_dacc()
    df = dacc.flat_en_embeddable[['conversation_id', 'content', 'num_of_tokens']]

    from oa import embeddings as embeddings_
    import pandas as pd
    from functools import partial
    import os

    embeddings = partial(embeddings_, validate=validate, model=model)

    def key_for_chunk_index(i):
        return f"flat_en_embeddings/{i:04d}.parquet"

    def store_chunk(i, chunk):
        save_path = os.path.join(dacc.saves.rootdir, key_for_chunk_index(i))
        chunk.to_parquet(save_path)

    for i, index_and_row in enumerate(batches(df, chk_size)):
        if i in exclude_chk_ids or (include_chk_ids and i not in include_chk_ids):
            # skip this chunk if it is in the exclude list or if the
            # include list is not empty and this chunk is not in it
            __clog(
                f"Skipping {i=} because it is in the exclude list or not in the include list."
            )
            continue
        if not overwrite_chunks and key_for_chunk_index(i) in dacc.saves:
            _clog(f"Skipping {i=} because it is already saved.")
            continue
        # else...
        if i % 100 == 0:
            _clog(f"Processing {i=}")
        try:
            chunk = pd.DataFrame(
                [x[1] for x in index_and_row], index=[x[0] for x in index_and_row]
            )
            vectors = embeddings(chunk.content.tolist())
            chunk['embeddings'] = vectors
            store_chunk(i, chunk)
        except Exception as e:
            _clog(f"--> ERROR: {i=}, {e=}")


def compute_and_save_planar_embeddings(dacc=None, verbose=1):
    from imbed import umap_2d_embeddings

    dacc = dacc or mk_dacc()
    _clog = partial(clog, verbose)

    _clog("Getting flat_en_embeddings")
    dacc.flat_en_embeddings

    _clog(f"{len(dacc.flat_en_embeddings.shape)=}")
    _clog("Making an embeddings store from it, using flat_end_embeddings keys as keys")
    embdeddings_store = {
        id_: row.embeddings for id_, row in dacc.flat_en_embeddings.iterrows()
    }

    _clog("Offload the flat_en_embeddings from memory")
    del dacc.flat_en_embeddings

    _clog("Computing the 2d embeddings (the long process)...")
    planar_embeddings = umap_2d_embeddings(embdeddings_store)

    _clog("Reformatting the embeddings into a DataFrame")
    planar_embeddings = pd.DataFrame(planar_embeddings, index=['x', 'y']).T

    _clog("Saving the planar embeddings to planar_embeddings.parquet'")
    dacc.saves['planar_embeddings.parquet'] = planar_embeddings


def compute_and_save_planar_embeddings_light(dacc=None, verbose=1):
    from imbed import umap_2d_embeddings

    _clog = partial(clog, verbose)

    _clog("Loading embeddings")
    import numpy as np

    X = np.load(data_saves_join('flat_en_embeddings.npy'))

    target_vector_size = 500
    _clog(f"Taking the PCA ({target_vector_size=}) of the embeddings")
    from sklearn.decomposition import PCA

    X = PCA(n_components=target_vector_size).fit_transform(X)

    # _clog("Taking only part of them")
    # X = X[(X.shape[0] // 2):]

    _clog(f"{len(X.shape)=}")
    import umap

    _clog("And now, UMAP... Crossing fingers")
    planar_embeddings = umap.UMAP(n_components=2).fit_transform(X)

    _clog("deleting X")
    del X
    _clog("Saving the planar embeddings to planar_embeddings_3.npy'")
    np.save(
        data_saves_join('planar_embeddings_3.npy'),
        planar_embeddings,
    )


def compute_and_save_grouped_embeddings(dacc=None, verbose=1):
    from imbed import umap_2d_embeddings

    _clog = partial(clog, verbose)

    _clog("Loading grouped embeddings")
    dacc = dacc or mk_dacc()
    saves = dacc.saves

    d = dacc.flat_en_conversation_grouped_embeddings

    _clog("Making key-value store from them")
    d = {k: v['embeddings'] for k, v in d.iterrows()}
    _clog("Deleting dacc to free up memory")
    del dacc

    _clog("Computing planar embeddings")
    planar_embeddings = umap_2d_embeddings(d)

    _clog("Saving the planar embeddings to planar_embeddings_grouped.pkl'")
    saves['planar_embeddings_grouped.pkl'] = planar_embeddings


def compute_and_save_embeddings_pca(
    dacc=None,
    verbose: int = 1,
    *,
    pca_components: int = 100,
    chk_size: int = 50_000,
    data_name: str = 'pca_model.pkl',
):
    import numpy as np

    _clog = partial(clog, verbose)

    dacc = dacc or mk_dacc()
    saves = dacc.saves

    _clog("Loading data...")
    X = dacc.flat_en_embeddings.embeddings
    del dacc  # offloading data

    _clog("Making a data matrix X from it...")
    X = np.vstack(X.tolist())
    _clog(f"{len(X)=}")

    from sklearn.decomposition import IncrementalPCA

    pca = IncrementalPCA(n_components=pca_components)

    # fit the PCA by chunks
    chk_size = 10_000
    n = len(X)
    for i in range(0, n, chk_size):
        # print progress on the same line
        _clog(f"Rows {i}/{n}")
        X_chunk = X[i : (i + chk_size)]
        pca.partial_fit(X_chunk)

    if data_name:
        _clog(f"Saving the PCA to {data_name}")
        saves[data_name] = pca

    return pca


def compute_and_save_dbscan(
    dacc=None,
    verbose: int = 1,
    *,
    eps=0.3,
    min_samples=10,
    data_name='dbscan_pca100.npy',
    source_data_name='flat_en_embeddings_pca100.npy',
):
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    _clog = partial(clog, verbose)

    _clog("Loading data...")
    dacc = dacc or mk_dacc()

    X = dacc.saves[source_data_name]

    _clog("Standardizing the data...")
    X = StandardScaler().fit_transform(X)

    _clog(f"Computing DBSCAN(eps={eps}, min_samples={min_samples})...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

    rootdir = getattr(dacc.saves, 'rootdir', '')
    _clog(f"Saving the DBSCAN to {os.path.join(rootdir, data_name)}")
    dacc.saves[data_name] = dbscan

    return dbscan


if __name__ == '__main__':
    from argh import dispatch_commands

    dispatch_commands(
        [
            compute_and_save_embeddings,
            compute_and_save_planar_embeddings,
            compute_and_save_planar_embeddings_light,
            compute_and_save_grouped_embeddings,
            compute_and_save_embeddings_pca,
            compute_and_save_dbscan,
        ]
    )
