"""HCP publications data"""

import os
from functools import cached_property, partial
from dataclasses import dataclass
from typing import Sequence, Callable
import pandas as pd
import numpy as np
import io

from dol import Files

from imbed.util import (
    umap_2d_embeddings,
    umap_2d_embeddings_df,
    two_d_embedding_dict_to_df,
    save_df_to_zipped_tsv,
)


# def embeddings_for_ids(ids: Sequence[int], embeddings):
#     return np.vstack(embeddings.loc[ids].values)
def embeddings_for_ids(ids: Sequence[int], embeddings_df: pd.DataFrame):
    return np.vstack(embeddings_df.loc[ids].values)


def aggregate_values(d: dict, value_aggregator: Callable, value_func: Callable):
    return {k: value_aggregator(value_func(v)) for k, v in d.items()}


# a version of the mean that *might* better reflect the cosine similarity between the vectors
def cosine_mean_aggregator(vectors: np.ndarray):
    return vectors.mean(axis=0) / np.linalg.norm(vectors.mean(axis=0))


# normalized means (where we first normalize each vector before taking the mean)
# Normalization is done by dividing each vector by its norm
# This is in order to (perhaps) better reflect the cosine similarity between the vectors
# Note: What about `vectors.mean(axis=0) / np.linalg.norm(vectors.mean(axis=0))`?
def normalized_mean_aggregator(vectors: np.ndarray):
    return (vectors / np.linalg.norm(vectors, axis=1)[:, None]).mean(axis=0)


simple_mean_aggregator = partial(np.mean, axis=0)


@dataclass
class Hcp3Dacc:
    src_key: str
    mean_aggregator: Callable[[np.ndarray], np.ndarray] = cosine_mean_aggregator
    src_store_factory: Callable = Files

    titles_aggr_src_key = 'publications-hcp3-with_titles_aggregates.parquet'
    embeddings_src_key = 'publications-hcp2-embeddings.parquet'
    citations_src_key = 'citation-links-hcp3.tsv.zip'
    info_src_key = 'publications-hcp2.tsv.zip'

    @cached_property
    def pjoin(self):
        return lambda *args: os.path.join(self.src_key, *args)

    @cached_property
    def store(self):
        return Files(self.src_key)

    @cached_property
    def embeddings_filepath(self):
        return self.pjoin(self.embeddings_src_key)

    @cached_property
    def citations_hcp3_src(self):
        return self.pjoin(self.citations_src_key)

    @cached_property
    def info_filepath(self):
        return self.pjoin(self.info_src_key)

    @cached_property
    def embeddings_hcp2(self):
        embeddings_hcp2 = pd.read_parquet(self.embeddings_filepath)
        embeddings_hcp2 = embeddings_hcp2.set_index('id')['embedding']
        return embeddings_hcp2

    @cached_property
    def _citations_hcp3(self):
        return pd.read_csv(self.citations_hcp3_src, sep='\t')

    @property
    def _citations_ids(self):
        return set(self._citations_hcp3['citing_pub_id']) | set(
            self._citations_hcp3['cited_pub_id']
        )

    @cached_property
    def embeddings_ids(self):
        return set(self.embeddings_hcp2.index)

    @cached_property
    def missing_embedding_ids(self):
        return self._citations_ids - self.embeddings_ids

    @cached_property
    def missing_cited_ids(dacc):
        cited_ids = set(dacc.citations_hcp3['cited_pub_id'])
        return set(dacc.info_hcp2['id']) - cited_ids

    @cached_property
    def citations_hcp3(self):
        return self._citations_hcp3[
            ~self._citations_hcp3['citing_pub_id'].isin(self.missing_embedding_ids)
            & ~self._citations_hcp3['cited_pub_id'].isin(self.missing_embedding_ids)
        ]

    @cached_property
    def cited_by(self):
        return (
            self.citations_hcp3.groupby('cited_pub_id')['citing_pub_id']
            .apply(set)
            .apply(list)
            .to_dict()
        )

    @cached_property
    def citations_of(self):
        return (
            self.citations_hcp3.groupby('citing_pub_id')['cited_pub_id']
            .apply(set)
            .apply(list)
            .to_dict()
        )

    @cached_property
    def ids(self):
        return next(iter(self.citations_of.values()))

    @cached_property
    def mean_aggregates(self):
        value_func = partial(embeddings_for_ids, embeddings_df=self.embeddings_hcp2)
        return aggregate_values(self.citations_of, self.mean_aggregator, value_func)

    @cached_property
    def planar_mean_embeddings(self):
        return umap_2d_embeddings_df(self.mean_aggregates, key_col='id')

    @cached_property
    def info_hcp2(self):
        return pd.read_csv(self.info_filepath, sep='\t', encoding='latin-1')

    @cached_property
    def planar_mean_embeddings_with_ids(self):
        return pd.merge(self.planar_mean_embeddings, self.info_hcp2, on='id')

    @cached_property
    def titles_aggr_df(self):
        """Precomputed titles aggregates, with info"""
        return pd.read_parquet(io.BytesIO(self.store[self.titles_aggr_src_key]))

    def titles_aggregate(
        dacc, *, titles_sep='\n\n###\n\n', main_title_sep='\n\n\n######\n\n\n'
    ):
        """For each article, aggregate its title with the titles of the papers it cites.

        We use `dacc.info_hcp2.set_index('id')['title']` to get the titles of an id.
        If a title is missing, it is ignored.

        Returns a generator of `(id, aggregated_titles)` pairs.
        """
        titles = dacc.info_hcp2.set_index('id')['title']
        for id, cited_ids in dacc.citations_of.items():
            cited_ids = sorted(set(cited_ids) - dacc.missing_cited_ids)
            cited_titles = titles.loc[cited_ids].dropna()
            citing_title = titles.loc[id]
            if citing_title and len(cited_titles):
                title_aggregate = (
                    citing_title + main_title_sep + titles_sep.join(cited_titles)
                )
                yield id, title_aggregate

        def titles_aggregate_sr(
            dacc,
            *,
            titles_sep='\n\n###\n\n',
            main_title_sep='\n\n\n######\n\n\n',
            print_progress_every=50_000,
        ):
            """For each article, aggregate its title with the titles of the papers it cites.

            We use `dacc.info_hcp2.set_index('id')['title']` to get the titles of an id.
            If a title is missing, it is ignored.

            Returns a generator of `(id, aggregated_titles)` pairs.
            """
            titles_aggregate_sr = dict()
            it = dacc.titles_aggregate(
                titles_sep=titles_sep, main_title_sep=main_title_sep
            )
            for i, (citing_id, titles_aggregate) in enumerate(it, 1):
                if i % print_progress_every == 0:
                    print(f"Proccessed {i} elements")
                titles_aggregate_sr[citing_id] = titles_aggregate
            return pd.Series(titles_aggregate_sr, name='titles_aggregate')
