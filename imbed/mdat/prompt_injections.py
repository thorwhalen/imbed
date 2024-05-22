"""Prep for prompt injections data."""

from functools import cached_property
from dataclasses import dataclass, field
import pandas as pd

from imbed.base import HugfaceDaccBase, compute_and_save_embeddings, compute_and_save_planar_embeddings

@dataclass
class Dacc(HugfaceDaccBase):
    huggingface_data_stub: str = field(
        kw_only=True, default='deepset/prompt-injections'
    )

    label_key = 'label'
    data_attr = 'all_data'  # e.g. 'train_data', 'test_data', 'all_data'
    text_col = 'text'
    planar_embeddings_save_key = 'planar_embeddings.parquet'
    embeddings_key = 'embeddings'


    def __post_init__(self):
        super().__post_init__()

    @property
    def data(self):
        return getattr(self, self.data_attr)

    @cached_property
    def label_counts(self):
        """Series of label counts."""
        return self.all_data[self.label_key].value_counts()
    
    def compute_and_save_embeddings(self):
        return compute_and_save_embeddings(
            self.data, 
            save_store=self.embeddings_chunks_store, 
            text_col=self.text_col,
            embeddings_col=self.embeddings_key,
        )
    
    @cached_property
    def embeddings_df(self):
        return pd.concat(list(self.embeddings_chunks_store.values()))

    def compute_and_save_planar_embeddings(self):
        compute_and_save_planar_embeddings(
            self.embeddings_df[self.embeddings_key].to_dict(),
            save_store=self.saves,
            save_key=self.planar_embeddings_save_key,
        )

    @cached_property
    def planar_embeddings(self):
        return self.saves[self.planar_embeddings_save_key]
    

def mk_dacc(*, saves_dir=None):
    return Dacc(saves_dir=saves_dir)
