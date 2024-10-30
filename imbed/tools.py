"""Tools around imbeddings tasks"""

import oa
from functools import partial
from operator import itemgetter
from typing import Iterable
import numpy as np

DFLT_N_SAMPLES = 99
DFLT_TRUNCATE_SEGMENT_AT_INDEX = 100


class ClusterLabeler:
    def __init__(
        self,
        truncate_segment_at_index=DFLT_TRUNCATE_SEGMENT_AT_INDEX,
        n_samples=DFLT_N_SAMPLES,
        context=' ',
        n_words=4,
        cluster_idx_col='cluster_idx',
        get_row_segments=itemgetter('segment'),
        max_unique_clusters=40,
    ):
        self.truncate_segment_at_index = truncate_segment_at_index
        self.n_samples = n_samples
        self.context = context
        self.n_words = n_words
        self.cluster_idx_col = cluster_idx_col
        self.get_row_segments = get_row_segments
        self.max_unique_clusters = max_unique_clusters

    @property
    def _title_data(self):
        return oa.prompt_function(
            """
            I want a title for the data below.
            Have the title be no more than {n_words:4} words long.
            I will give you the context of the data. 
            You should not include this context in the title. 
            Readers of the title will assume the context, so only particulars of 
            the data should be included in the title.
            The data represents a sample of the text segments of a particular topic.
            You should infer what the topic is and the title should be a very short 
            description of how that topic my differ from other topics of the same context.
            Again, your title should reflect the particulars of the text segments 
            within the given context, not the context itself.

            This is the context of the data: {context: }.
                                           
            The data:
                                           
            {data}
            """
        )

    def title_data(self, data):
        return self._title_data(data, n_words=self.n_words, context=self.context)

    def descriptions_of_segments(self, segments: Iterable[str]):
        """A method that returns the descriptions of a cluster"""
        random_sample_of_segments = np.random.choice(segments, self.n_samples)
        descriptions_text = '\n\n'.join(
            map(
                lambda x: x[: self.truncate_segment_at_index] + '...',
                filter(None, random_sample_of_segments),
            )
        )
        return descriptions_text

    def titles_of_segments(self, segments: Iterable[str]):
        """A method that returns the labels of a cluster"""
        data = self.descriptions_of_segments(segments)
        return self.title_data(data=data)

    def _cluster_idx_and_segments_sample(self, df):
        """A method that returns cluster indices and a sample of segments"""
        unique_clusters = df[self.cluster_idx_col].unique()

        if len(unique_clusters) > self.max_unique_clusters:
            raise ValueError(
                f'Too many unique clusters: {len(unique_clusters)} > {self.max_unique_clusters}. '
                'You can raise the `max_unique_clusters` parameter.'
            )

        for cluster_idx in unique_clusters:
            # Use get_row_segments to get the segments of each row
            segments = df[df[self.cluster_idx_col] == cluster_idx].apply(
                self.get_row_segments, axis=1
            )
            yield cluster_idx, segments

    def _label_clusters(self, df):
        """A method that returns labels for all clusters of a DataFrame"""
        for cluster_idx, segments in self._cluster_idx_and_segments_sample(df):
            yield cluster_idx, self.titles_of_segments(segments)

    def label_clusters(self, df):
        """A method that returns labels for all clusters of a DataFrame"""
        return dict(self._label_clusters(df))
