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

# -------------------------------------------------------------------------------------
# Embeddings computation in bulk
# TODO: WIP

import time
from operator import itemgetter
from typing import Mapping, Iterable, Optional
from itertools import chain

from dol import Pipe
from oa.stores import OaStores
from oa.batches import get_output_file_data
from oa.util import oa_extractor, jsonl_loads_iter



def concat_lists(lists: Iterable[Iterable]):
    """Contatinate a bunch of same lists together (like flatten)"""
    return list(chain(*lists))


get_embeddings_from_output_data = Pipe(oa_extractor, itemgetter('response.body.data.*.embedding'))


def compute_embeddings_in_bulk(text_segments, poll_interval: float = 5.0, max_polls: Optional[int]=None):
    """
    Given a dictionary of {id: text_segment, ...}, uploads the data, submits it for batch
    embedding computation, and retrieves the result once complete.
    
    Args:
        text_segments (dict): A dictionary where keys are unique identifiers and values are text segments.
        poll_interval (int): Time interval (in seconds) to wait between polling checks for batch completion.

    Returns:
        dict: A dictionary with {id: embedding_vector, ...} for each input text segment.
    """
    max_polls = max_polls or int(24 * 3600 / poll_interval)
    # Initialize OpenAI API client through OaStores
    oa_stores = OaStores()

    input_file_id = oa_stores.files.create_embedding_task(text_segments, custom_id_per_text=False)

    # Start the batch process for embeddings
    batch_id = oa_stores.batches.append(input_file_id, endpoint="/v1/embeddings")

    # TODO: Replace with more robust error handling (like get_output_file_data?)
    # Poll for batch completion
    for try_num in range(max_polls):
        batch = oa_stores.batches[batch_id]
        if batch.status == "completed":
            break
        elif batch.status == "failed":
            raise RuntimeError(f"Batch failed with error: {batch.get('error')}")
        else:
            print(f"Batch status: {batch.status}. Checking again in {poll_interval} seconds...")
            time.sleep(poll_interval)
    
    # Retrieve output embeddings
    output_data_obj = get_output_file_data(batch_id, oa_stores=oa_stores)
    # return output_data_obj
    output_data = concat_lists(map(get_embeddings_from_output_data, jsonl_loads_iter(output_data_obj.content)))

    if isinstance(text_segments, Mapping):
        return dict(zip(text_segments.keys(), output_data))
    else:
        return list(output_data)
    
    