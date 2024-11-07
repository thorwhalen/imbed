"""Tools around imbeddings tasks"""

import oa
from functools import partial
from operator import itemgetter
from typing import Iterable, Generator
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
from typing import (
    Mapping,
    MutableMapping,
    Iterable,
    Optional,
    Dict,
    Callable,
    List,
    Union,
)
from itertools import chain

from lkj import clog
from dol import Pipe
from oa.stores import OaStores
from oa.batches import get_output_file_data
from oa.util import oa_extractors_obj, jsonl_loads_iter

from imbed.base import SegmentsSpec
from imbed.segmentation import fixed_step_chunker, chunk_mapping


def concat_lists(lists: Iterable[Iterable]):
    """Concatenate a list of lists into a single list."""
    return list(chain.from_iterable(lists))


# batch_imbed_extractor = SimpleNamespace(
#     embeddings_from_output_data = Pipe(oa_extractor, itemgetter('response.body.data.*.embedding')),
#     embeddings_from_file_obj = Pipe(oa_extractor, itemgetter('body.input'))
# )

extractors = oa_extractors_obj(
    embeddings_from_output_data='response.body.data.*.embedding',
    inputs_from_file_obj='body.input',
)


class EmbeddingBatchManager:
    def __init__(
        self,
        text_segments: SegmentsSpec,
        *,
        batcher: Union[int, Callable] = 1000,
        poll_interval: float = 5.0,
        max_polls: Optional[int] = None,
        on_fail: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable] = None,
        on_success: Optional[Callable[[Dict], None]] = None,
        verbosity: bool = 1,
        log_func: Callable = print,
        completed_batches_factory: Callable[[], MutableMapping] = list,
        misc_store_factory: Callable[[], MutableMapping] = dict,
    ):
        """
        Initialize the EmbeddingBatchManager.

        Args:
            text_segments: Iterable of text segments or a mapping of identifiers to text segments.
            batcher: Function that yields batches of an iterable input, or the size of a fixed-batch-size batcher.
            poll_interval: Time interval (in seconds) to wait between polling checks for batch completion.
            max_polls: Maximum number of polling attempts.
            on_fail: Callback function when a batch fails.
            on_progress: Callback function for progress updates.
            on_success: Callback function when a batch succeeds.
        """
        self.text_segments = text_segments
        if isinstance(self.text_segments, str):
            self.text_segments = [self.text_segments]

        self.batcher = batcher
        if isinstance(self.batcher, int):
            # get a batcher function that yields fixed-size batches
            batch_size = self.batcher
            self.batcher = partial(
                fixed_step_chunker, chk_size=batch_size, return_tail=True
            )

        self.poll_interval = poll_interval
        self.max_polls = max_polls or int(24 * 3600 / poll_interval)
        self.on_fail = on_fail
        self.on_progress = on_progress
        self.on_success = on_success

        self.oa_stores = OaStores()
        self.batches_info = (
            []
        )  # To store information about each batch (input_file_id, batch_id)
        self.completed_batches_factory = completed_batches_factory
        self.verbosity = verbosity
        self.log_func = log_func

        self.completed_batches = self.completed_batches_factory()
        self.misc_store = misc_store_factory()
        self._log_level_1 = clog(verbosity >= 1, log_func=log_func)
        self._log_level_2 = clog(verbosity >= 2, log_func=log_func)

    def batch_segments(
        self,
    ) -> Generator[Union[Mapping[str, str], List[str]], None, None]:
        """Split text segments into batches."""
        if isinstance(self.text_segments, Mapping):
            return chunk_mapping(self.text_segments, chunker=self.batcher)
        else:
            return self.batcher(self.text_segments)

    def upload_files(self, batched_segments) -> Iterable[str]:
        """Upload chunks and return a list of input file IDs.

        Note: It's a generator, and needs to be consumed to upload files.

        For example: uploaded_file_ids = list(self.upload_files(self.batch_segments()))
        """
        for segments_batch in batched_segments:
            yield self.oa_stores.files.create_embedding_task(
                segments_batch, custom_id_per_text=False
            )

    def launch_batch_processes(self, input_file_ids: List[str]) -> Iterable[str]:
        """Launch batch processes using the input file IDs.

        Note: It's a generator, and needs to actually launch batches.

        For example: batches = list(self.launch_processes(input_file_ids))
        """
        for file_id in input_file_ids:
            yield self.oa_stores.batches.append(file_id, endpoint="/v1/embeddings")

    def check_status(self, batch_id: str) -> str:
        """Check the status of a batch process."""
        batch = self.oa_stores.batches[batch_id]
        return batch.status

    def retrieve_segments_and_embeddings(self, batch_id: str) -> List:
        """Retrieve output embeddings for a completed batch."""
        output_data_obj = get_output_file_data(batch_id, oa_stores=self.oa_stores)

        batch = self.oa_stores.batches[batch_id]
        input_data_file_id = batch.input_file_id
        input_data = self.oa_stores.json_files[input_data_file_id]

        segments = extractors.inputs_from_file_obj(input_data)

        embeddings = concat_lists(
            map(
                extractors.embeddings_from_output_data,
                jsonl_loads_iter(output_data_obj.content),
            )
        )

        return segments, embeddings

    def launch_remote_processes(self):
        # Upload files and get input file IDs
        input_file_ids = list(self.upload_files(self.batch_segments()))
        self.misc_store['input_file_ids'] = input_file_ids

        # Launch processes and get batch IDs
        batches = list(self.launch_batch_processes(input_file_ids))
        self.misc_store['batches'] = batches

        return batches

    def segments_and_embeddings_completed_batches(
        self,
    ):
        """Retrieve all completed batches, and combine results"""
        for batch_id in self.completed_batches:
            yield self.retrieve_segments_and_embeddings(batch_id)

    def aggregate_completed_batches(self):
        import pandas as pd

        segments_and_embeddings = self.segments_and_embeddings_completed_batches()
        return pd.DataFrame(
            chain.from_iterable(
                zip(segments, embeddings)
                for segments, embeddings in segments_and_embeddings
            ),
            columns=['segment', 'embedding']
        )


    def run(self) -> Union[Dict[str, List[float]], List[List[float]]]:
        """Execute the entire batch processing workflow."""

        batches = self.launch_remote_processes()

        # Initialize tracking variables
        total_batches = len(batches)
        unfinished_batch_ids = batches

        for try_num in range(1, self.max_polls + 1):
            for batch in unfinished_batch_ids:
                batch_id = batch.id
                status = self.check_status(batch_id)

                # TODO: Use get_output_file_data for this logic
                if status == "completed":
                    # if self.on_progress:
                    #     self.on_progress(batch_id, total_batches)
                    # embeddings = self.retrieve_results(batch)
                    # self.completed_batches[batch_id] = embeddings
                    self.completed_batches.append(batch_id)
                    if self.on_success:
                        self.on_success(batch)
                    # check if we have all the batches
                    if len(self.completed_batches) == total_batches:
                        # ... and if so, aggregate the results and return them
                        return self.aggregate_completed_batches()
                elif status == "failed":
                    if self.on_fail:
                        self.on_fail(batch)
                    raise RuntimeError(f"Batch {batch_id} failed.")
                else:
                    self._log_level_1(
                        f"Batch {batch_id}: {try_num}/{self.max_polls} tries - Status: {status}. "
                        f"Checking again in {self.poll_interval} seconds..."
                    )
            time.sleep(self.poll_interval)  # wait a bit before trying again

        # If we reach this point, we've exceeded the maximum polling time
        raise TimeoutError(
            f"Batch {batch_id} did not complete within the maximum polling tries."
        )


# TODO: Add verbose option
# TODO: Return Polling object that can be iterogate via a generator?

from i2 import Sig


@Sig(EmbeddingBatchManager)
def compute_embeddings_in_bulk(*args, **kwargs):
    """
    Given a dictionary of {id: text_segment, ...}, uploads the data, submits it for batch
    embedding computation, and retrieves the result once complete.

    Args:
        text_segments (dict): A dictionary where keys are unique identifiers and values are text segments.
        poll_interval (int): Time interval (in seconds) to wait between polling checks for batch completion.

    Returns:
        dict: A dictionary with {id: embedding_vector, ...} for each input text segment.
    """
    return EmbeddingBatchManager(*args, **kwargs).run()
