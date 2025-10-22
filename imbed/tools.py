"""Tools around imbeddings tasks"""

import oa
from functools import partial
from operator import itemgetter
from typing import Union
from collections.abc import Iterable, Generator, Callable
import numpy as np

DFLT_N_SAMPLES = 99
DFLT_TRUNCATE_SEGMENT_AT_INDEX = 100


DFLT_LABELER_PROMPT = """
I want a title for the data below.
Have the title be no more than {n_words} words long.
I will give you the context of the data. 
You should not include this context in the title. 
Readers of the title will assume the context, so only particulars of 
the data should be included in the title.
The data represents a sample of the text segments of a particular topic.
You should infer what the topic is and the title should be a very short 
description of how that topic my differ from other topics of the same context.
Again, your title should reflect the particulars of the text segments 
within the given context, not the context itself.

Do not surround the title with quotes or brackets or such.

This is the context of the data: {context}.
                
The data:
                
{data}
"""


class ClusterLabeler:
    """
    A class that labels clusters give a DataFrame of text segments & cluster indices
    """

    def __init__(
        self,
        *,
        truncate_segment_at_index=DFLT_TRUNCATE_SEGMENT_AT_INDEX,
        n_samples=DFLT_N_SAMPLES,
        context=" ",
        n_words=4,
        cluster_idx_col="cluster_idx",
        get_row_segments: Callable | str = "segment",
        max_unique_clusters: int = 40,
        prompt: str = DFLT_LABELER_PROMPT,
    ):
        self.truncate_segment_at_index = truncate_segment_at_index
        self.n_samples = n_samples
        self.context = context
        self.n_words = n_words
        self.cluster_idx_col = cluster_idx_col
        if isinstance(get_row_segments, str):
            get_row_segments = itemgetter(get_row_segments)
        self.get_row_segments = get_row_segments
        self.max_unique_clusters = max_unique_clusters
        self.prompt = prompt

    @property
    def _title_data_prompt(self):
        prompt = self.prompt.replace("{n_words}", "{n_words:" + str(self.n_words) + "}")
        prompt = prompt.replace("{context}", "{context:" + str(self.context) + "}")
        return prompt

    @property
    def _title_data(self):
        return oa.prompt_function(self._title_data_prompt)

    def title_data(self, data):
        return self._title_data(data, n_words=self.n_words, context=self.context)

    def descriptions_of_segments(self, segments: Iterable[str]):
        """A method that returns the descriptions of a cluster"""
        random_sample_of_segments = np.random.choice(segments, self.n_samples)
        descriptions_text = "\n\n".join(
            map(
                lambda x: x[: self.truncate_segment_at_index] + "...",
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
                f"Too many unique clusters: {len(unique_clusters)} > {self.max_unique_clusters}. "
                "You can raise the `max_unique_clusters` parameter."
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
            yield cluster_idx, self._clean_title(self.titles_of_segments(segments))

    def _clean_title(self, title):
        title = title.strip()
        title = title.replace("\n", " ")
        # remove any quotes or brackets that might have been added
        title = title.strip("'\"[]")
        return title

    def label_clusters(self, df):
        """A method that returns labels for all clusters of a DataFrame"""
        return dict(self._label_clusters(df))


def cluster_labeler(
    df,
    *,
    truncate_segment_at_index=DFLT_TRUNCATE_SEGMENT_AT_INDEX,
    n_samples=DFLT_N_SAMPLES,
    context=" ",
    n_words=4,
    cluster_idx_col="cluster_idx",
    get_row_segments: Callable | str = "segment",
    max_unique_clusters: int = 40,
    prompt: str = DFLT_LABELER_PROMPT,
):
    """
    A function that labels clusters give a DataFrame of text segments & cluster indices
    """
    return ClusterLabeler(
        truncate_segment_at_index=truncate_segment_at_index,
        n_samples=n_samples,
        context=context,
        n_words=n_words,
        cluster_idx_col=cluster_idx_col,
        get_row_segments=get_row_segments,
        max_unique_clusters=max_unique_clusters,
        prompt=prompt,
    ).label_clusters(df)


# -------------------------------------------------------------------------------------
# Embeddings computation in bulk
# TODO: WIP


import time
from operator import itemgetter
from typing import (
    Optional,
    Dict,
    List,
    Union,
)
from collections.abc import Mapping, MutableMapping, Iterable, Callable
from itertools import chain
from types import SimpleNamespace

from lkj import clog
from dol import Pipe
from oa.stores import OaDacc
from oa.batches import get_output_file_data, mk_batch_file_embeddings_task
from oa.util import extractors, jsonl_loads_iter, concat_lists

from imbed.base import SegmentsSpec
from imbed.segmentation_util import fixed_step_chunker, chunk_mapping


class EmbeddingBatchManager:
    def __init__(
        self,
        text_segments: SegmentsSpec,
        *,
        batcher: int | Callable = 1000,
        poll_interval: float = 5.0,
        max_polls: int | None = None,
        verbosity: bool = 1,
        log_func: Callable = print,
        store_factories=dict(
            submitted_batches=dict, completed_batches=list, embeddings=dict
        ),
        misc_store_factory: Callable[[], MutableMapping] = dict,
        imbed_task_dict_kwargs=dict(custom_id_per_text=False),  # change to immutable?
    ):
        """
        Initialize the EmbeddingBatchManager.

        Args:
            text_segments: Iterable of text segments or a mapping of identifiers to text segments.
            batcher: Function that yields batches of an iterable input, or the size of a fixed-batch-size batcher.
            poll_interval: Time interval (in seconds) to wait between polling checks for batch completion.
            max_polls: Maximum number of polling attempts.
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

        self.dacc = OaDacc()

        local_stores = dict()
        for store_name, store_factory in store_factories.items():
            local_stores[store_name] = store_factory()

        self.local_stores = SimpleNamespace(**local_stores)

        self.batches_info = (
            []
        )  # To store information about each batch (input_file_id, batch_id)
        self.verbosity = verbosity
        self.log_func = log_func

        self.misc_store = misc_store_factory()
        self._log_level_1 = clog(verbosity >= 1, log_func=log_func)
        self._log_level_2 = clog(verbosity >= 2, log_func=log_func)

        self._imbed_task_dict_kwargs = dict(imbed_task_dict_kwargs)

        self.processing_manager = None

    def batch_segments(
        self,
    ) -> Generator[Mapping[str, str] | list[str], None, None]:
        """Split text segments into batches."""
        # TODO: Just return the chunk_mapping call, to eliminate the if-else?
        if isinstance(self.text_segments, Mapping):
            return chunk_mapping(self.text_segments, chunker=self.batcher)
        else:
            return self.batcher(self.text_segments)

    def check_status(self, batch_id: str) -> str:
        """Check the status of a batch process."""
        batch = self.dacc.s.batches[batch_id]
        return batch.status

    def retrieve_segments_and_embeddings(self, batch_id: str) -> list:
        """Retrieve output embeddings for a completed batch."""
        output_data_obj = self.dacc.get_output_file_data(batch_id)

        batch = self.dacc.s.batches[batch_id]
        input_data_file_id = batch.input_file_id
        input_data = self.dacc.s.json_files[input_data_file_id]

        segments = extractors.inputs_from_file_obj(input_data)

        embeddings = concat_lists(
            map(
                extractors.embeddings_from_output_data,
                jsonl_loads_iter(output_data_obj.content),
            )
        )

        return segments, embeddings

    def launch_remote_processes(self):
        """Launch remote processes for all batches."""
        # Upload files and get input file IDs
        for segments_batch in self.batch_segments():
            batch = self.dacc.launch_embedding_task(
                segments_batch, **self._imbed_task_dict_kwargs
            )
            self.local_stores.submitted_batches[batch.id] = batch  # remember this batch
            # self.local_stores.submitted_batches.append(batch)  # remember this batch
            yield batch

    def segments_and_embeddings_of_completed_batches(
        self, batches: Iterable[str] = None
    ):
        """Retrieve all completed batches, and combine results"""
        if batches is None:
            batches = self.local_stores.completed_batches
        for batch_id in batches:
            yield self.retrieve_segments_and_embeddings(batch_id)

    def aggregate_completed_batches(self, batches: Iterable[str] = None):
        segments_and_embeddings = self.segments_and_embeddings_of_completed_batches(
            batches
        )
        return list(
            chain.from_iterable(
                zip(segments, embeddings)
                for segments, embeddings in segments_and_embeddings
            ),
        )

    def aggregate_completed_batches_df(self, batches: Iterable[str] = None):
        import pandas as pd

        return pd.DataFrame(
            self.aggregate_completed_batches(batches),
            columns=["segment", "embedding"],
        )

    def get_batch_processing_manager(self, batches):
        return batch_processing_manager(
            self.dacc.s,
            batches,
            status_checking_frequency=self.poll_interval,
            max_cycles=self.max_polls,
            get_output_file_data=partial(get_output_file_data, oa_stores=self.dacc.s),
        )

    def run(self) -> dict[str, list[float]] | list[list[float]]:
        """Execute the entire batch processing workflow."""

        batches = dict()
        batches.update((batch.id, batch) for batch in self.launch_remote_processes())

        self.processing_manager = self.get_batch_processing_manager(batches)

        # go loop until all batches are completed, and the complete batches
        self.completed_batches = self.processing_manager.process_items()

        # return aggregated segments and embeddings
        return self.aggregate_completed_batches(self.completed_batches)
        # return self.completed_batches


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


# -------------------------------------------------------------------------------------

from typing import Optional, Any, Tuple, Dict, Set
from collections.abc import Callable

# Assuming get_output_file_data is defined as provided
# Assuming ProcessingManager is imported and defined as per your code

from oa.util import ProcessingManager

# from imbed_data_prep.embeddings_of_aggregations import *
from typing import Optional


def on_completed_batch(oa_stores, batch_obj):
    return oa_stores.files_base[batch_obj.output_file_id]


def get_batch_obj(oa_stores, batch):
    return oa_stores.batches[batch]


def get_output_file_data(
    batch: "Batch",
    *,
    oa_stores,
    get_batch_obj: Callable = get_batch_obj,
):
    """
    Get the output file data for a batch, along with its status.
    Returns a tuple of (status, data), where data is None if not completed.
    """
    batch_obj = get_batch_obj(oa_stores, batch)

    status = batch_obj.status

    if status == "completed":
        return status, on_completed_batch(oa_stores, batch_obj)
    else:
        return status, None


def batch_processing_manager(
    oa_stores,
    batches: set["Batch"],
    status_checking_frequency: float,
    max_cycles: int | None,
    get_output_file_data: Callable,
) -> ProcessingManager:
    """
    Sets up the ProcessingManager with the necessary functions and parameters.

    Args:
        oa_stores: The OpenAI stores object for API interactions.
        batches: A set of batch objects to process.
        status_checking_frequency: Minimum number of seconds per cycle.
        max_cycles: Maximum number of cycles to perform.
        get_output_file_data: Function to get batch status and output data.

    Returns:
        manager: An instance of ProcessingManager.
    """

    # Define the processing_function
    def processing_function(batch_id: str) -> tuple[str, Any | None]:
        status, output_data = get_output_file_data(batch_id, oa_stores=oa_stores)
        return status, output_data

    # Define the handle_status_function
    def handle_status_function(
        batch_id: str, status: str, output_data: Any | None
    ) -> bool:
        if status == "completed":
            print(f"Batch {batch_id} completed.")
            return True
        elif status == "failed":
            print(f"Batch {batch_id} failed.")
            return True
        else:
            print(f"Batch {batch_id} status: {status}")
            return False

    # Define the wait_time_function
    def wait_time_function(cycle_duration: float, local_vars: dict) -> float:
        status_check_interval = local_vars["self"].status_check_interval
        sleep_duration = max(0, status_check_interval - cycle_duration)
        return sleep_duration

    # Prepare pending_items for ProcessingManager
    # pending_batches = {batch.id: batch.id for batch in batches}
    pending_batches = batches.copy()

    # Initialize the ProcessingManager
    manager = ProcessingManager(
        pending_items=pending_batches,
        processing_function=processing_function,
        handle_status_function=handle_status_function,
        wait_time_function=wait_time_function,
        status_check_interval=status_checking_frequency,
        max_cycles=max_cycles,
    )

    return manager


def process_batches(
    oa_stores,
    batches: set["Batch"],
    *,
    status_checking_frequency: float = 5.0,
    max_cycles: int | None = None,
    get_output_file_data: Callable = get_output_file_data,
) -> dict[str, Any]:
    """
    Processes a set of batches using ProcessingManager, checking their status in cycles until all are completed
    or the maximum number of cycles is reached.

    Args:
        oa_stores: The OpenAI stores object for API interactions.
        batches: A set of batch objects to process.
        status_checking_frequency: Minimum number of seconds per cycle.
        max_cycles: Maximum number of cycles to perform.
        get_output_file_data: Function to get batch status and output data.

    Returns:
        completed_batches: A dictionary of batch IDs to their output data.
    """

    manager = batch_processing_manager(
        oa_stores, batches, status_checking_frequency, max_cycles, get_output_file_data
    )

    # Start the processing loop
    manager.process_items()

    # Collect completed batches
    completed_batches = (
        manager.completed_items
    )  # completed_items doesn't seem to exist anymore

    # Optionally, you can handle any remaining batches if max_cycles was reached
    if manager.pending_items:
        print(f"Max cycles reached. The following batches did not complete:")
        for batch_id in manager.pending_items.keys():
            print(f"- Batch {batch_id}")

    return completed_batches
