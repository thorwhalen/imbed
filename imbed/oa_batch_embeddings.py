"""
Simplified interface for computing embeddings in bulk using OpenAI's batch API.

This module provides a clean, reusable interface for generating embeddings from
text segments using OpenAI's batch API, handling the async nature of the API
and providing status monitoring, error handling, and result aggregation.
"""

import time
import json
import logging
import asyncio
from typing import (
    Dict,
    List,
    Union,
    Optional,
    Callable,
    Any,
    TypeVar,
    Iterable,
    Mapping,
    MutableMapping,
    Tuple,
    Iterator,
)
from dataclasses import dataclass, field
from collections import defaultdict
from functools import partial
from contextlib import contextmanager
import numpy as np

from oa.stores import OaDacc
from oa.util import jsonl_loads_iter, concat_lists, extractors, ProcessingManager

# Define BatchRequestCounts since it's not available in oa.util
from typing import Optional
from dataclasses import dataclass


@dataclass
class BatchRequestCounts:
    """Counts of batch requests by status"""

    completed: int = 0
    failed: int = 0
    total: int = 0


from oa.batches import (
    BatchId,
    BatchObj,
    BatchSpec,
    mk_batch_file_embeddings_task,
    get_batch_obj,
    get_output_file_data,
)
from oa.base import DFLT_EMBEDDINGS_MODEL

from imbed.segmentation_util import fixed_step_chunker

# Type aliases for improved readability
Segment = str
Segments = Union[List[Segment], Dict[str, Segment]]
Embedding = List[float]
Embeddings = List[Embedding]
SegmentsMapper = Dict[BatchId, List[Segment]]
EmbeddingsMapper = Dict[BatchId, List[Embedding]]

# Default values
DFLT_BATCH_SIZE = 1000
DFLT_POLL_INTERVAL = 5.0  # seconds
DFLT_MAX_POLLS = None  # None means unlimited
DFLT_VERBOSITY = 1
DFLT_KEEP_PROCESSING_INFO = False


class BatchStatus:
    """Status constants for batch processing"""

    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"

    @classmethod
    def is_terminal(cls, status: str) -> bool:
        """Check if a status represents a terminal state"""
        return status in {cls.COMPLETED, cls.FAILED, cls.EXPIRED, cls.CANCELLED}

    @classmethod
    def is_success(cls, status: str) -> bool:
        """Check if a status represents successful completion"""
        return status == cls.COMPLETED


class BatchError(Exception):
    """Base exception for batch processing errors"""

    pass


from dol import (
    process_path,
    wrap_kvs,
    ensure_clear_to_kv_store,
    Files,
    JsonFiles as _JsonFiles,
    PickleFiles as _PickleFiles,
)
from tabled.wrappers import single_column_parquet_encode, single_column_parquet_decode

JsonFiles = ensure_clear_to_kv_store(_JsonFiles)
PickleFiles = ensure_clear_to_kv_store(_PickleFiles)
SingleColParquetFiles = ensure_clear_to_kv_store(
    wrap_kvs(
        Files,
        value_encoder=single_column_parquet_encode,
        value_decoder=single_column_parquet_decode,
    )
)
DFLT_MALL_STORE_CLASS = PickleFiles


dflt_store_cls_dict = {
    "current": PickleFiles,
    "segments": PickleFiles,
    "finished": PickleFiles,
    "erred": PickleFiles,
    "embeddings": PickleFiles,
}


class ProcessingMall(Mapping):
    """
    Container for all stores needed during batch processing.

    The ProcessingMall contains five stores, all keyed by batch_id:
    - current: Batches that are currently being processed
    - segments: The text segments corresponding to each batch
    - finished: Completed batches
    - erred: Batches that encountered errors
    - embeddings: The computed embeddings for each batch
    """

    _store_names = ("current", "segments", "finished", "erred", "embeddings")

    def __iter__(self) -> Iterator[str]:
        return iter(self._store_names)

    def __len__(self) -> int:
        return len(self._store_names)

    def __getitem__(self, key: str) -> MutableMapping:
        if key not in self._store_names:
            raise KeyError(f"Invalid store name: {key}")
        return getattr(self, key)

    def __init__(
        self,
        *,
        current: Optional[MutableMapping] = None,
        segments: Optional[MutableMapping] = None,
        finished: Optional[MutableMapping] = None,
        erred: Optional[MutableMapping] = None,
        embeddings: Optional[MutableMapping] = None,
    ):
        """
        Initialize the ProcessingMall with optional custom stores.

        Args:
            current: Store for batches currently being processed
            segments: Store for segments corresponding to each batch
            finished: Store for completed batches
            erred: Store for batches that encountered errors
            embeddings: Store for computed embeddings
        """
        self.current = current if current is not None else {}
        self.segments = segments if segments is not None else {}
        self.finished = finished if finished is not None else {}
        self.erred = erred if erred is not None else {}
        self.embeddings = embeddings if embeddings is not None else {}

    @classmethod
    def with_folder(cls, rootdir: str, store_cls=JsonFiles):
        """
        Create a ProcessingMall with stores that persist to local files.

        Args:
            rootdir: Directory where the stores will be saved
        """
        from imbed.util import DFLT_BATCHES_DIR
        import os

        if not isinstance(store_cls, Mapping):
            assert callable(
                store_cls
            ), "store_cls must be a callable or a dict of callables"
            store_cls = {name: store_cls for name in cls._store_names}

        # if rootdir doesn't have any path separator, and doesn't exist, assume it's a
        # a name to use to make an actual directory in the imbed app data dir
        if "/" not in rootdir and "\\" not in rootdir and not os.path.exists(rootdir):
            rootdir = os.path.join(DFLT_BATCHES_DIR, rootdir)

        def stores():
            for store_name in cls._store_names:
                store_rootdir = process_path(
                    rootdir, store_name, ensure_dir_exists=True
                )
                _store_cls = store_cls.get(store_name, DFLT_MALL_STORE_CLASS)
                yield store_name, _store_cls(store_rootdir)

        instance = cls(**dict(stores()))
        instance.rootdir = rootdir
        return instance

    def clear(self) -> None:
        """Clear all stores"""
        for store in (
            self.current,
            self.segments,
            self.finished,
            self.erred,
            self.embeddings,
        ):
            store.clear()

    def is_complete(self) -> bool:
        """Check if processing is complete (no batches in current)"""
        return len(self.current) == 0


class EmbeddingsBatchProcess:
    """
    Manages the lifecycle of batch embedding requests.

    This class handles submitting batch requests to the OpenAI API,
    monitoring their status, retrieving results, and aggregating them.
    It can be used as a context manager for automatic cleanup.
    """

    def __init__(
        self,
        segments: Optional[Segments] = None,
        processing_mall: Optional[Union[str, ProcessingMall]] = None,
        *,
        model: str = DFLT_EMBEDDINGS_MODEL,
        batch_size: int = DFLT_BATCH_SIZE,
        poll_interval: float = DFLT_POLL_INTERVAL,
        max_polls: Optional[int] = DFLT_MAX_POLLS,
        verbosity: int = DFLT_VERBOSITY,
        keep_processing_info: bool = DFLT_KEEP_PROCESSING_INFO,
        dacc: Optional[OaDacc] = None,
        logger: Optional[logging.Logger] = None,
        **embeddings_kwargs,
    ):
        """
        Initialize a new EmbeddingsBatchProcess for embedding generation.

        Args:
            segments: Text segments to embed, either as a list or a dictionary.
            model: OpenAI embedding model to use
            batch_size: Maximum number of segments per batch
            poll_interval: Seconds between status checks
            max_polls: Maximum number of status checks before timing out
            verbosity: Level of logging detail (0-2)
            processing_mall: Optional custom ProcessingMall
            keep_processing_info: Whether to keep mall data after completion
            dacc: Optional custom OaDacc instance
            logger: Optional custom logger
            **embeddings_kwargs: Additional parameters for embedding generation
        """
        self.segments = segments
        self.model = model
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        self.max_polls = max_polls or int(
            24 * 3600 / poll_interval
        )  # Default to 24h worth of polls
        self.verbosity = verbosity

        if processing_mall is None:
            processing_mall = ProcessingMall()
        elif isinstance(processing_mall, str):
            # If a string is provided, treat it as a directory for JsonFiles
            processing_mall = ProcessingMall.with_folder(processing_mall)
        elif not isinstance(processing_mall, ProcessingMall):
            raise TypeError(
                "processing_mall must be a ProcessingMall instance or a directory path"
            )
        self.processing_mall = processing_mall

        self.keep_processing_info = keep_processing_info
        self.dacc = dacc or OaDacc()
        self.embeddings_kwargs = embeddings_kwargs

        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.setLevel(
            logging.ERROR
            if verbosity == 0
            else logging.INFO if verbosity == 1 else logging.DEBUG
        )

        # Internal state
        self.processing_manager = None
        self._is_running = False
        self._is_complete = False
        self._result = None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        if not self.keep_processing_info and self._is_complete:
            self.processing_mall.clear()
        return False  # Don't suppress exceptions

    def _prepare_batches(self) -> Iterator[Tuple[List[Segment], BatchId]]:
        """
        Prepare and chunk segments into batches.

        Returns:
            Iterator yielding (segments_batch, batch_id) tuples
        """
        # Convert segments to a list if it's a dictionary
        if isinstance(self.segments, dict):
            segment_values = list(self.segments.values())
        else:
            segment_values = self.segments

        # Create a batcher function
        batcher = partial(
            fixed_step_chunker, chk_size=self.batch_size, return_tail=True
        )

        # Process segments in batches
        for segments_batch in batcher(segment_values):
            # Submit batch to OpenAI API
            batch_obj = self.dacc.launch_embedding_task(
                segments_batch, **self.embeddings_kwargs
            )
            # Extract the string ID from the batch object
            batch_id = batch_obj.id

            yield segments_batch, batch_id

    def submit_batches(self) -> Dict[BatchId, BatchObj]:
        """
        Submit all segment batches to the OpenAI API.

        Returns:
            Dictionary mapping batch IDs to batch objects
        """
        submitted_batches = {}

        self.logger.info(f"Submitting batches for {len(self.segments)} segments")

        # Process segments in batches
        for segments_batch, batch_id in self._prepare_batches():
            # Store batch info and segments
            batch_obj = self.dacc.s.batches[batch_id]
            # Use string batch_id as key
            self.processing_mall.current[batch_id] = batch_obj
            self.processing_mall.segments[batch_id] = segments_batch

            submitted_batches[batch_id] = batch_obj

            self.logger.debug(
                f"Submitted batch {batch_id} with {len(segments_batch)} segments"
            )

        self.logger.info(f"Submitted {len(submitted_batches)} batches")
        return submitted_batches

    def _process_batch_status(
        self, batch_id: BatchId, status: str, output_data: Any
    ) -> bool:
        """
        Process status updates for a batch.

        Args:
            batch_id: The ID of the batch
            status: Current status of the batch
            output_data: Data returned for completed batches

        Returns:
            True if batch is complete (terminal state), False otherwise
        """
        if status == BatchStatus.COMPLETED:
            self.logger.info(f"Batch {batch_id} completed successfully")

            # Extract embeddings from output data
            embeddings = concat_lists(
                map(
                    extractors.embeddings_from_output_data,
                    jsonl_loads_iter(output_data.content),
                )
            )

            # Store embeddings and move batch to finished
            self.processing_mall.embeddings[batch_id.id] = embeddings
            self.processing_mall.finished[batch_id.id] = self.processing_mall.current[
                batch_id.id
            ]
            del self.processing_mall.current[batch_id.id]

            return True

        elif BatchStatus.is_terminal(status):
            self.logger.warning(f"Batch {batch_id} ended with status: {status}")

            # Move batch to erred
            self.processing_mall.erred[batch_id.id] = self.processing_mall.current[
                batch_id.id
            ]
            del self.processing_mall.current[batch_id.id]

            return True

        else:
            self.logger.debug(f"Batch {batch_id} status: {status}")

            # Update batch status in current store
            # Ensure we're using string batch_id as key
            self.processing_mall.current[batch_id.id] = self.dacc.s.batches[batch_id.id]

            return False

    def monitor_batches(self) -> None:
        """
        Monitor the status of all submitted batches until completion.
        """
        if not self.processing_mall.current:
            self.logger.warning("No batches to monitor")
            self._is_complete = True
            return

        self.logger.info(f"Monitoring {len(self.processing_mall.current)} batches")
        self._is_running = True

        # Define the processing function
        def batch_processor(batch_id: BatchId) -> Tuple[str, Any]:
            try:
                # Get batch status and output data
                # Ensure we're using string batch_id
                batch_obj = self.dacc.s.batches[batch_id.id]
                status = batch_obj.status

                if status == BatchStatus.COMPLETED:
                    output_data = self.dacc.s.files_base[batch_obj.output_file_id]
                    return status, output_data
                else:
                    return status, None

            except Exception as e:
                self.logger.error(f"Error checking batch {batch_id}: {str(e)}")
                return BatchStatus.FAILED, None

        # Define wait time function to control polling interval
        def wait_time_function(cycle_duration: float, local_vars: dict) -> float:
            """Calculate how long to wait before the next cycle"""
            # Ensure we wait at least poll_interval seconds between checks
            return max(0.0, self.poll_interval - cycle_duration)

        # Create processing manager with dictionary of string batch IDs
        pending_items = {
            batch_id: batch_obj
            for batch_id, batch_obj in self.processing_mall.current.items()
        }

        self.processing_manager = ProcessingManager(
            pending_items=pending_items,
            processing_function=batch_processor,
            handle_status_function=self._process_batch_status,
            wait_time_function=wait_time_function,
            status_check_interval=self.poll_interval,
            max_cycles=self.max_polls,
        )

        # Process all batches
        self.processing_manager.process_items()

        self._is_running = False
        self._is_complete = self.processing_mall.is_complete()

        # Log summary
        self.logger.info(
            f"Batch processing complete: "
            f"{len(self.processing_mall.finished)} successful, "
            f"{len(self.processing_mall.erred)} failed"
        )

    def aggregate_results(self) -> Tuple[List[Segment], List[Embedding]]:
        """
        Aggregate all segments and embeddings from completed batches.

        Returns:
            Tuple of (all_segments, all_embeddings)
        """
        if not self._is_complete:
            raise BatchError(
                "Cannot aggregate results before processing is complete. "
                "Call monitor_batches() first or use run() to submit and monitor."
            )

        if self.processing_mall.erred:
            self.logger.warning(
                f"{len(self.processing_mall.erred)} batches failed. "
                f"Results will be incomplete."
            )

        all_segments = []
        all_embeddings = []

        # Collect all segments and embeddings in order
        for batch_id in self.processing_mall.finished:
            if (
                batch_id in self.processing_mall.segments
                and batch_id in self.processing_mall.embeddings
            ):
                segments = self.processing_mall.segments[batch_id]
                embeddings = self.processing_mall.embeddings[batch_id]

                # Ensure segments and embeddings align
                if len(segments) != len(embeddings):
                    self.logger.warning(
                        f"Mismatch in batch {batch_id}: "
                        f"{len(segments)} segments, {len(embeddings)} embeddings"
                    )
                    continue

                all_segments.extend(segments)
                all_embeddings.extend(embeddings)

        # If original segments was a dict, restore keys
        if isinstance(self.segments, dict):
            # This assumes order preservation, which is guaranteed in Python 3.7+
            keys = list(self.segments.keys())
            if len(keys) == len(all_segments):
                return keys, all_embeddings

        return all_segments, all_embeddings

    def run(self) -> Tuple[List[Segment], List[Embedding]]:
        """
        Execute the complete batch embedding workflow and return results.

        This method submits batches, monitors their status until completion,
        and returns the aggregated results.

        Returns:
            Tuple of (all_segments, all_embeddings)
        """
        # Submit batches
        self.submit_batches()

        # Monitor until completion
        self.monitor_batches()

        # Aggregate and return results
        result = self.aggregate_results()
        self._result = result

        return result

    @property
    def is_running(self) -> bool:
        """Check if batch processing is currently running"""
        return self._is_running

    @property
    def is_complete(self) -> bool:
        """Check if batch processing is complete"""
        return self._is_complete

    @property
    def result(self) -> Optional[Tuple[List[Segment], List[Embedding]]]:
        """Get the aggregated results if available"""
        return self._result

    def get_status_summary(self) -> Dict[str, int]:
        """
        Get a summary of batch statuses.

        Returns:
            Dictionary with counts of batches in each status
        """
        summary = defaultdict(int)

        # Count current batches by status
        for batch_id, batch_obj in self.processing_mall.current.items():
            summary[batch_obj.status] += 1

        # Add completed and failed batches
        summary["completed"] = len(self.processing_mall.finished)
        summary["failed"] = len(self.processing_mall.erred)

        return dict(summary)


# TODO: Review and refactor. Consider encorporating "normal" non-batch computation.
def compute_embeddings(
    segments: Segments,
    model: str = DFLT_EMBEDDINGS_MODEL,
    *,
    batch_size: int = DFLT_BATCH_SIZE,
    poll_interval: float = DFLT_POLL_INTERVAL,
    max_polls: Optional[int] = DFLT_MAX_POLLS,
    verbosity: int = DFLT_VERBOSITY,
    processing_mall: Optional[ProcessingMall] = None,
    keep_processing_info: bool = DFLT_KEEP_PROCESSING_INFO,
    dacc: Optional[OaDacc] = None,
    return_process: bool = False,
    logger: Optional[logging.Logger] = None,
    **embeddings_kwargs,
) -> Union[Tuple[List[Segment], List[Embedding]], EmbeddingsBatchProcess]:
    """
    Compute embeddings for text segments using OpenAI's batch API.

    This function manages the complete lifecycle of batch embedding requests,
    from submitting batches to the API, monitoring their status, and
    aggregating the results.

    Args:
        segments: Text segments to embed, either as a list or a dictionary
        model: OpenAI embedding model to use
        batch_size: Maximum number of segments per batch
        poll_interval: Seconds between status checks
        max_polls: Maximum number of status checks before timing out
        verbosity: Level of logging detail (0-2)
        processing_mall: Optional custom ProcessingMall
        keep_processing_info: Whether to keep mall data after completion
        dacc: Optional custom OaDacc instance
        return_process: If True, return the EmbeddingsBatchProcess object instead of results
        logger: Optional custom logger
        **embeddings_kwargs: Additional parameters for embedding generation

    Returns:
        If return_process is False (default):
            Tuple of (segments, embeddings)
        If return_process is True:
            EmbeddingsBatchProcess object for further interaction
    """
    # Create batch process
    process = EmbeddingsBatchProcess(
        segments=segments,
        model=model,
        batch_size=batch_size,
        poll_interval=poll_interval,
        max_polls=max_polls,
        verbosity=verbosity,
        processing_mall=processing_mall,
        keep_processing_info=keep_processing_info,
        dacc=dacc,
        logger=logger,
        **embeddings_kwargs,
    )

    # Return process if requested
    if return_process:
        return process

    # Otherwise, run the process and return results
    return process.run()


# Create a pandas-friendly wrapper
def compute_embeddings_df(
    segments: Segments, model: str = DFLT_EMBEDDINGS_MODEL, **kwargs
) -> 'pandas.DataFrame':
    """
    Compute embeddings and return results as a pandas DataFrame.

    Args:
        segments: Text segments to embed
        model: OpenAI embedding model to use
        **kwargs: Additional arguments passed to compute_embeddings

    Returns:
        DataFrame with 'segment' and 'embedding' columns
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for compute_embeddings_df")

    segments_result, embeddings = compute_embeddings(segments, model, **kwargs)

    # If segments_result is a list of keys (from dictionary input)
    if isinstance(segments, dict) and len(segments_result) == len(segments):
        # Restore the original text segments
        segments_text = [segments[key] for key in segments_result]
        # Create DataFrame with keys as index
        df = pd.DataFrame(
            {'segment': segments_text, 'embedding': embeddings}, index=segments_result
        )
    else:
        # Create standard DataFrame
        df = pd.DataFrame({'segment': segments_result, 'embedding': embeddings})

    return df
