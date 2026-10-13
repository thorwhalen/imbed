# imbed.tools

Tools around imbeddings tasks

### Functions

| [`batch_processing_manager`](#imbed.tools.batch_processing_manager)(oa_stores, batches, ...)   | Sets up the ProcessingManager with the necessary functions and parameters.                                                                              |
|------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`cluster_labeler`](#imbed.tools.cluster_labeler)(df, \*[, ...])                      | A function that labels clusters give a DataFrame of text segments & cluster indices                                                                     |
| [`compute_embeddings_in_bulk`](#imbed.tools.compute_embeddings_in_bulk)(text_segments, \*)       | Given a dictionary of {id: text_segment, ...}, uploads the data, submits it for batch embedding computation, and retrieves the result once complete.    |
| `get_batch_obj`(oa_stores, batch)                                                                    |                                                                                                                                                         |
| [`get_output_file_data`](#imbed.tools.get_output_file_data)(batch, \*, oa_stores[, ...])   | Get the output file data for a batch, along with its status.                                                                                            |
| `on_completed_batch`(oa_stores, batch_obj)                                                           |                                                                                                                                                         |
| [`process_batches`](#imbed.tools.process_batches)(oa_stores, batches, \*[, ...])      | Processes a set of batches using ProcessingManager, checking their status in cycles until all are completed or the maximum number of cycles is reached. |

### Classes

| [`ClusterLabeler`](#imbed.tools.ClusterLabeler)(\*[, ...])   | A class that labels clusters give a DataFrame of text segments & cluster indices   |
|------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| `EmbeddingBatchManager`(text_segments, \*[, ...])                            |                                                                                    |

### *class* imbed.tools.ClusterLabeler(, truncate_segment_at_index=100, n_samples=99, context=' ', n_words=4, cluster_idx_col='cluster_idx', get_row_segments='segment', max_unique_clusters=40, prompt='\\\\nI want a title for the data below.\\\\nHave the title be no more than {n_words} words long.\\\\nI will give you the context of the data. \\\\nYou should not include this context in the title. \\\\nReaders of the title will assume the context, so only particulars of \\\\nthe data should be included in the title.\\\\nThe data represents a sample of the text segments of a particular topic.\\\\nYou should infer what the topic is and the title should be a very short \\\\ndescription of how that topic my differ from other topics of the same context.\\\\nAgain, your title should reflect the particulars of the text segments \\\\nwithin the given context, not the context itself.\\\\n\\\\nDo not surround the title with quotes or brackets or such.\\\\n\\\\nThis is the context of the data: {context}.\\\\n                \\\\nThe data:\\\\n                \\\\n{data}\\\\n')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A class that labels clusters give a DataFrame of text segments & cluster indices

#### descriptions_of_segments(segments)

A method that returns the descriptions of a cluster

#### label_clusters(df)

A method that returns labels for all clusters of a DataFrame

#### titles_of_segments(segments)

A method that returns the labels of a cluster

### imbed.tools.batch_processing_manager(oa_stores, batches, status_checking_frequency, max_cycles, get_output_file_data)

Sets up the ProcessingManager with the necessary functions and parameters.

* **Parameters:**
  * **oa_stores** – The OpenAI stores object for API interactions.
  * **batches** ([`set`](https://docs.python.org/3/builtins/stdtypes.html#set)[Batch]) – A set of batch objects to process.
  * **status_checking_frequency** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Minimum number of seconds per cycle.
  * **max_cycles** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Maximum number of cycles to perform.
  * **get_output_file_data** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – Function to get batch status and output data.
* **Returns:**
  An instance of ProcessingManager.
* **Return type:**
  `ProcessingManager`

### imbed.tools.cluster_labeler(df, , truncate_segment_at_index=100, n_samples=99, context=' ', n_words=4, cluster_idx_col='cluster_idx', get_row_segments='segment', max_unique_clusters=40, prompt='\\\\nI want a title for the data below.\\\\nHave the title be no more than {n_words} words long.\\\\nI will give you the context of the data. \\\\nYou should not include this context in the title. \\\\nReaders of the title will assume the context, so only particulars of \\\\nthe data should be included in the title.\\\\nThe data represents a sample of the text segments of a particular topic.\\\\nYou should infer what the topic is and the title should be a very short \\\\ndescription of how that topic my differ from other topics of the same context.\\\\nAgain, your title should reflect the particulars of the text segments \\\\nwithin the given context, not the context itself.\\\\n\\\\nDo not surround the title with quotes or brackets or such.\\\\n\\\\nThis is the context of the data: {context}.\\\\n                \\\\nThe data:\\\\n                \\\\n{data}\\\\n')

A function that labels clusters give a DataFrame of text segments & cluster indices

### imbed.tools.compute_embeddings_in_bulk(text_segments, \*, batcher=1000, poll_interval=5.0, max_polls=None, verbosity=1, log_func=<built-in function print>, store_factories={'completed_batches': <class 'list'>, 'embeddings': <class 'dict'>, 'submitted_batches': <class 'dict'>}, misc_store_factory=<class 'dict'>, imbed_task_dict_kwargs={'custom_id_per_text': False})

Given a dictionary of {id: text_segment, …}, uploads the data, submits it for batch
embedding computation, and retrieves the result once complete.

* **Parameters:**
  * **text_segments** (`Union`[`Segment` ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`Segment` ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str))], [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[`SegmentKey` ([`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`)), `Segment` ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str))]]) – A dictionary where keys are unique identifiers and values are text segments.
  * **poll_interval** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Time interval (in seconds) to wait between polling checks for batch completion.
* **Returns:**
  A dictionary with {id: embedding_vector, …} for each input text segment.
* **Return type:**
  [*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)

### imbed.tools.get_output_file_data(batch, \*, oa_stores, get_batch_obj=<function get_batch_obj>)

Get the output file data for a batch, along with its status.
Returns a tuple of (status, data), where data is None if not completed.

### imbed.tools.process_batches(oa_stores, batches, \*, status_checking_frequency=5.0, max_cycles=None, get_output_file_data=<function get_output_file_data>)

Processes a set of batches using ProcessingManager, checking their status in cycles until all are completed
or the maximum number of cycles is reached.

* **Parameters:**
  * **oa_stores** – The OpenAI stores object for API interactions.
  * **batches** ([`set`](https://docs.python.org/3/builtins/stdtypes.html#set)[Batch]) – A set of batch objects to process.
  * **status_checking_frequency** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Minimum number of seconds per cycle.
  * **max_cycles** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Maximum number of cycles to perform.
  * **get_output_file_data** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – Function to get batch status and output data.
* **Returns:**
  A dictionary of batch IDs to their output data.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]
