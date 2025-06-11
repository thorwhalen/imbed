"""
Example usage of the batch embeddings module.
"""

from imbed.oa_batch_embeddings import (
    compute_embeddings,
    EmbeddingsBatchProcess,
    compute_embeddings_df,
)


# Example 1: Simple blocking usage
def simple_example():
    """
    Example usage of compute_embeddings in a blocking manner.
    This function computes embeddings for a list of text segments
    and prints the results.

    >>> segments, embeddings = simple_example()  # doctest: +SKIP
    2025-03-28 14:41:06,828 - __main__ - INFO - Submitting batches for 4 segments
    2025-03-28 14:41:08,330 - __main__ - INFO - Submitted 1 batches
    2025-03-28 14:41:08,331 - __main__ - INFO - Monitoring 1 batches
    2025-03-28 14:41:14,062 - __main__ - INFO - Batch Batch(id='batch_67e6a6f40c408190aef26e761efd0b6f', completion_window='24h', created_at=1743169268, endpoint='/v1/embeddings', input_file_id='file-UkK9V2BcoTJosRCkPRdBTM', object='batch', status='validating', cancelled_at=None, cancelling_at=None, completed_at=None, error_file_id=None, errors=None, expired_at=None, expires_at=1743255668, failed_at=None, finalizing_at=None, in_progress_at=None, metadata=None, output_file_id=None, request_counts=BatchRequestCounts(completed=0, failed=0, total=0)) completed successfully
    2025-03-28 14:41:14,064 - __main__ - INFO - Batch processing complete: 1 successful, 0 failed
    Generated 4 embeddings
    First embedding (first 5 dimensions): [-0.018423624, -0.0072260704, 0.003638412, -0.054205045, -0.022725008]
    >>> segments  # doctest: +SKIP
    ['The quick brown fox jumps over the lazy dog.',
    'Machine learning models transform input data into useful representations.',
    'Embeddings capture semantic meaning in dense vector spaces.',
    'Natural language processing enables computers to understand human language.']
    >>> len(embeddings)  # doctest: +SKIP
    4
    >>> len(embeddings[0])  # doctest: +SKIP
    1536
    >>> len(embeddings[0][:5])  # doctest: +SKIP
    [-0.018421143, -0.007218754, 0.0036062053, -0.054197744, -0.022721948]

    """
    # Sample text segments
    segments = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models transform input data into useful representations.",
        "Embeddings capture semantic meaning in dense vector spaces.",
        "Natural language processing enables computers to understand human language.",
    ]

    # Compute embeddings (blocking call)
    result_segments, embeddings = compute_embeddings(
        segments=segments,
        verbosity=1,  # Show basic progress information
        batch_size=100,  # Small batch size for example purposes
    )

    # Print results
    print(f"Generated {len(embeddings)} embeddings")
    print(f"First embedding (first 5 dimensions): {embeddings[0][:5]}")

    return result_segments, embeddings


# Example 2: Non-blocking usage with manual control
def non_blocking_example():
    """
    Example usage of compute_embeddings in a non-blocking manner.
    Also demonstrates that when your segments are a dict (or Mapping), your output
    is also a dict.

    >>> segment_keys, embeddings = non_blocking_example()  # doctest: +SKIP
    2025-03-28 14:57:35,443 - imbed.oa_batch_embeddings - INFO - Submitting batches for 4 segments
    2025-03-28 14:57:36,784 - imbed.oa_batch_embeddings - DEBUG - Submitted batch batch_67e6aad079b08190a0e2b45e1aecd628 with 2 segments
    2025-03-28 14:57:37,819 - imbed.oa_batch_embeddings - DEBUG - Submitted batch batch_67e6aad18d5081909ec022687e474f65 with 2 segments
    2025-03-28 14:57:37,820 - imbed.oa_batch_embeddings - INFO - Submitted 2 batches
    2025-03-28 14:57:37,820 - imbed.oa_batch_embeddings - INFO - Monitoring 2 batches
    2025-03-28 14:57:38,001 - imbed.oa_batch_embeddings - DEBUG - Batch Batch(id='batch_67e6aad079b08190a0e2b45e1aecd628', completion_window='24h', created_at=1743170256, endpoint='/v1/embeddings', input_file_id='file-E5GXa9wne4UxRPn6YSeJAV', object='batch', status='validating', ...) status: in_progress
    Initial status summary: {'validating': 2, 'completed': 0, 'failed': 0}
    2025-03-28 14:57:38,419 - imbed.oa_batch_embeddings - DEBUG - Batch Batch(id='batch_67e6aad18d5081909ec022687e474f65', completion_window='24h', created_at=1743170257, endpoint='/v1/embeddings', input_file_id='file-V78FMHmz8xNkhfRtzkkBSP', object='batch', status='validating', ...) status: validating
    2025-03-28 14:57:41,589 - imbed.oa_batch_embeddings - INFO - Batch Batch(id='batch_67e6aad079b08190a0e2b45e1aecd628', completion_window='24h', created_at=1743170256, endpoint='/v1/embeddings', input_file_id='file-E5GXa9wne4UxRPn6YSeJAV', object='batch', status='validating', ...) completed successfully
    2025-03-28 14:57:41,793 - imbed.oa_batch_embeddings - DEBUG - Batch Batch(id='batch_67e6aad18d5081909ec022687e474f65', completion_window='24h', created_at=1743170257, endpoint='/v1/embeddings', input_file_id='file-V78FMHmz8xNkhfRtzkkBSP', object='batch', status='validating', ...) status: in_progress
    2025-03-28 14:57:43,999 - imbed.oa_batch_embeddings - DEBUG - Batch Batch(id='batch_67e6aad18d5081909ec022687e474f65', completion_window='24h', created_at=1743170257, endpoint='/v1/embeddings', input_file_id='file-V78FMHmz8xNkhfRtzkkBSP', object='batch', status='validating', ...) status: finalizing
    2025-03-28 14:57:47,555 - imbed.oa_batch_embeddings - INFO - Batch Batch(id='batch_67e6aad18d5081909ec022687e474f65', completion_window='24h', created_at=1743170257, endpoint='/v1/embeddings', input_file_id='file-V78FMHmz8xNkhfRtzkkBSP', object='batch', status='validating', ...) completed successfully
    2025-03-28 14:57:47,556 - imbed.oa_batch_embeddings - INFO - Batch processing complete: 2 successful, 0 failed
    Generated 4 embeddings for keys: ['fox', 'ml', 'embeddings', 'nlp']
    >>> segment_keys  # doctest: +SKIP
    ['fox', 'ml', 'embeddings', 'nlp']

    """
    # Sample text segments as a DICTIONARY
    segments = {
        "fox": "The quick brown fox jumps over the lazy dog.",
        "ml": "Machine learning models transform input data into useful representations.",
        "embeddings": "Embeddings capture semantic meaning in dense vector spaces.",
        "nlp": "Natural language processing enables computers to understand human language.",
    }

    # Get a process object instead of results
    process = compute_embeddings(
        segments=segments,
        verbosity=2,  # Show detailed progress information
        batch_size=2,  # Split into multiple batches
        poll_interval=3.0,  # Check status every 3 seconds
        return_process=True,  # Return the process instead of results
    )

    # Submit batches
    process.submit_batches()

    # Check status without blocking
    print("Initial status summary:", process.get_status_summary())

    # Now monitor until completion (this will block)
    process.monitor_batches()

    # Get and print results
    segment_keys, embeddings = process.aggregate_results()

    print(f"Generated {len(embeddings)} embeddings for keys: {segment_keys}")

    return segment_keys, embeddings


# Example 3: Using as a context manager
def context_manager_example():
    """
    Example usage of compute_embeddings as a context manager.
    This function computes embeddings for a list of text segments
    and prints the results.

    >>> segments, embeddings = context_manager_example()  # doctest: +SKIP
    2025-03-28 15:00:02,999 - imbed.oa_batch_embeddings - INFO - Submitting batches for 4 segments
    2025-03-28 15:00:05,087 - imbed.oa_batch_embeddings - INFO - Submitted 2 batches
    2025-03-28 15:00:05,088 - imbed.oa_batch_embeddings - INFO - Monitoring 2 batches
    2025-03-28 15:00:16,313 - imbed.oa_batch_embeddings - INFO - Batch Batch(id='batch_67e6ab64cb408190b80daa5e8ab92bf7', completion_window='24h', created_at=1743170404, ... status='validating', ...) completed successfully

    """
    # Sample text segments
    segments = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models transform input data into useful representations.",
        "Embeddings capture semantic meaning in dense vector spaces.",
        "Natural language processing enables computers to understand human language.",
    ]

    # Use as context manager for automatic cleanup
    with compute_embeddings(
        segments=segments, verbosity=1, batch_size=2, return_process=True
    ) as process:
        # Run the entire process
        segments, embeddings = process.run()

        print(f"Generated {len(embeddings)} embeddings in context")

    # After context exit, the processing mall is cleared (if not persist_processing_mall)

    return segments, embeddings


# Example 4: Using with pandas
def pandas_example():
    """
    Example usage of compute_embeddings with pandas DataFrame.
    This function computes embeddings for a dictionary of text segments
    and returns the results as a pandas DataFrame.

    >>> df = pandas_example()  # doctest: +SKIP
    2025-03-28 15:07:43,617 - imbed.oa_batch_embeddings - INFO - Submitting batches for 4 segments
    2025-03-28 15:07:45,843 - imbed.oa_batch_embeddings - INFO - Submitted 2 batches
    2025-03-28 15:07:45,844 - imbed.oa_batch_embeddings - INFO - Monitoring 2 batches
    2025-03-28 15:07:51,844 - imbed.oa_batch_embeddings - INFO - Batch Batch(id='batch_67e6ad3190048190ad587e2edd65ea33', ...) completed successfully
    2025-03-28 15:09:36,698 - imbed.oa_batch_embeddings - INFO - Batch Batch(id='batch_67e6ad306948819095e79cabd8263f0c', ...) completed successfully
    2025-03-28 15:09:36,700 - imbed.oa_batch_embeddings - INFO - Batch processing complete: 2 successful, 0 failed
    DataFrame shape: (4, 2)
    DataFrame index: ['fox', 'ml', 'embeddings', 'nlp']
    First row segment: The quick brown fox jumps over the lazy dog.
    First row embedding (first 5 dims): [4.308471e-05, -0.006475493, -0.00071540475, 0.018186275, 0.023950174]
                                                        segment  \
    fox              The quick brown fox jumps over the lazy dog.   
    ml          Machine learning models transform input data i...   
    embeddings  Embeddings capture semantic meaning in dense v...   
    nlp         Natural language processing enables computers ...   

                                                        embedding  
    fox         [4.308471e-05, -0.006475493, -0.00071540475, 0...  
    ml          [-0.021480283, 0.02021441, 0.012085131, 0.0159...  
    embeddings  [-0.018421143, -0.007218754, 0.0036062053, -0....  
    nlp         [0.015456875, 0.0016184314, 0.012820516, -0.04...

    """
    # Sample text segments
    segments = {
        "fox": "The quick brown fox jumps over the lazy dog.",
        "ml": "Machine learning models transform input data into useful representations.",
        "embeddings": "Embeddings capture semantic meaning in dense vector spaces.",
        "nlp": "Natural language processing enables computers to understand human language.",
    }

    # Get results as a pandas DataFrame
    df = compute_embeddings_df(segments=segments, verbosity=1, batch_size=2)

    # Display DataFrame information
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame index: {list(df.index)}")
    print(f"First row segment: {df.iloc[0]['segment']}")
    print(f"First row embedding (first 5 dims): {df.iloc[0]['embedding'][:5]}")

    return df


# Example 5: Error handling demonstration
def error_handling_example():
    """
    Example usage of compute_embeddings with error handling.
    This function demonstrates how to handle errors gracefully
    and continue processing valid segments.
    It intentionally includes invalid segments to trigger an error.
    The function will catch the error, print a message, and
    continue processing valid segments.
    This is useful for demonstrating error handling in a test environment.

    >>> result_segments, embeddings = error_handling_example()  # doctest: +SKIP
    2025-03-28 15:13:39,827 - imbed.oa_batch_embeddings - INFO - Submitting batches for 4 segments
    2025-03-28 15:13:39,828 - imbed.oa_batch_embeddings - INFO - Submitting batches for 2 segments
    Caught expected error: TypeError: argument 'text': 'int' object cannot be converted to 'PyString'
    Correcting and continuing with valid segments...
    2025-03-28 15:13:40,891 - imbed.oa_batch_embeddings - INFO - Submitted 1 batches
    2025-03-28 15:13:40,892 - imbed.oa_batch_embeddings - INFO - Monitoring 1 batches
    2025-03-28 15:13:56,657 - imbed.oa_batch_embeddings - INFO - Batch Batch(id='batch_67e6ae94932c819092092eb1bf91f3c9', ...) completed successfully
    2025-03-28 15:13:56,659 - imbed.oa_batch_embeddings - INFO - Batch processing complete: 1 successful, 0 failed
    Successfully generated 2 embeddings after correction
    >>> print(f"{len(result_segments)=}, {len(embeddings)=}")  # doctest: +SKIP
    len(result_segments)=2, len(embeddings)=2

    """
    try:
        # Intentionally invalid segments to trigger an error
        segments = [None, "Valid text", 123, "Another valid text"]

        result_segments, embeddings = compute_embeddings(segments=segments, verbosity=1)
    except Exception as e:
        print(f"Caught expected error: {type(e).__name__}: {str(e)}")

        # Show how to handle and continue
        print("Correcting and continuing with valid segments...")
        valid_segments = [seg for seg in segments if isinstance(seg, str)]
        result_segments, embeddings = compute_embeddings(
            segments=valid_segments, verbosity=1
        )

        print(f"Successfully generated {len(embeddings)} embeddings after correction")

    return result_segments, embeddings
