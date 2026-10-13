# imbed.segmentation_util

Tools for segmentation, batching, chunking…

The words segmentation, batching, chunking, along with slicing, partitioning, etc.
are often used interchangeably in the context of data processing.
Here we will try to clarify the meaning of these terms in the context of our package

We will use the term “segmentation” when the process is about producing (smaller)
segments of text from a (larger) input text.

We will use the term “batching” when the process is about producing batches of
data from a data input stream (for example, an iterable of text segments that need
to be embedded, but we need to batch them to avoid resource limitation issues).

We will use the term “chunking” to denote a more general process of dividing a
sequence of items into chunks, usually of a fixed size.

### Functions

| [`chunk_mapping`](#imbed.segmentation_util.chunk_mapping)(mapping[, chunker])                 | Use the chunker to chunk the items of mapping, yielding sub-mappings                                                                                                            |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ensure_chunker`(chunker)                                                                          |                                                                                                                                                                                 |
| [`fixed_step_chunker`](#imbed.segmentation_util.fixed_step_chunker)(it, chk_size[, chk_step, ...]) | a function to get (an iterator of) segments (bt, tt) of chunks from an iterator (or list) of the for [it_1, it_2...], given a chk_size, chk_step, and a start_at and a stop_at. |
| [`rechunker`](#imbed.segmentation_util.rechunker)(chks, chk_size[, chk_step, ...])        | Takes an iterable of chks and produces another iterable of chunks.                                                                                                              |
| [`wrapped_chunker`](#imbed.segmentation_util.wrapped_chunker)(src, chunker, \*[, ingress, ...]) | A function to extend chunking functionality to any source of iterables, with the ability to wrap the chunks in a function before yielding them.                                 |
| [`yield_from`](#imbed.segmentation_util.yield_from)(it)                                    | A function to do `yield from it`.                                                                                                                                               |

### Classes

| [`SegmentStore`](#imbed.segmentation_util.SegmentStore)(docs, segment_keys)   | A class to represent a mapping between segments and documents.   |
|-------------------------------------------------------------------------------------|------------------------------------------------------------------|

### *class* imbed.segmentation_util.SegmentStore(docs, segment_keys)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A class to represent a mapping between segments and documents.

### imbed.segmentation_util.chunk_dataframe(src, chunker, \*, ingress=operator.methodcaller('iterrows'), egress=<function identity>)

Yield chunks of rows from a DataFrame.
The yielded chunks are lists of (index, row) tuples.

* **Return type:**
  [*Iterable*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[*ChunkBasedObj*]

### imbed.segmentation_util.chunk_mapping(mapping, chunker=None)

Use the chunker to chunk the items of mapping, yielding sub-mappings

* **Parameters:**
  * **chunker** (`Union`[[`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`IterableToChunk`, bound= [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`T`)])], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Chunk`, bound= [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`T`)])]], [`int`](https://docs.python.org/3/builtins/functions.html#int)]) – a chunker function
  * **mapping** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`T`)]) – a mapping of items
* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`T`)]]
* **Returns:**
  an iterator of sub-mappings in the form of dictionaries

### Examples

```pycon
>>> from functools import partial
>>> chunker = partial(fixed_step_chunker, chk_size=2)
>>> mapping = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e'}
>>> list(chunk_mapping(mapping, chunker))
[{1: 'a', 2: 'b'}, {3: 'c', 4: 'd'}, {5: 'e'}]
```

### imbed.segmentation_util.fixed_step_chunker(it, chk_size, chk_step=None, \*, start_at=None, stop_at=None, return_tail=True, chunk_egress=<class 'list'>)

a function to get (an iterator of) segments (bt, tt) of chunks from an iterator
(or list) of the for [it_1, it_2…], given a chk_size, chk_step, and a start_at
and a stop_at.
The start_at, stop_at act like slices indices for a list: start_at is included
and stop_at is excluded

* **Parameters:**
  * **it** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`T`)]) – iterator of elements of any type
  * **chk_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – length of the chunks
  * **chk_step** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – step between chunks
  * **start_at** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – index of the first term of the iterator at which we begin building
    the chunks (inclusive)
  * **stop_at** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – index of the last term from the iterator included in the chunks
  * **return_tail** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – if set to false, only the chunks with max element with index
    less than stop_at are yielded
    if set to true, any chunks with minimum index value no more than stop_at are
    returned but they contain term with index no more than stop_at
* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`T`)]]
* **Returns:**
  an iterator of the chunks

1. If stop_at is not None and return_tail is False:
   : will return all full chunks with maximum element index less than stop_at
     or until the iterator is exhausted. Only full chunks are returned here.
2. If stop_at is not None and return_tail is True:
   : will return all full chunks as above along with possibly cut off chunks
     containing one term whose index is stop_at-1 or one (last) term which is the
     last element of it
3. If stop_at is None and return_tail is False:
   : will return all full chunks with maximum element index less or equal to the last
     element of it
4. If stop_at is None and return_tail is True:
   : will return all full chunks with maximum element index less or equal to the last
     element of it plus cut off chunks whose maximum term index is the last term of it

### Examples

```pycon
>>> list(fixed_step_chunker([1, 2, 3, 4, 5, 6, 7, 8], chk_size=3))
[[1, 2, 3], [4, 5, 6], [7, 8]]
>>> list(fixed_step_chunker([1, 2, 3, 4, 5, 6, 7, 8], chk_size=3, return_tail=False))
[[1, 2, 3], [4, 5, 6]]
>>> list(fixed_step_chunker([1, 2, 3, 4, 5, 6, 7, 8], chk_size=3))
[[1, 2, 3], [4, 5, 6], [7, 8]]
>>> list(fixed_step_chunker([1, 2, 3, 4, 5, 6, 7, 8], chk_size=3, chk_step=2, return_tail=False))
[[1, 2, 3], [3, 4, 5], [5, 6, 7]]
>>> chunks = fixed_step_chunker(
...     range(1, 17, 1), chk_size=3, chk_step=4,
...     start_at=1, stop_at=7,
... )
>>> list(chunks)
[[2, 3, 4], [6, 7]]
```

### imbed.segmentation_util.rechunker(chks, chk_size, chk_step=None, start_at=None, stop_at=None, return_tail=False)

Takes an iterable of chks and produces another iterable of chunks.
The chunks generated by the input chks iterable is assumed to be gap-less and without overlap,
but these do not need to be of fixed size.
The output will be though.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Chunk`, bound= [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`T`)])]

### imbed.segmentation_util.wrapped_chunker(src, chunker, \*, ingress=<function identity>, egress=<function identity>)

A function to extend chunking functionality to any source of iterables,
with the ability to wrap the chunks in a function before yielding them.

* **Parameters:**
  * **src** ([`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`IterableSrc`, bound= [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`T`)])) – an iterable of items
  * **chunker** (`Union`[[`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`IterableToChunk`, bound= [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`T`)])], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Chunk`, bound= [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`T`)])]], [`int`](https://docs.python.org/3/builtins/functions.html#int)]) – a chunker function or an integer
  * **ingress** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`IterableSrc`, bound= [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`T`)])], [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`IterableToChunk`, bound= [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`T`)])]) – a function to wrap the input iterable
  * **egress** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Chunk`, bound= [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`T`)])], [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`ChunkBasedObj`)]) – a function to wrap the output chunks
* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`ChunkBasedObj`)]
* **Returns:**
  an iterator of chunk-based objects

### Examples

```pycon
>>> list(wrapped_chunker(range(1, 6), 2))
[[1, 2], [3, 4], [5]]
```

### imbed.segmentation_util.yield_from(it)

A function to do `yield from it`.
Looks like a chunker of chk_size=1, but careful, elements are not wrapped in lists.
