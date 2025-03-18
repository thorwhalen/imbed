"""
Tools for segmentation, batching, chunking...

The words segmentation, batching, chunking, along with slicing, partitioning, etc.
are often used interchangeably in the context of data processing.
Here we will try to clarify the meaning of these terms in the context of our package

We will use the term "segmentation" when the process is about producing (smaller)
segments of text from a (larger) input text.

We will use the term "batching" when the process is about producing batches of
data from a data input stream (for example, an iterable of text segments that need
to be embedded, but we need to batch them to avoid resource limitation issues).

We will use the term "chunking" to denote a more general process of dividing a
sequence of items into chunks, usually of a fixed size.

"""

from itertools import islice, chain
from operator import methodcaller
from functools import partial
from typing import (
    Iterable,
    T,
    List,
    Callable,
    Optional,
    Tuple,
    List,
    Mapping,
    KT,
    Dict,
    Sequence,
    Union,
    Any,
    TypeVar,
)

# TODO: Use these (and more) to complete the typing annotations
from imbed.util import identity
from imbed.base import Text, Texts, Segment, Segments

DocKey = KT
KeyAndIntervalSegmentKey = Tuple[DocKey, int, int]
Docs = Mapping[DocKey, Text]


class SegmentStore:
    """A class to represent a mapping between segments and documents."""

    def __init__(self, docs: Docs, segment_keys: List[KeyAndIntervalSegmentKey]):
        self.docs = docs
        self.segment_keys = segment_keys
        self.document_keys = list(docs.keys())

    def __iter__(self):
        yield from self.segment_keys

    def __getitem__(self, key: KeyAndIntervalSegmentKey) -> Segment:
        if isinstance(key, str):
            return self.docs[key]
        elif isinstance(key, Tuple):
            doc_key, start_idx, end_idx = key
            return self.docs[doc_key][start_idx:end_idx]
        else:
            raise TypeError("Key must be a string or a tuple")

    # TODO: Test
    def __setitem__(self, key: KeyAndIntervalSegmentKey, value: str):
        if isinstance(key, str):
            self.docs[key] = value
            return
        else:
            doc_key, start_idx, end_idx = key
            self.segment_keys.append(key)
            self.docs[doc_key] = (
                self.docs.get(doc_key, "")[:start_idx]
                + value
                + self.docs.get(doc_key, "")[end_idx:]
            )

    def __add__(self, other):
        """Add two SegmentStore objects together. This will concatenate the documents and segment keys."""
        return SegmentStore(
            {**self.docs, **other.docs}, self.segment_keys + other.segment_keys
        )

    def __len__(self) -> int:
        return len(self.segment_keys)

    def __contains__(self, key: KeyAndIntervalSegmentKey):
        if isinstance(key, str):
            return key in self.document_keys
        elif isinstance(key, Tuple):
            return key in self.segment_keys
        else:
            raise TypeError("Key must be a string or a tuple")

    def __repr__(self):
        representation = ""
        for key in self.segment_keys:
            representation += str(key) + " : " + str(self.__getitem__(key)) + "\n"
        return representation

    def values(self):
        for key in self.segment_keys:
            yield self.__getitem__(key)


inf = float("inf")


def _validate_chk_size(chk_size):
    assert (
        isinstance(chk_size, int) and chk_size > 0
    ), "chk_size should be a positive interger"


def _validate_chk_size_and_step(chk_size, chk_step):
    _validate_chk_size(chk_size)
    if chk_step is None:
        chk_step = chk_size
    assert (
        isinstance(chk_step, int) and chk_step > 0
    ), "chk_step should be a positive integer"
    return chk_size, chk_step


def _validate_fixed_step_chunker_args(chk_size, chk_step, start_at, stop_at):
    chk_size, chk_step = _validate_chk_size_and_step(chk_size, chk_step)

    if start_at is None:
        start_at = 0
    if stop_at is not None:
        assert stop_at > start_at, "stop_at should be larger than start_at"
        if stop_at is not inf:
            assert isinstance(stop_at, int), "stop_at should be an integer"

    # checking a few things
    assert isinstance(start_at, int), "start_at should be an integer"
    assert start_at >= 0, "start_at should be a non negative integer"
    return chk_step, start_at


# TODO: Make these generics (of T)
IterableToChunk = TypeVar("IterableToChunk", bound=Iterable[T])
Chunk = TypeVar("Chunk", bound=Sequence[T])
Chunks = Iterable[Chunk]
Chunker = Callable[[IterableToChunk], Chunks]
ChunkerSpec = Union[Chunker, int]


def fixed_step_chunker(
    it: Iterable[T],
    chk_size: int,
    chk_step: Optional[int] = None,
    *,
    start_at: Optional[int] = None,
    stop_at: Optional[int] = None,
    return_tail: bool = True,
    chunk_egress: Callable[[Iterable[T]], Iterable[T]] = list,
) -> Iterable[Sequence[T]]:
    """
    a function to get (an iterator of) segments (bt, tt) of chunks from an iterator
    (or list) of the for [it_1, it_2...], given a chk_size, chk_step, and a start_at
    and a stop_at.
    The start_at, stop_at act like slices indices for a list: start_at is included
    and stop_at is excluded

    :param it: iterator of elements of any type
    :param chk_size: length of the chunks
    :param chk_step: step between chunks
    :param start_at: index of the first term of the iterator at which we begin building
        the chunks (inclusive)
    :param stop_at: index of the last term from the iterator included in the chunks
    :param return_tail: if set to false, only the chunks with max element with index
        less than stop_at are yielded
    if set to true, any chunks with minimum index value no more than stop_at are
        returned but they contain term with index no more than stop_at
    :return: an iterator of the chunks

    1) If stop_at is not None and return_tail is False:
        will return all full chunks with maximum element index less than stop_at
        or until the iterator is exhausted. Only full chunks are returned here.

    2) If stop_at is not None and return_tail is True:
        will return all full chunks as above along with possibly cut off chunks
        containing one term whose index is stop_at-1 or one (last) term which is the
        last element of it

    3) If stop_at is None and return_tail is False:
        will return all full chunks with maximum element index less or equal to the last
        element of it

    4) If stop_at is None and return_tail is True:
        will return all full chunks with maximum element index less or equal to the last
        element of it plus cut off chunks whose maximum term index is the last term of it


    Examples:

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

    """

    chk_step, start_at = _validate_fixed_step_chunker_args(
        chk_size, chk_step, start_at, stop_at
    )

    if chk_step == chk_size and not return_tail:
        yield from map(chunk_egress, zip(*([iter(it)] * chk_step)))
    elif chk_step < chk_size:

        it = islice(it, start_at, stop_at)
        chk = chunk_egress(islice(it, chk_size))

        while len(chk) == chk_size:
            yield chk
            chk = chk[chk_step:] + chunk_egress(islice(it, chk_step))

    else:
        it = islice(it, start_at, stop_at)
        chk = chunk_egress(islice(it, chk_size))
        gap = chk_step - chk_size

        while len(chk) == chk_size:
            yield chk
            chk = chunk_egress(islice(it, gap, gap + chk_size))

    if return_tail:
        while len(chk) > 0:
            yield chk
            chk = chk[chk_step:]


fixed_step_chunker: Chunker


def rechunker(
    chks: Chunks,
    chk_size,
    chk_step=None,
    start_at=None,
    stop_at=None,
    return_tail=False,
) -> Chunks:
    """Takes an iterable of chks and produces another iterable of chunks.
    The chunks generated by the input chks iterable is assumed to be gap-less and without overlap,
    but these do not need to be of fixed size.
    The output will be though.
    """
    yield from fixed_step_chunker(
        chain.from_iterable(chks), chk_size, chk_step, start_at, stop_at, return_tail
    )


def yield_from(it):
    """A function to do `yield from it`.
    Looks like a chunker of chk_size=1, but careful, elements are not wrapped in lists.
    """
    yield from it


def ensure_chunker(chunker: ChunkerSpec) -> Chunker:
    if callable(chunker):
        return chunker
    elif isinstance(chunker, int):
        chk_size = chunker
        return partial(fixed_step_chunker, chk_size=chk_size)
    # elif isinstance(chunker, None):
    #     return yield_from  # TODO: Not a chunker, so what should we do?


IterableSrc = TypeVar("IterableSrc", bound=Iterable[T])
ChunkBasedObj = TypeVar("ChunkBasedObj")
# IterableToChunk = Iterable[T]


def wrapped_chunker(
    src: IterableSrc,
    chunker: ChunkerSpec,
    *,
    ingress: Callable[[IterableSrc], IterableToChunk] = identity,
    egress: Callable[[Chunk], ChunkBasedObj] = identity,
) -> Iterable[ChunkBasedObj]:
    """
    A function to extend chunking functionality to any source of iterables,
    with the ability to wrap the chunks in a function before yielding them.

    :param src: an iterable of items
    :param chunker: a chunker function or an integer
    :param ingress: a function to wrap the input iterable
    :param egress: a function to wrap the output chunks

    :return: an iterator of chunk-based objects

    Examples:

    >>> list(wrapped_chunker(range(1, 6), 2))
    [[1, 2], [3, 4], [5]]

    """
    iterable = ingress(src)
    chunker = ensure_chunker(chunker)
    for chunk in chunker(iterable):
        yield egress(chunk)


def chunk_mapping(
    mapping: Mapping[KT, T], chunker: ChunkerSpec = None
) -> Iterable[Dict[KT, T]]:
    """
    Use the chunker to chunk the items of mapping, yielding sub-mappings

    :param chunker: a chunker function
    :param mapping: a mapping of items

    :return: an iterator of sub-mappings in the form of dictionaries

    Examples:

    >>> from functools import partial
    >>> chunker = partial(fixed_step_chunker, chk_size=2)
    >>> mapping = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e'}
    >>> list(chunk_mapping(mapping, chunker))
    [{1: 'a', 2: 'b'}, {3: 'c', 4: 'd'}, {5: 'e'}]

    """
    return wrapped_chunker(mapping, chunker, ingress=methodcaller("items"), egress=dict)


chunk_dataframe = partial(wrapped_chunker, ingress=methodcaller("iterrows"))
chunk_dataframe.__doc__ = """
    Yield chunks of rows from a DataFrame.
    The yielded chunks are lists of (index, row) tuples.

    """
