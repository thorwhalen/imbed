"""Tools for segmentation"""

from itertools import islice, chain
from typing import Iterable, T, List, Callable, Optional

from typing import Tuple, List, Mapping

DocKey = str
SegmentKey = Tuple[DocKey, int, int]


class SegmentMapping:
    """A class to represent a mapping between segments and documents."""

    def __init__(self, docs: Mapping, segment_keys: List[SegmentKey]):
        self.docs = docs
        self.segment_keys = segment_keys
        self.document_keys = list(docs.keys())

    def __iter__(self):
        yield from self.segment_keys

    def __getitem__(self, key: SegmentKey):
        if isinstance(key, str):
            return self.docs[key]
        elif isinstance(key, Tuple):
            doc_key, start_idx, end_idx = key
            return self.docs[doc_key][start_idx:end_idx]
        else:
            raise TypeError("Key must be a string or a tuple")

    # TODO: Test
    def __setitem__(self, key: SegmentKey, value: str):
        if isinstance(key, str):
            self.docs[key] = value
            return
        else:
            doc_key, start_idx, end_idx = key
            self.segment_keys.append(key)
            self.docs[doc_key] = (
                self.docs.get(doc_key, '')[:start_idx]
                + value
                + self.docs.get(doc_key, '')[end_idx:]
            )

    def __add__(self, other):
        """Add two SegmentMapping objects together. This will concatenate the documents and segment keys."""
        return SegmentMapping(
            {**self.docs, **other.docs}, self.segment_keys + other.segment_keys
        )

    def __len__(self):
        return len(self.segment_keys)

    def __contains__(self, key: SegmentKey):
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


inf = float('inf')


def _validate_chk_size(chk_size):
    assert (
        isinstance(chk_size, int) and chk_size > 0
    ), 'chk_size should be a positive interger'


def _validate_chk_size_and_step(chk_size, chk_step):
    _validate_chk_size(chk_size)
    if chk_step is None:
        chk_step = chk_size
    assert (
        isinstance(chk_step, int) and chk_step > 0
    ), 'chk_step should be a positive integer'
    return chk_size, chk_step


def _validate_fixed_step_chunker_args(chk_size, chk_step, start_at, stop_at):
    chk_size, chk_step = _validate_chk_size_and_step(chk_size, chk_step)

    if start_at is None:
        start_at = 0
    if stop_at is not None:
        assert stop_at > start_at, 'stop_at should be larger than start_at'
        if stop_at is not inf:
            assert isinstance(stop_at, int), 'stop_at should be an integer'

    # checking a few things
    assert isinstance(start_at, int), 'start_at should be an integer'
    assert start_at >= 0, 'start_at should be a non negative integer'
    return chk_step, start_at


def rechunker(
    chks, chk_size, chk_step=None, start_at=None, stop_at=None, return_tail=False
):
    """Takes an iterable of chks and produces another iterable of chunks.
    The chunks generated by the input chks iterable is assumed to be gap-less and without overlap,
    but these do not need to be of fixed size.
    The output will be though.
    """
    yield from fixed_step_chunker(
        chain.from_iterable(chks), chk_size, chk_step, start_at, stop_at, return_tail
    )


def fixed_step_chunker(
    it: Iterable,
    chk_size: int,
    chk_step: Optional[int] = None,
    *,
    start_at: Optional[int] = None,
    stop_at: Optional[int] = None,
    return_tail: bool = False,
    chunk_egress: Callable[[Iterable[T]], Iterable[T]] = list,
) -> Iterable[List[T]]:
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
    [[1, 2, 3], [4, 5, 6]]
    >>> list(fixed_step_chunker([1, 2, 3, 4, 5, 6, 7, 8], chk_size=3, return_tail=True))
    [[1, 2, 3], [4, 5, 6], [7, 8]]
    >>> list(fixed_step_chunker([1, 2, 3, 4, 5, 6, 7, 8], chk_size=3, chk_step=2))
    [[1, 2, 3], [3, 4, 5], [5, 6, 7]]
    >>> chunks = fixed_step_chunker(
    ...     range(1, 17, 1), chk_size=3, chk_step=4,
    ...     start_at=1, stop_at=7, return_tail=True
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
