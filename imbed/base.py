"""Base functionality of imbded."""

from typing import Callable, Protocol, Iterable, Sequence, Union, KT


Text = Union[str, KT]  # the text itself, or a key to retrieve it
Texts = Iterable[Text]
Vector = Sequence[float]
Vectors = Iterable[Vector]


class Embed(Protocol):
    """A callable that embeds text."""

    def __call__(self, text: Union[Text, Texts]) -> Union[Vector, Vectors]:
        """Embed a single text, or an iterable of texts.
        Note that this embedding could be calculated, or retrieved from a store,
        """


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
