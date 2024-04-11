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

