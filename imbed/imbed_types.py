"""Types for imbed"""

# ---------------------------------------------------------------------------------
# Typing
from typing import (
    Callable,
    Protocol,
    Iterable,
    Sequence,
    Union,
    KT,
    Mapping,
    Any,
    Optional,
    NewType,
    Tuple,
    Dict,
)

# Domain specific type aliases
# We use the convention that if THING is a type, then THINGs is an iterable of THING,
# and THINGMapping is a mapping from a key to a THING, and THINGSpec is a Union of
# objects that can specify THING explicitly or implicitly (for example, arguments to
# make a THING or the key to retrieve a THING).

Metadata = Any

# Text (also known as a document in some contexts)
Text = NewType("Text", str)
TextKey = NewType("TextKey", KT)
TextSpec = Union[str, TextKey]  # the text itself, or a key to retrieve it
Texts = Iterable[Text]
TextMapping = Mapping[TextKey, Text]

# The metadata of a text
TextMetadata = Metadata
MetadataMapping = Mapping[TextKey, TextMetadata]

# Text is usually segmented before vectorization.
# A Segment could be the whole text, or a part of the text (e.g. sentence, paragraph...)
Segment = NewType("Segment", str)
SegmentKey = NewType("SegmentKey", KT)
Segments = Iterable[Segment]
SingularTextSegmenter = Callable[[Text], Segments]
SegmentMapping = Mapping[SegmentKey, Segment]
SegmentsSpec = Union[Segment, Segments, SegmentMapping]

# NLP models often require a vector representation of the text segments.
# A vector is a sequence of floats.
# These vectors are also called embeddings.
Vector = Sequence[float]
Vectors = Iterable[Vector]
VectorMapping = Mapping[SegmentKey, Vector]
SingularSegmentVectorizer = Callable[[Segment], Vector]
BatchSegmentVectorizer = Callable[[Segments], Vectors]
SegmentVectorizer = Union[SingularSegmentVectorizer, BatchSegmentVectorizer]

# To visualize the vectors, we often project them to a 2d plane.
PlanarVector = Tuple[float, float]
PlanarVectors = Iterable[PlanarVector]
PlanarVectorMapping = Mapping[SegmentKey, PlanarVector]
SingularPlanarProjector = Callable[[Vector], PlanarVector]
BatchPlanarProjector = Callable[[Vectors], PlanarVectors]
PlanarProjector = Union[SingularPlanarProjector, BatchPlanarProjector]


EmbeddingType = Sequence[float]
Embedding = EmbeddingType  # backward compatibility alias
Embeddings = Iterable[Embedding]
EmbeddingMapping = Mapping[KT, Embedding]  # TODO: Same as VectorMapping. Refactor
PlanarEmbedding = Tuple[float, float]  # but really EmbeddingType of size two
PlanarVectorMapping = Dict[KT, PlanarEmbedding]

EmbeddingsDict = EmbeddingMapping
PlanarEmbeddingsDict = PlanarVectorMapping


class Embed(Protocol):
    """A callable that embeds text."""

    def __call__(self, text: Union[Text, Texts]) -> Union[Vector, Vectors]:
        """Embed a single text, or an iterable of texts.
        Note that this embedding could be calculated, or retrieved from a store,
        """
