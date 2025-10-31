"""
Segmentation functions to get text segments.

This includes tools to extract segments from text data, but also transform standard
segments sources into a format ready to be used with the imbed library.
"""

from collections.abc import Iterable
from typing import Union, Dict, List, TypeVar
from collections.abc import Callable, Mapping
from contextlib import suppress
from imbed.components.components_util import ComponentRegistry

suppress_import_errors = suppress(ImportError, ModuleNotFoundError)

K = TypeVar("K")
Text = str
Segment = str

SegmentsDict = dict[K, Segment]
SegmentsList = list[Segment]
Segments = Iterable[Segment]

# Create ComponentRegistry for segmenters
segmenters = ComponentRegistry('segmenters')

# --------------------------------------------------------------------------------------
# segmenters
# --------------------------------------------------------------------------------------


@segmenters.register()
def string_lines(text: Text) -> Segments:
    """
    Split a string into lines, removing leading and trailing whitespace.
    """
    return (line.strip() for line in text.splitlines() if line.strip())


@segmenters.register()
def jdict_to_segments(
    segments_src: Text | SegmentsDict | SegmentsList | Segments,
    *,
    str_handler: Callable = string_lines
) -> Segments:
    """
    Convert various JSON-friendly formats to segments.

    JSON-friendly formats we handle here are:
    - str (a single string)
    - list (and and iterable, of strings)
    - dict (whose values are strings)
    """
    if isinstance(segments_src, str):
        return str_handler(segments_src)
    elif isinstance(segments_src, (dict, list, tuple, Iterable)):
        return segments_src
    else:
        raise ValueError(
            "Unsupported JSON-friendly format (must be str, list, or dict)"
        )


@segmenters.register()
def field_values(segments_src: Mapping, field: str) -> Segments:
    """
    Extract values from a dictionary of segments based on a specific field.
    """
    return segments_src[field]


# --------------------------------------------------------------------------------------
# add default key
# --------------------------------------------------------------------------------------
# NOTE: This line must come towards end of module, after all segmenters are defined
from imbed.components.components_util import add_default_key

add_default_key(
    segmenters,
    default_key=jdict_to_segments,
    enviornment_var="DEFAULT_IMBED_SEGMENTER_KEY",
)
