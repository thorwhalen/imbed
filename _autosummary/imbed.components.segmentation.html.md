# imbed.components.segmentation

Segmentation functions to get text segments.

This includes tools to extract segments from text data, but also transform standard
segments sources into a format ready to be used with the imbed library.

### Functions

| [`field_values`](#imbed.components.segmentation.field_values)(segments_src, field)                  | Extract values from a dictionary of segments based on a specific field.   |
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [`jdict_to_segments`](#imbed.components.segmentation.jdict_to_segments)(segments_src, \*[, str_handler]) | Convert various JSON-friendly formats to segments.                        |
| [`string_lines`](#imbed.components.segmentation.string_lines)(text)                                 | Split a string into lines, removing leading and trailing whitespace.      |

### imbed.components.segmentation.field_values(segments_src, field)

Extract values from a dictionary of segments based on a specific field.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### imbed.components.segmentation.jdict_to_segments(segments_src, \*, str_handler=<function string_lines>)

Convert various JSON-friendly formats to segments.

JSON-friendly formats we handle here are:

- str (a single string)
- list (and and iterable, of strings)
- dict (whose values are strings)

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### imbed.components.segmentation.string_lines(text)

Split a string into lines, removing leading and trailing whitespace.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
