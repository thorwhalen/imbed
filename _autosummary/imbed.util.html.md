# imbed.util

Utils for imbed package.

### Functions

| [`async_call`](#imbed.util.async_call)(func, \*args, \*\*kwargs)             | Calls a function, awaiting it if it's asynchronous, and running it in a thread if it's synchronous.                                                                                            |
|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`async_sync_wrapper`](#imbed.util.async_sync_wrapper)(func)                         | A decorator that adds an async and a sync version of a function.                                                                                                                               |
| [`cosine_similarity`](#imbed.util.cosine_similarity)(u, v, \*[, cartesian_product]) | Computes the cosine similarity between two vectors or arrays of vectors.                                                                                                                       |
| `counts`(sr)                                                                                      |                                                                                                                                                                                                |
| `dict_slice`(d, \*args)                                                                           |                                                                                                                                                                                                |
| `ensure_cache`(cache)                                                                             |                                                                                                                                                                                                |
| [`ensure_embedding_dict`](#imbed.util.ensure_embedding_dict)(embeddings)                | Ensure that the embeddings are in the correct format.                                                                                                                                          |
| [`ensure_fullpath`](#imbed.util.ensure_fullpath)(filepath[, conditional_rootdir]) | Ensures a full path, prepending a rootdir if input is a (slash-less) file name.                                                                                                                |
| [`ensure_segments_mapping`](#imbed.util.ensure_segments_mapping)(segments)                | Ensure that the segments are in the correct format.                                                                                                                                            |
| [`fuzzy_induced_graph`](#imbed.util.fuzzy_induced_graph)(graph, inducing_node_set)    | Keep only those (node, neighbors) pairs where both node and a minimum proportion of neighbors are in inducing_node_set.                                                                        |
| [`hash_text`](#imbed.util.hash_text)(text)                                  | Return a hash of the text, ignoring punctuation and capitalization.                                                                                                                            |
| `identity`(x)                                                                                     |                                                                                                                                                                                                |
| `is_string_with_path_seps`(x)                                                                     |                                                                                                                                                                                                |
| `is_submodule_path`(path)                                                                         |                                                                                                                                                                                                |
| `lenient_bytes_decoder`(bytes_)                                                                   |                                                                                                                                                                                                |
| [`load_if_saved`](#imbed.util.load_if_saved)([key, store_attr, ...])            | Decorator to load the value from the store if it is saved, otherwise compute it.                                                                                                               |
| `lower_alphanumeric`(text)                                                                        |                                                                                                                                                                                                |
| [`match_aliases`](#imbed.util.match_aliases)(fields, aliases)                   | Matches the keys of aliases to the given fields, using the values of aliases as the matching logic (could be a list of possible fields, a regular expression, or a custom matching function.). |
| [`merge_data`](#imbed.util.merge_data)(data_1, data_2, \*[, merge_on, ...])  | Merges two sources of data, returning a dataframe.                                                                                                                                             |
| `module_name`(path)                                                                               |                                                                                                                                                                                                |
| [`planar_embeddings`](#imbed.util.planar_embeddings)(kd_embeddings, \*[, ...])      | Takes a mapping of k-dimensional (kd) embeddings and returns a dict of the 2d umap embeddings                                                                                                  |
| [`planar_embeddings_dict_to_df`](#imbed.util.planar_embeddings_dict_to_df)(...[, x_col, ...])  | A function that takes a dict of planar embeddings and returns a pandas DataFrame of the 2d embeddings                                                                                          |
| `planar_embeddings_func`([embeddings_func, ...])                                                  |                                                                                                                                                                                                |
| `submodules_of`(pkg[, include_init])                                                              |                                                                                                                                                                                                |
| [`transpose_iterable`](#imbed.util.transpose_iterable)(iterable)                     | This is useful to do things like:                                                                                                                                                              |
| [`two_d_embedding_dict_to_df`](#imbed.util.two_d_embedding_dict_to_df)(...[, x_col, ...])    | A function that takes a dict of planar embeddings and returns a pandas DataFrame of the 2d embeddings                                                                                          |
| [`umap_2d_embeddings_df`](#imbed.util.umap_2d_embeddings_df)(kd_embeddings, \*[, ...])  | A function that takes a mapping of kd embeddings and returns a pandas DataFrame of the 2d umap embeddings                                                                                      |

### Classes

| `Embeddings`(embeddings[, keys, meta, ...])   |    |
|-----------------------------------------------|----|

### *async* imbed.util.async_call(func, \*args, \*\*kwargs)

Calls a function, awaiting it if it’s asynchronous, and running it in a thread if it’s synchronous.

* **Parameters:**
  * **func** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`ParamSpec`](https://docs.python.org/3/library/typing.html#typing.ParamSpec)(`P`, bound= [`None`](https://docs.python.org/3/builtins/constants.html#None))], [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`R`)]) – The function to call.
  * **\*args** ([`ParamSpecArgs`](https://docs.python.org/3/library/typing.html#typing.ParamSpecArgs)) – Positional arguments to pass to the function.
  * **\*\*kwargs** ([`ParamSpecKwargs`](https://docs.python.org/3/library/typing.html#typing.ParamSpecKwargs)) – Keyword arguments to pass to the function.
* **Return type:**
  [`Coroutine`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Coroutine)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`R`)]
* **Returns:**
  The result of the function call.

### imbed.util.async_sync_wrapper(func)

A decorator that adds an async and a sync version of a function.

### imbed.util.cosine_similarity(u, v, , cartesian_product=False)

Computes the cosine similarity between two vectors or arrays of vectors.

If both inputs are 1D vectors, returns a float.
If one or both inputs are 2D arrays, returns either a 1D array (row-wise)
or a 2D array (cartesian product of rows) depending on the cartesian_product flag.

Behavior for row-wise (cartesian_product=False):

> - If both arrays have the same number of rows, compares row i of u to row i of v.
> - If one array has only 1 row, it is broadcast against each row of the other array.
>   (Returns a 1D array of length k, where k is the number of rows in the multi-row array.)
* **Parameters:**
  * **u** (*array-like*) – A single vector (1D) or a 2D array (k1 x n),
    where each row is a separate vector.
  * **v** (*array-like*) – A single vector (1D) or a 2D array (k2 x n).
  * **cartesian_product** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool) *,* *optional*) – 
    - If False (default), the function compares rows in a one-to-one fashion (u[i] vs. v[i]),
      **except** if one array has exactly 1 row and the other has multiple rows, in which case
      that single row is broadcast to all rows of the other array.
    - If True, computes the similarity for every combination of rows
      (results in a 2D array of shape (k1, k2)).
* **Returns:**
  - A float if both u and v are 1D vectors.
  - A 1D numpy array if either u or v is 2D and cartesian_product=False.
  - A 2D numpy array if cartesian_product=True.
* **Return type:**
  [*float*](https://docs.python.org/3/builtins/functions.html#float) or np.ndarray
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – 
  - If the number of columns in u and v do not match.
        - If cartesian_product=False, both arrays have multiple rows but differ in row count.

### Examples

[See here for an explanation of the cases](https://github.com/thorwhalen/imbed/discussions/9#discussioncomment-11968528).

[See here for a performance comparison of numpy (this function) versus scipy](https://github.com/thorwhalen/imbed/discussions/9#discussioncomment-11971474).

Case 1: Both are single 1D vectors

```pycon
>>> u1d = [2, 0]
>>> v1d = [2, 0]
>>> float(cosine_similarity(u1d, v1d))
1.0
```

Case 2: Single 1D vector vs. a 2D array (row-wise broadcast)

```pycon
>>> import numpy as np
>>> M1 = np.array([
...     [2, 0],
...     [0, 2],
...     [2, 2]
... ])
>>> cosine_similarity(u1d, M1)
array([1.        , 0.        , 0.70710678...])
```

Case 3: Two 2D arrays of different row lengths, cartesian_product=False (raises ValueError)

```pycon
>>> M2_different = np.array([
...     [0, 2],
...     [2, 2]
... ])
>>> # Expect a ValueError because M1 has 3 rows and M2_different has 2 rows
>>> cosine_similarity(M1, M2_different, cartesian_product=False)
Traceback (most recent call last):
    ...
ValueError: For row-wise comparison, u and v must have the same number of rows...
```

Case 4: Two 2D arrays of the same number of rows, cartesian_product=False

```pycon
>>> M2 = np.array([
...     [0, 2],
...     [2, 0],
...     [2, 2]
... ])
>>> cosine_similarity(M1, M2, cartesian_product=False)
array([0., 0., 1.])
```

Case 5: Two 2D arrays of the same size, `cartesian_product=True`
(computes every combination of rows => 3 x 3)

```pycon
>>> res5 = cosine_similarity(M1, M2, cartesian_product=True)
>>> np.round(res5, 3)
array([[0.   , 1.   , 0.707],
       [1.   , 0.   , 0.707],
       [0.707, 0.707, 1.   ]])
```

### imbed.util.ensure_embedding_dict(embeddings)

Ensure that the embeddings are in the correct format.

* **Parameters:**
  **embeddings** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – a dict of embeddings
* **Return type:**
  [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  a dict of embeddings

### imbed.util.ensure_fullpath(filepath, conditional_rootdir='')

Ensures a full path, prepending a rootdir if input is a (slash-less) file name.

If you pass in a file name, it will be considered relative to the current directory.
In all other situations, the conditional_rootdir is ignored, and the filepath is
taken at face value.
All outputs will be processed to ensure a full path is returned.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> ensure_fullpath('apple/sauce')
'.../apple/sauce'
>>> assert (
...     ensure_fullpath('apple/sauce')
...     == ensure_fullpath('./apple/sauce')
...     == ensure_fullpath('apple/sauce', '')
... )
```

The only time you actually use the rootdir is when you pass in a file name
that doesn’t have slashes in it.

```pycon
>>> ensure_fullpath('apple', '/root/dir')
'/root/dir/apple'
```

### imbed.util.ensure_segments_mapping(segments)

Ensure that the segments are in the correct format.

* **Parameters:**
  **segments** (`Union`[`Segment` ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`Segment` ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str))], [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[`SegmentKey` ([`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`)), `Segment` ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str))]]) – a SegmentMapping or a Sequence of Segments
* **Return type:**
  [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[`SegmentKey` ([`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`)), `Segment` ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str))]
* **Returns:**
  a SegmentMapping

### imbed.util.fuzzy_induced_graph(graph, inducing_node_set, min_proportion=1)

Keep only those (node, neighbors) pairs where both node and a minimum proportion of
neighbors are in inducing_node_set.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]]]

### imbed.util.hash_text(text)

Return a hash of the text, ignoring punctuation and capitalization.

```pycon
>>> hash_text('Hello, world!')
'5eb63bbbe01eeed093cb22bb8f5acdc3'
>>> hash_text('hello world')
'5eb63bbbe01eeed093cb22bb8f5acdc3'
>>> hash_text('Hello, world!') == hash_text('hello world')
True
```

### imbed.util.load_if_saved(key=None, store_attr='saves', save_on_compute=True, print_when_loading_from_file=True)

Decorator to load the value from the store if it is saved, otherwise compute it.

### imbed.util.match_aliases(fields, aliases)

Matches the keys of aliases to the given fields,
using the values of aliases as the matching logic (could be a list of possible
fields, a regular expression, or a custom matching function.).

A dictionary

* **Parameters:**
  * **fields** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Field`, bound= [`str`](https://docs.python.org/3/builtins/stdtypes.html#str))]) – A list of fields
  * **aliases** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Role`, bound= [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)), `Union`[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Field`, bound= [`str`](https://docs.python.org/3/builtins/stdtypes.html#str))], [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Regex`, bound= [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Field`, bound= [`str`](https://docs.python.org/3/builtins/stdtypes.html#str))]], [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Field`, bound= [`str`](https://docs.python.org/3/builtins/stdtypes.html#str))]]]]) – 

    A dictionary where:
    - Keys are roles (e.g., ‘ID’, ‘Name’) we’re looking for
    - Values are either:
      : - A list of field “aliases” (e.g., [‘id’, ‘user_id’]).
        - A string representing a regular expression (e.g., r’user.\*id’).
        - A function that takes a list of fields and returns a matched field or None.
* **Returns:**
  A dictionary mapping each role to the first matching
  : field found in `fields`, or `None` if no match is
    found. Once a column is matched, it is removed
    from further matching, so it can’t be matched again.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Role`, bound= [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)), [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Field`, bound= [`str`](https://docs.python.org/3/builtins/stdtypes.html#str))]]

Example 1: List-based aliases, regex, and custom function matching

```pycon
>>> fields = ['user_id', 'full_name', 'created_at', 'email_address']
>>> aliases = {
...     'ID': ['id', 'user_id'],  # List of possible aliases for 'ID'
...     'Name': r'.*name',  # Regular expression for 'Name'
...     'Date': lambda cols: next((col for col in cols if "date" in col.lower() or "created" in col.lower()), None)  # Custom matching function
... }
>>> match_aliases(fields, aliases)
{'ID': 'user_id', 'Name': 'full_name', 'Date': 'created_at'}
```

### Example 2: Handles conflict resolution by removing matched columns

```pycon
>>> fields = ['id', 'full_name', 'id_created', 'email_address']
>>> aliases = {
...     'Primary ID': ['id'],  # List-based alias that should match 'id' first
...     'Secondary ID': r'id.*',  # Regex to match anything starting with 'id'
...     'Email': lambda cols: next((col for col in cols if 'email' in col.lower()), None)  # Custom function for email
... }
>>> match_aliases(fields, aliases)
{'Primary ID': 'id', 'Secondary ID': 'id_created', 'Email': 'email_address'}
```

### imbed.util.merge_data(data_1, data_2, , merge_on=None, data_1_cols=None, data_2_cols=None, column_index_cursor_start=0)

Merges two sources of data, returning a dataframe.

The sources of data could be numpy arrays or pandas DataFrames.

If they’re both dataframes, the merge_on specification is needed.
If at least one of them is a numpy array, data_1 and data_2 must have the same
number of rows and merge_on is ignored, since the merge will simply be the
concatination of the two datas over the rows (that is, the result will have
that common number of rows and the number of columns will be added).

The optional data_1_cols and data_2_cols are used to transform numpy matrices into
dataframes with the given column names.

* **Parameters:**
  * **data_1** (`Union`[`ndarray`, `DataFrame`]) – The first source of data.
  * **data_2** (`Union`[`ndarray`, `DataFrame`]) – The second source of data.
  * **merge_on** – The column to merge on, if both data_1 and data_2 are dataframes.
  * **data_1_cols** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The column names for the first source of data, if it is a numpy array.
  * **data_2_cols** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The column names for the second source of data, if it is a numpy array.
* **Return type:**
  `DataFrame`

### imbed.util.planar_embeddings(kd_embeddings, \*, embeddings_func='umap', preprocess=<bound method Pipeline.fit_transform of Pipeline(steps=[('standardscaler', StandardScaler()), ('pca', PCA())])>, egress=None)

Takes a mapping of k-dimensional (kd) embeddings and returns a dict of the 2d
umap embeddings

* **Parameters:**
  * **kd_embeddings** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – a dict of kd embeddings
  * **embeddings_func** (`Union`[[`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'umap'`, `'ncvis'`, `'tsne'`, `'pca'`], [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]]]) – the function to compute the embeddings
  * **preprocessors** – a list of preprocessors to apply to the embeddings
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  a dict of the 2d umap embeddings

### Example

```pycon
>>> # Make 7 random vectors of dimension 3 (seeded, so that the example is stable)
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> kd_embeddings = {i: rng.random(3) for i in range(7)}
>>> xy_pairs = planar_embeddings(kd_embeddings, embeddings_func="pca")
>>> list(xy_pairs)
[0, 1, 2, 3, 4, 5, 6]
>>> {len(xy) for xy in xy_pairs.values()}
{2}
```

(The default, `embeddings_func="umap"`, is randomized and needs far more points
than this to be meaningful, so the example uses the deterministic “pca”.)

```pycon
>>> x, y = planar_embeddings.transpose_iterable(xy_pairs.values())
>>> len(x), len(y)
(7, 7)
```

#### TIP
Should you normalize your features (use preprocessors, the default here)?

See [https://umap-learn.readthedocs.io/en/latest/faq.html?utm_source=chatgpt.com#should-i-normalise-my-features](https://umap-learn.readthedocs.io/en/latest/faq.html?utm_source=chatgpt.com#should-i-normalise-my-features)

#### TIP
If you need to get big vectors of the x and y coordinates, you can do this:

```text
x_values, y_values = zip(*planar_embeddings(kd_embeddings).values())
```

Or even, in case you have a pandas dataframe or dict d:

```text
d['x'], d['y'] = zip(*planar_embeddings(d).values())
```

#### TIP
Use planar_embeddings.transpose_iterable to do this in a readabble way:

```text
x_values, y_values = planar_embeddings.transpose_iterable(planar_embeddings(kd_embeddings).values())
```

### imbed.util.planar_embeddings_dict_to_df(planar_embeddings_kv, , x_col='x', y_col='y', index_name='id_', key_col=None)

A function that takes a dict of planar embeddings and returns a pandas DataFrame
of the 2d embeddings

If key_col is not None, the keys are added as a column in the dataframe.

* **Parameters:**
  * **planar_embeddings_kv** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – a dict of planar embeddings
  * **x_col** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the name of the x column
  * **y_col** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the name of the y column
  * **index_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the name of the index
  * **key_col** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – if you want to add a column with the index values copied into them
* **Return type:**
  `DataFrame`
* **Returns:**
  a pandas DataFrame of the 2d embeddings

### Example

```pycon
>>> planar_embeddings_kv = {1: (0.1, 0.2), 2: (0.3, 0.4)}
>>> planar_embeddings_dict_to_df(planar_embeddings_kv)
       x    y
id_
1    0.1  0.2
2    0.3  0.4
```

### imbed.util.transpose_iterable(iterable)

This is useful to do things like:

```pycon
>>> xy_values = [(1, 2), (3, 4), (5, 6)]
>>> x_values, y_values = transpose_iterable(xy_values)
>>> x_values
(1, 3, 5)
>>> y_values
(2, 4, 6)
```

Note that transpose_iterable is an [involution](https://en.wikipedia.org/wiki/Involution_(mathematics))
(if we disregard types).

```pycon
>>> list((x_values, y_values))
[(1, 3, 5), (2, 4, 6)]
```

### imbed.util.two_d_embedding_dict_to_df(planar_embeddings_kv, , x_col='x', y_col='y', index_name='id_', key_col=None)

A function that takes a dict of planar embeddings and returns a pandas DataFrame
of the 2d embeddings

If key_col is not None, the keys are added as a column in the dataframe.

* **Parameters:**
  * **planar_embeddings_kv** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – a dict of planar embeddings
  * **x_col** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the name of the x column
  * **y_col** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the name of the y column
  * **index_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the name of the index
  * **key_col** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – if you want to add a column with the index values copied into them
* **Return type:**
  `DataFrame`
* **Returns:**
  a pandas DataFrame of the 2d embeddings

### Example

```pycon
>>> planar_embeddings_kv = {1: (0.1, 0.2), 2: (0.3, 0.4)}
>>> planar_embeddings_dict_to_df(planar_embeddings_kv)
       x    y
id_
1    0.1  0.2
2    0.3  0.4
```

### imbed.util.umap_2d_embeddings(kd_embeddings, \*, embeddings_func='umap', preprocess=<bound method Pipeline.fit_transform of Pipeline(steps=[('standardscaler', StandardScaler()), ('pca', PCA())])>, egress=None)

Takes a mapping of k-dimensional (kd) embeddings and returns a dict of the 2d
umap embeddings

* **Parameters:**
  * **kd_embeddings** – a dict of kd embeddings
  * **embeddings_func** – the function to compute the embeddings
  * **preprocessors** – a list of preprocessors to apply to the embeddings
* **Returns:**
  a dict of the 2d umap embeddings
* **Return type:**
  [*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)[KT, [*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[*float*](https://docs.python.org/3/builtins/functions.html#float), [*float*](https://docs.python.org/3/builtins/functions.html#float)]]

### Example

```pycon
>>> # Make 7 random vectors of dimension 3 (seeded, so that the example is stable)
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> kd_embeddings = {i: rng.random(3) for i in range(7)}
>>> xy_pairs = planar_embeddings(kd_embeddings, embeddings_func="pca")
>>> list(xy_pairs)
[0, 1, 2, 3, 4, 5, 6]
>>> {len(xy) for xy in xy_pairs.values()}
{2}
```

(The default, `embeddings_func="umap"`, is randomized and needs far more points
than this to be meaningful, so the example uses the deterministic “pca”.)

```pycon
>>> x, y = planar_embeddings.transpose_iterable(xy_pairs.values())
>>> len(x), len(y)
(7, 7)
```

#### TIP
Should you normalize your features (use preprocessors, the default here)?

See [https://umap-learn.readthedocs.io/en/latest/faq.html?utm_source=chatgpt.com#should-i-normalise-my-features](https://umap-learn.readthedocs.io/en/latest/faq.html?utm_source=chatgpt.com#should-i-normalise-my-features)

#### TIP
If you need to get big vectors of the x and y coordinates, you can do this:

```text
x_values, y_values = zip(*planar_embeddings(kd_embeddings).values())
```

Or even, in case you have a pandas dataframe or dict d:

```text
d['x'], d['y'] = zip(*planar_embeddings(d).values())
```

#### TIP
Use planar_embeddings.transpose_iterable to do this in a readabble way:

```text
x_values, y_values = planar_embeddings.transpose_iterable(planar_embeddings(kd_embeddings).values())
```

### imbed.util.umap_2d_embeddings_df(kd_embeddings, , x_col='x', y_col='y', index_name='id_', key_col=None)

A function that takes a mapping of kd embeddings and returns a pandas DataFrame
of the 2d umap embeddings

* **Return type:**
  `DataFrame`
