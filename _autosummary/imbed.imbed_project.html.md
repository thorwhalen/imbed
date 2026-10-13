# imbed.imbed_project

Project interface for text embedding system.

This module provides the core Project class that manages segments, embeddings,
planarizations, and clusterings with automatic invalidation and async computation
support via the au framework.

### Module Attributes

| [`COMPONENT_KINDS`](#imbed.imbed_project.COMPONENT_KINDS)                | The component kinds [`Project.compute()`](#imbed.imbed_project.Project.compute) knows how to dispatch.                                                                                            |
|---------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`ASYNC_CAPABLE_COMPONENT_KINDS`](#imbed.imbed_project.ASYNC_CAPABLE_COMPONENT_KINDS)  | The subset of [`COMPONENT_KINDS`](#imbed.imbed_project.COMPONENT_KINDS) whose work can be handed to the async (`au`) backend.                                                                     |
| [`RESULT_STORE_BY_COMPONENT_KIND`](#imbed.imbed_project.RESULT_STORE_BY_COMPONENT_KIND) | Which [`Project`](#imbed.imbed_project.Project) store each component kind's results are filed in, under the caller's `save_key`.                                                          |
| [`MissingEmbeddingsPolicy`](#imbed.imbed_project.MissingEmbeddingsPolicy)        | What [`Project.compute()`](#imbed.imbed_project.Project.compute) may do when the default planarizer/clusterer input it assembles is missing some segments (their embeddings have not landed yet). |
| [`MISSING_EMBEDDINGS_POLICIES`](#imbed.imbed_project.MISSING_EMBEDDINGS_POLICIES)    | [`MissingEmbeddingsPolicy`](#imbed.imbed_project.MissingEmbeddingsPolicy)'s members, for runtime validation and messages.                                                                                 |
| [`DFLT_ON_MISSING_EMBEDDINGS`](#imbed.imbed_project.DFLT_ON_MISSING_EMBEDDINGS)     | Default reaction to a partial default input.                                                                                                                                                             |
| [`MAX_MISSING_KEYS_SHOWN`](#imbed.imbed_project.MAX_MISSING_KEYS_SHOWN)         | How many missing segment keys an error/warning names before eliding the rest.                                                                                                                            |

### Functions

| `DFLT_GET_PROJECT_MALL`([project_id])                                                          |                                                         |
|------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| [`clear_store`](#imbed.imbed_project.clear_store)(store)                            | Clear all items in a store                              |
| [`get_local_mall`](#imbed.imbed_project.get_local_mall)([project_id, mall_keys, ...])  | Get the user stores for the package.                    |
| [`get_mall`](#imbed.imbed_project.get_mall)([project_id, get_project_mall, ...]) | Get the registry mall containing all function stores    |
| `get_ram_project_mall`([project_id])                                                           |                                                         |
| `named_partial`(func, \*args[, \_\_name_\_])                                                   |                                                         |
| [`validate_component_kind_tables`](#imbed.imbed_project.validate_component_kind_tables)()              | Assert the component-kind tables agree with each other. |
| `validate_mall_keys`(mall)                                                                     |                                                         |
| `validate_mall_kinds`()                                                                        |                                                         |

### Classes

| [`PartializedFuncs`](#imbed.imbed_project.PartializedFuncs)(store)              | A mapping that allows retrieval of functions with optional partial application.   |
|---------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| [`Project`](#imbed.imbed_project.Project)([segments, embeddings, ...]) | Central project interface - facade for all operations.                            |
| [`Projects`](#imbed.imbed_project.Projects)([store_factory])            | Container for projects with MutableMapping interface.                             |

### imbed.imbed_project.ASYNC_CAPABLE_COMPONENT_KINDS *= frozenset({'embedder'})*

The subset of [`COMPONENT_KINDS`](#imbed.imbed_project.COMPONENT_KINDS) whose work can be handed to the async (`au`)
backend. Single source of truth on purpose: [`Project.compute()`](#imbed.imbed_project.Project.compute) both *resolves*
and *validates* `async_mode` against this one name, so the two cannot drift — and
drift between them is exactly how `async_mode=True` came to be silently downgraded
to a synchronous run for the kinds the async branch did not cover.

### imbed.imbed_project.COMPONENT_KINDS *= ('embedder', 'planarizer', 'clusterer')*

The component kinds [`Project.compute()`](#imbed.imbed_project.Project.compute) knows how to dispatch. Each name
`kind` needs a `Project.{kind}s` registry attribute to resolve components from,
and a way to store its results (see [`RESULT_STORE_BY_COMPONENT_KIND`](#imbed.imbed_project.RESULT_STORE_BY_COMPONENT_KIND)).

### imbed.imbed_project.DFLT_ON_MISSING_EMBEDDINGS *: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['raise', 'warn', 'ignore']* *= 'warn'*

Default reaction to a partial default input. `"warn"` rather than `"raise"`
because “process what is ready” is a legitimate thing to want — pending embeddings
are the normal state of an async project — but it must never be what a caller gets
*without knowing*, since a result covering part of the project is indistinguishable
from one covering all of it. `"raise"` is one keyword away for all-or-nothing
callers, `"ignore"` for those who already know the input is partial.

### imbed.imbed_project.MAX_MISSING_KEYS_SHOWN *= 5*

How many missing segment keys an error/warning names before eliding the rest.

### imbed.imbed_project.MISSING_EMBEDDINGS_POLICIES *= ('raise', 'warn', 'ignore')*

[`MissingEmbeddingsPolicy`](#imbed.imbed_project.MissingEmbeddingsPolicy)’s members, for runtime validation and messages.

### imbed.imbed_project.MissingEmbeddingsPolicy

What [`Project.compute()`](#imbed.imbed_project.Project.compute) may do when the default planarizer/clusterer input it
assembles is missing some segments (their embeddings have not landed yet).

alias of [`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[‘raise’, ‘warn’, ‘ignore’]

### *class* imbed.imbed_project.PartializedFuncs(store)

Bases: [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)]

A mapping that allows retrieval of functions with optional partial application.

```pycon
>>> store = {'add': lambda x, y: x + y, 'subtract': lambda x, y: x - y}
>>> partialized_ops = PartializedFuncs(store)
```

When using a non-dict key, it will return the function directly:

```pycon
>>> func1 = partialized_ops['add']
>>> func1(2, 3)
5
```

If the key is a dictionary with one item, it will return a partial function:

```pycon
>>> func2 = partialized_ops[{'add': {'y': 3}}]
>>> func2(2)
5
```

### *class* imbed.imbed_project.Project(segments=<factory>, embeddings=<factory>, planar_coords=<factory>, cluster_indices=<factory>, embedders=<factory>, planarizers=<factory>, clusterers=<factory>, default_embedder='default', \_active_computations=<factory>, \_invalidation_cascade=True, \_auto_compute_embeddings=True, \_async_embeddings=False, \_async_base_path=None, \_id=None, \_async_backend=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Central project interface - facade for all operations.

Manages segments, embeddings, planarizations, and clusterings with
automatic computation and invalidation cascade. Supports both synchronous
and asynchronous embedding computation via the au framework.

#### add_segments(segments)

Add segments and trigger embedding computation.

* **Parameters:**
  **segments** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[`SegmentKey` ([`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`)), `Segment` ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str))]) – Mapping of segment keys to segment text
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[`SegmentKey` ([`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`))]
* **Returns:**
  List of segment keys that were added

#### cleanup_async_storage()

Clean up expired async computation results.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

#### compute(component_kind, component_key, data=None, , save_key=None, async_mode=None, on_missing_embeddings='warn')

Generic computation dispatcher.

Runs the component the caller named, on the data the caller gave, in the mode
the caller asked for — or raises saying why it cannot. It never quietly
substitutes a different component, a different mode, or a different input:
every path here returns the same kind of save key, so a silent substitution
would be indistinguishable from success.

* **Parameters:**
  * **component_kind** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Type of component; one of [`COMPONENT_KINDS`](#imbed.imbed_project.COMPONENT_KINDS).
  * **component_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Key of the component in the corresponding registry.
  * **data** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping) | [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Input data. If None, an appropriate default is used (see below).
    For an embedder it must be a `Mapping` of segment key -> segment,
    since embeddings are stored per segment key and a bare sequence carries
    no keys to store them under.
  * **save_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Key to file the results under. Generated (uniquely) if not given.
    An explicitly given key is honoured as given, and so overwrites any
    previous result stored under it.
  * **async_mode** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Whether to run asynchronously. `None` (the default) uses the
    project’s `_async_embeddings` setting, which applies only to kinds in
    [`ASYNC_CAPABLE_COMPONENT_KINDS`](#imbed.imbed_project.ASYNC_CAPABLE_COMPONENT_KINDS). Passing `True` for any other
    kind raises: those kinds have no async path, and running them
    synchronously would silently contradict an explicit request.
  * **on_missing_embeddings** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'raise'`, `'warn'`, `'ignore'`]) – What to do when the *default* input for a
    planarizer/clusterer is incomplete because some segments have no
    embedding yet — one of [`MISSING_EMBEDDINGS_POLICIES`](#imbed.imbed_project.MISSING_EMBEDDINGS_POLICIES), default
    [`DFLT_ON_MISSING_EMBEDDINGS`](#imbed.imbed_project.DFLT_ON_MISSING_EMBEDDINGS). Only consulted when `data` is
    None; when you supply `data`, its completeness is yours to decide.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  The save key. For a planarizer or clusterer this is the key its results are
  stored under, in `planar_coords` / `cluster_indices` respectively. For
  an embedder it identifies the *computation*: embedding results are merged
  into `embeddings` under their own segment keys, sync or async alike.

Default input by kind:

> - embedder: all of `segments`.
> - planarizer, clusterer: the embeddings of every segment that has one, in
>   segment order (see `on_missing_embeddings` for the ones that do not).

How results are attributed to segments:
: A component returning a mapping is taken at its word. A component returning
  a plain sequence is matched to its input by position — which requires
  knowing the input’s keys, so the sequence case is supported when `data`
  was defaulted or given as a mapping, and refused (rather than guessed) when
  `data` was given as a bare sequence.

* **Raises:**
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – unknown `component_kind`, unknown `component_key`, async
        requested for a kind that cannot do it, an unknown
        `on_missing_embeddings`, or results that cannot be attributed to
        segments.
  * [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – embedder `data` that is not a Mapping.
* **Return type:**
  [*str*](https://docs.python.org/3/builtins/stdtypes.html#str)

#### *property* embedding_status *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [int](https://docs.python.org/3/builtins/functions.html#int)]*

Get counts of embedding statuses.

Returns counts of: present, missing, computing

#### get_computation_status(computation_id)

Get status of a tracked async computation.

* **Return type:**
  `ComputationStatus` | [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### get_embeddings(segment_keys=None)

Get embeddings for specified segments (or all if None)

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]

#### list_active_computations()

List IDs of active async computations.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### set_async_mode(enabled)

Enable or disable async embedding computation.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *property* valid_embeddings *: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[KT, [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[float](https://docs.python.org/3/builtins/functions.html#float)]]*

Get all available computed embeddings

#### wait_for_embeddings(segment_keys=None, timeout=10.0, poll_interval=0.1)

Wait for embeddings to be available.

This works for both sync and async modes - in sync mode, embeddings
are immediately available; in async mode, we poll until they appear.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### *class* imbed.imbed_project.Projects(store_factory=<class 'dict'>)

Bases: [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Project`](#imbed.imbed_project.Project)]

Container for projects with MutableMapping interface.

```pycon
>>> projects = Projects()
>>> p = Project(_id='test', segments={}, embeddings={},
...             planar_coords={}, cluster_indices={},
...             embedders={}, planarizers={}, clusterers={})
>>> projects["test"] = p
>>> list(projects)
['test']
>>> projects["test"]._id
'test'
```

#### append(project)

Append a project to the collection.

* **Parameters:**
  **project** ([`Project`](#imbed.imbed_project.Project)) – Project instance to add
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### create_project(\*, project_id=None, segments_store_factory=<class 'dict'>, embeddings_store_factory=<class 'dict'>, planar_store_factory=<class 'dict'>, cluster_store_factory=<class 'dict'>, embedders=None, planarizers=None, clusterers=None, async_embeddings=True, async_base_path=None, async_backend=None, overwrite=False)

Create and add a new project.

* **Parameters:**
  * **project_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – ID for the new project (optional)
  * **\*_store_factory** – Factory functions for various stores
  * **embedders** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Component registry for embedders
  * **planarizers** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Component registry for planarizers
  * **clusterers** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Component registry for clusterers
  * **async_embeddings** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to use async embedding computation
  * **async_base_path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Base path for au async computation storage
  * **async_backend** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Backend for async computation (StdLibQueueBackend, RQ, etc)
  * **overwrite** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, replace any existing project with the same id
* **Return type:**
  [`Project`](#imbed.imbed_project.Project)
* **Returns:**
  The created Project instance

### imbed.imbed_project.RESULT_STORE_BY_COMPONENT_KIND *= {'clusterer': 'cluster_indices', 'planarizer': 'planar_coords'}*

Which [`Project`](#imbed.imbed_project.Project) store each component kind’s results are filed in, under the
caller’s `save_key`. `"embedder"` is deliberately absent: embeddings are merged
into `Project.embeddings` under their *segment* keys, so an embedder’s
`save_key` names the computation rather than a stored value.

### imbed.imbed_project.clear_store(store)

Clear all items in a store

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### imbed.imbed_project.get_local_mall(project_id='default_project', \*, mall_keys=('misc', 'segments', 'embeddings', 'clusters', 'planar_embeddings', 'statuses', 'cluster_labels'), default_store_maker=functools.partial(<function mk_blob_store_for_path>, path_to_bytes_store=<class 'dol.filesys.Files'>, base_store_wrap=functools.partial(<function extension_based_wrap>, extension_to_decoder={'csv': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'txt': <method 'decode' of 'bytes' objects>, 'tsv': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, sep='\\\\t', index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'parquet': functools.partial(<function read_from_bytes>, <function read_parquet>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'json': <function loads>, 'html': functools.partial(<function read_from_bytes>, functools.partial(<function read_html>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'p': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'pickle': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'pkl': <built-in function loads>, 'xml': functools.partial(<function read_from_bytes>, <function read_xml>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sql': functools.partial(<function read_from_bytes>, <function read_sql>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sqlite': functools.partial(<function read_from_bytes>, <function read_sql>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'feather': functools.partial(<function read_from_bytes>, <function read_feather>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'stata': functools.partial(<function read_from_bytes>, functools.partial(<function read_stata>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'dta': functools.partial(<function read_from_bytes>, functools.partial(<function read_stata>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sas': functools.partial(<function read_from_bytes>, <function read_sas>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'h5': functools.partial(<function read_from_bytes>, <function read_hdf>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'hdf5': functools.partial(<function read_from_bytes>, <function read_hdf>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xls': functools.partial(<function read_from_bytes>, functools.partial(<function read_excel>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xlsx': functools.partial(<function read_from_bytes>, functools.partial(<function read_excel>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'orc': functools.partial(<function read_from_bytes>, <function read_orc>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sav': functools.partial(<function read_from_bytes>, <function read_spss>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'mat': functools.partial(<function read_from_bytes>, <function mat_bytes_to_dataframe>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'arrow': functools.partial(<function read_from_bytes>, <function arrow_bytes_to_dataframe>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), '': <function loads>, 'dill': <function loads>}, extension_to_encoder={'csv': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_csv>, index=False), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'txt': <method 'encode' of 'str' objects>, 'tsv': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_csv>, index=False, sep='\\\\t', escapechar='\\\\\\\\', quotechar='"'), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'json': <dol.util.Pipe object>, 'html': functools.partial(<function written_bytes>, functools.partial(<function DataFrame.to_html>, index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'p': functools.partial(<function written_bytes>, <function NDFrame.to_pickle>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'pickle': functools.partial(<function written_bytes>, <function NDFrame.to_pickle>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'pkl': <built-in function dumps>, 'npy': functools.partial(<function written_bytes>, <function save>, obj_arg_position_in_writer=1, io_buffer_cls=<class '_io.BytesIO'>), 'zip': <function save_df_to_zipped_tsv>, 'h5': functools.partial(<function written_bytes>, <function NDFrame.to_hdf>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'hdf5': functools.partial(<function written_bytes>, <function NDFrame.to_hdf>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'stata': functools.partial(<function written_bytes>, functools.partial(<function DataFrame.to_stata>, write_index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'dta': functools.partial(<function written_bytes>, functools.partial(<function DataFrame.to_stata>, write_index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'sql': functools.partial(<function written_bytes>, <function NDFrame.to_sql>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'sqlite': functools.partial(<function written_bytes>, <function NDFrame.to_sql>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'gbq': functools.partial(<function written_bytes>, <function \_to_gbq_unavailable>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'xls': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_excel>, index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'xlsx': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_excel>, index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'xml': functools.partial(<function written_bytes>, <function DataFrame.to_xml>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'parquet': functools.partial(<function written_bytes>, <function cast_to_parquet>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'feather': functools.partial(<function written_bytes>, <function DataFrame.to_feather>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'orc': functools.partial(<function written_bytes>, <function DataFrame.to_orc>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'mat': functools.partial(<function written_bytes>, <function dataframe_to_mat_bytes>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'arrow': functools.partial(<function written_bytes>, <function dataframe_to_arrow_bytes>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), '': <function dumps>, 'dill': <function dumps>}), filename_suffix='.dill'))

Get the user stores for the package.

* **Returns:**
  A dictionary containing paths to various user stores.
* **Return type:**
  [*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)

### imbed.imbed_project.get_mall(project_id='default_project', \*, get_project_mall=<function get_ram_project_mall>, include_signature_stores=True)

Get the registry mall containing all function stores

* **Return type:**
  [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)]]
* **Returns:**
  A dictionary of stores, each containing registered processing functions

### imbed.imbed_project.validate_component_kind_tables()

Assert the component-kind tables agree with each other.

They are read in different places — registry lookup, async gating, result storage —
and a kind present in one but absent from another is precisely the failure mode
this module keeps hitting: a request that resolves fine and is then dropped.
