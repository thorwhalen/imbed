# imbed.base

Base functionality of imbded.

### Functions

| `add_token_info_to_df`(df, segments_col, \*[, ...])                           |                                                    |
|-------------------------------------------------------------------------------|----------------------------------------------------|
| [`batches`](#imbed.base.batches)(df[, chk_size])      | Yield batches of rows from a DataFrame.            |
| `compute_and_save_embeddings`(df[, ...])                                      |                                                    |
| `compute_and_save_planar_embeddings`(...[, ...])                              |                                                    |
| [`get_empty_temporary_folder`](#imbed.base.get_empty_temporary_folder)() | Returns the path of a new, empty temporary folder. |
| `identity`(x)                                                                 |                                                    |
| `mk_local_store`(rootdir)                                                     |                                                    |

### Classes

| [`ComputedValuesMapping`](#imbed.base.ComputedValuesMapping)(keys_factory[, ...])    | A mapping that returns empty values for all keys.   |
|------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| [`HugfaceDaccBase`](#imbed.base.HugfaceDaccBase)(huggingface_data_stub[, ...]) |                                                     |
| `ImbedDaccBase`()                                                                              |                                                     |
| `LocalSavesMixin`()                                                                            |                                                     |

### *class* imbed.base.ComputedValuesMapping(keys_factory, value_of_key=functools.partial(<function identity>, None))

Bases: `KvReader`, [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

A mapping that returns empty values for all keys.

Example usage:

```pycon
>>> m = ComputedValuesMapping(('apple', 'crumble'), value_of_key=len)
>>> list(m)
['apple', 'crumble']
>>> m['apple']
5
```

### *class* imbed.base.HugfaceDaccBase(huggingface_data_stub, name=None, , saves_dir=None, root_saves_dir='/home/runner/.config/imbed/saves', verbose=1)

Bases: `LocalSavesMixin`

### imbed.base.batches(df, chk_size=1000)

Yield batches of rows from a DataFrame.

The yielded batches are lists of (index, row) tuples.

If chk_size is None, yield the whole DataFrame as a single batch.

### imbed.base.get_empty_temporary_folder()

Returns the path of a new, empty temporary folder.
