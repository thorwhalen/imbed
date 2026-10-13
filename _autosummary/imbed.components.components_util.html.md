# imbed.components.components_util

Utils for components

### Functions

| [`add_default_key`](#imbed.components.components_util.add_default_key)(d, default_key[, ...])   | Add a 'default' key to a registry pointing to a specified component.   |
|-------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| [`get_component_store`](#imbed.components.components_util.get_component_store)(component)           | Get the store for a specific component type                            |
| [`get_standard_components`](#imbed.components.components_util.get_standard_components)([refresh])       | Get the standard components for the project.                           |

### Classes

| [`ComponentRegistry`](#imbed.components.components_util.ComponentRegistry)(name, \*[, factory])   | A dict-based registry with decorator support for registration.   |
|-------------------------------------------------------------------------------------------|------------------------------------------------------------------|

### *class* imbed.components.components_util.ComponentRegistry(name, , factory=None)

Bases: [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

A dict-based registry with decorator support for registration.

This registry provides:

- Decorator-based registration with @registry.register()
- Custom key support with @registry.register(‘custom_key’)
- Direct registration via registry.register_item(key, item)
- Optional factory functions to wrap registered items
- Full dict compatibility for backward compatibility

### Examples

```pycon
>>> registry = ComponentRegistry('my_components')
>>> @registry.register()
... def my_function(x):
...     return x * 2
>>> registry['my_function'](5)
10
>>> @registry.register('custom_name')
... def another_function(x):
...     return x + 1
>>> registry['custom_name'](5)
6
```

#### register(key=None, \*\*config)

Decorator to register a component.

* **Parameters:**
  * **key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Registration key (defaults to function/class name)
  * **\*\*config** – Additional configuration passed to factory if present
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`T`)], [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`T`)]
* **Returns:**
  Decorator function that registers and returns the component

### Examples

```pycon
>>> registry = ComponentRegistry('test')
>>> @registry.register()
... def my_func():
...     return 42
>>> registry['my_func']()
42
>>> @registry.register('custom')
... def other_func():
...     return 99
>>> registry['custom']()
99
```

#### register_item(key, item, \*\*config)

Register an item directly (non-decorator usage).

* **Parameters:**
  * **key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Registration key
  * **item** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Item to register
  * **\*\*config** – Additional configuration passed to factory if present

### Examples

```pycon
>>> registry = ComponentRegistry('test')
>>> def my_func():
...     return 42
>>> registry.register_item('my_key', my_func)
>>> registry['my_key']()
42
```

### imbed.components.components_util.add_default_key(d, default_key, enviornment_var=None)

Add a ‘default’ key to a registry pointing to a specified component.

* **Parameters:**
  * **d** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – The registry dict to add default to
  * **default_key** – Either a string key or a callable (uses \_\_name_\_ as key)
  * **enviornment_var** – Optional environment variable to override default_key
* **Returns:**
  The modified dict
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If the default_key is not found in the registry

### imbed.components.components_util.get_component_store(component)

Get the store for a specific component type

### imbed.components.components_util.get_standard_components(refresh=False)

Get the standard components for the project.

* **Returns:**
  A dictionary of standard components, each containing registered processing functions
