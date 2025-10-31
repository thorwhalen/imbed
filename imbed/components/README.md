# Components Module

The `imbed.components` module provides a unified, lazy-loading interface for registering and accessing different types of components used in embedding workflows.

## Overview

The module is organized around four main component types:
- **segmenters**: Text segmentation functions
- **embedders**: Vectorization/embedding functions
- **planarizers**: 2D projection functions for visualization
- **clusterers**: Clustering algorithms

## Features

### 1. Lazy Loading

Components are loaded only when first accessed, minimizing import time:

```python
# Fast import - nothing is loaded yet
from imbed.components import segmenters

# Components load on first access
list(segmenters.keys())  # Now segmentation module is loaded
```

### 2. Multiple Access Patterns

**Direct import:**
```python
from imbed.components import segmenters, embedders
```

**Via components object (attribute access):**
```python
from imbed.components import components

segmenters = components.segmenters  # Lazy load
embedders = components.embedders    # Lazy load

# Visual feedback on what's loaded
print(components)  # <LazyComponents(segmenters=✓, embedders=✓, planarizers=○, clusterers=○)>
```

**Via components object (dict-like access):**
```python
from imbed.components import components

# Access like a dictionary
segmenters = components['segmenters']
planarizers = components['planarizers']

# Iterate over component types
for name in components:
    print(name)  # 'segmenters', 'embedders', 'planarizers', 'clusterers'

# Check membership
if 'embedders' in components:
    print('Embedders available!')

# Use mapping methods
print(f"Available: {list(components.keys())}")
print(f"Total types: {len(components)}")
```

The `components` object is a `Mapping`, supporting all standard dict operations while maintaining lazy loading.

### 3. ComponentRegistry

All component collections use `ComponentRegistry`, which provides:
- Dict-like interface (fully backward compatible)
- Decorator-based registration
- Custom key support
- Optional factory functions for wrapping components

**Basic registration:**
```python
from imbed.components import segmenters

@segmenters.register()
def my_segmenter(text):
    """Split text by commas"""
    return text.split(',')

# Access it
result = segmenters['my_segmenter']("a,b,c")
```

**Custom key:**
```python
@segmenters.register('comma_splitter')
def another_segmenter(text):
    return text.split(',')

result = segmenters['comma_splitter']("x,y,z")
```

**Direct registration:**
```python
def my_function(text):
    return text.split()

segmenters.register_item('word_splitter', my_function)
```

### 4. Factory Pattern Support

You can provide a factory function to wrap registered components:

```python
from imbed.components.components_util import ComponentRegistry

def add_metadata_factory(func, **metadata):
    """Factory that adds metadata to components"""
    func._metadata = metadata
    return func

registry = ComponentRegistry('my_components', factory=add_metadata_factory)

@registry.register(author='Thor', version='1.0')
def process_data(data):
    return data.upper()

# Metadata is attached
print(registry['process_data']._metadata)  # {'author': 'Thor', 'version': '1.0'}
```

### 5. Default Components

Each registry has a 'default' key pointing to a recommended component:

```python
from imbed.components import segmenters, embedders

# Use default components
default_segmenter = segmenters['default']
default_embedder = embedders['default']
```

## Architecture

### ComponentRegistry Class

Located in `components_util.py`, `ComponentRegistry` is a dict subclass with:
- `register()`: Decorator for registration
- `register_item()`: Direct registration method
- `name`: Registry identifier
- Optional `factory`: Function to wrap registered items

### LazyComponents Mapping

The `components` singleton is a `Mapping` that provides:
- **Lazy loading**: Imports only when accessed
- **Dual access patterns**: Both `components.segmenters` and `components['segmenters']`
- **Standard mapping operations**: `len()`, `in`, iteration, `.keys()`, `.values()`, `.items()`
- **Visual feedback**: `repr()` shows which components are loaded (✓) or not (○)
- **Caching**: Once loaded, returns same object for subsequent accesses

Example use cases:
```python
from imbed.components import components

# Iteration for discovery
for component_type in components:
    registry = components[component_type]
    print(f"{component_type}: {len(registry)} items")

# Dynamic access
component_name = 'embedders'  # Could come from config
registry = components[component_name]

# Safe membership testing
if 'planarizers' in components:
    planarizers = components['planarizers']

# Functional operations
total = sum(len(reg) for reg in components.values())
```

### Lazy Loading Mechanism

The `__init__.py` uses Python's `__getattr__` to defer imports:

```python
def __getattr__(name: str):
    """Load component registry on first access"""
    if name in _module_map:
        # Import and cache the registry
        ...
```

This ensures:
- Fast initial imports
- On-demand loading of heavy dependencies
- Caching for subsequent access

## Backward Compatibility

The refactoring maintains full backward compatibility:
- Registries are still dicts
- All existing keys and values work unchanged
- Registration patterns remain compatible
- Default keys continue to function

## Testing

Comprehensive tests are in `tests/test_components.py`:

```bash
pytest tests/test_components.py -v
```

Tests cover:
- ComponentRegistry functionality
- Lazy loading behavior
- Registration patterns
- Actual component functionality
- Backward compatibility

## Migration Guide

For users of older versions:

**Old pattern (still works):**
```python
from imbed.components.segmentation import segmenters

# Direct dict access
my_segmenter = segmenters['string_lines']
```

**New pattern (recommended):**
```python
from imbed.components import segmenters

# Use decorator for registration
@segmenters.register()
def my_custom_segmenter(text):
    return text.split()
```

**Benefits of new pattern:**
- Lazy loading for faster imports
- Consistent registration across all component types
- Factory support for advanced use cases
- Better introspection (ComponentRegistry has useful repr)

## Examples

### Creating a Custom Segmenter

```python
from imbed.components import segmenters

@segmenters.register('sentence_splitter')
def split_sentences(text):
    """Split text into sentences"""
    import re
    return re.split(r'[.!?]+', text)

# Use it
segments = list(segmenters['sentence_splitter']("Hello! How are you? I'm fine."))
```

### Creating a Custom Embedder

```python
from imbed.components import embedders

@embedders.register('simple_counter')
def count_vectorizer(texts):
    """Simple character count vectorizer"""
    if isinstance(texts, dict):
        return {k: [len(v)] for k, v in texts.items()}
    return [[len(t)] for t in texts]

# Use it
vectors = embedders['simple_counter'](["hello", "world"])
```

### Using Factory for Validation

```python
from imbed.components.components_util import ComponentRegistry

def validate_factory(func, **config):
    """Factory that validates component signatures"""
    import inspect
    sig = inspect.signature(func)
    
    def wrapped(*args, **kwargs):
        # Add validation logic here
        return func(*args, **kwargs)
    
    wrapped.__wrapped__ = func
    wrapped.__config__ = config
    return wrapped

validated_registry = ComponentRegistry('validated', factory=validate_factory)

@validated_registry.register(requires_text=True)
def my_processor(text):
    return text.upper()
```

## Performance

Lazy loading provides significant benefits:

- **Before**: Importing `imbed.components` loaded all 4 modules and their dependencies
- **After**: Only loads what you use, when you use it

Example impact:
```python
# Old: ~2-3 seconds for all heavy deps
from imbed.components.vectorization import embedders

# New: ~0.1 seconds until you access embedders
from imbed.components import embedders  # Fast!
list(embedders.keys())  # Loads only now
```

## Contributing

To add a new component type:

1. Create the module with a `ComponentRegistry`
2. Register components using `@registry.register()`
3. Add lazy loading entry in `__init__.py`
4. Add tests in `tests/test_components.py`

Example:
```python
# my_new_component.py
from imbed.components.components_util import ComponentRegistry

my_components = ComponentRegistry('my_components')

@my_components.register()
def my_function():
    pass
```
