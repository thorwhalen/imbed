"""Utils for components"""

import pickle
from functools import lru_cache, wraps
import os
from typing import Callable, TypeVar, Any
from imbed.util import pkg_files

T = TypeVar('T')

component_files = pkg_files.joinpath("components")
standard_components_file = component_files.joinpath("standard_components.pickle")


class ComponentRegistry(dict):
    """A dict-based registry with decorator support for registration.

    This registry provides:
    - Decorator-based registration with @registry.register()
    - Custom key support with @registry.register('custom_key')
    - Direct registration via registry.register_item(key, item)
    - Optional factory functions to wrap registered items
    - Full dict compatibility for backward compatibility

    Examples:
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
    """

    def __init__(self, name: str, *, factory: Callable | None = None):
        """Initialize registry.

        Args:
            name: Name of the registry (for error messages and repr)
            factory: Optional factory function to wrap registered items.
                     Signature: factory(item, **config) -> wrapped_item
        """
        super().__init__()
        self.name = name
        self._factory = factory

    def register(self, key: str = None, **config) -> Callable[[T], T]:
        """Decorator to register a component.

        Args:
            key: Registration key (defaults to function/class name)
            **config: Additional configuration passed to factory if present

        Returns:
            Decorator function that registers and returns the component

        Examples:
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
        """

        def decorator(func: T) -> T:
            nonlocal key
            if key is None:
                # Use function/class name as key
                key = getattr(func, '__name__', str(func))

            # Apply factory if present
            item = self._factory(func, **config) if self._factory else func
            self[key] = item

            # Return original function for continued use
            return func

        return decorator

    def register_item(self, key: str, item: Any, **config):
        """Register an item directly (non-decorator usage).

        Args:
            key: Registration key
            item: Item to register
            **config: Additional configuration passed to factory if present

        Examples:
            >>> registry = ComponentRegistry('test')
            >>> def my_func():
            ...     return 42
            >>> registry.register_item('my_key', my_func)
            >>> registry['my_key']()
            42
        """
        self[key] = self._factory(item, **config) if self._factory else item

    def __repr__(self):
        """Provide useful repr showing registry name and contents."""
        items_preview = list(self.keys())[:5]  # Show first 5 items
        items_str = ', '.join(items_preview)
        if len(self) > 5:
            items_str += f', ... ({len(self)} total)'
        return f"{self.__class__.__name__}({self.name!r}, items=[{items_str}])"


def add_default_key(d: dict, default_key, enviornment_var=None):
    """Add a 'default' key to a registry pointing to a specified component.

    Args:
        d: The registry dict to add default to
        default_key: Either a string key or a callable (uses __name__ as key)
        enviornment_var: Optional environment variable to override default_key

    Returns:
        The modified dict

    Raises:
        ValueError: If the default_key is not found in the registry
    """
    if enviornment_var:
        default_key = os.getenv(enviornment_var, default_key)
    if not isinstance(default_key, str):
        assert callable(default_key), "default_key must be a string or callable"
        func = default_key
        default_key = func.__name__
    if default_key not in d:
        raise ValueError(f"Default key {default_key} not found in dictionary")
    d["default"] = d[default_key]
    return d


_component_kinds = ("segmenters", "embedders", "clusterers", "planarizers")
component_store_names = _component_kinds


def get_component_store(component: str):
    """Get the store for a specific component type"""
    if component == "segmenters":
        from imbed.components.segmentation import segmenters as component_store
    elif component == "embedders":
        from imbed.components.vectorization import embedders as component_store
    elif component == "clusterers":
        from imbed.components.clusterization import clusterers as component_store
    elif component == "planarizers":
        from imbed.components.planarization import planarizers as component_store
    else:
        raise ValueError(f"Unknown component type: {component}")
    return component_store.copy()


def _get_standard_components_from_modules():
    return {kind: get_component_store(kind) for kind in _component_kinds}


def _get_standard_components_from_file(refresh=False):
    """Load the standard components from a pickle file."""
    if refresh or not standard_components_file.exists():
        components = _get_standard_components_from_modules()
        standard_components_file.write_bytes(pickle.dumps(components))
    return pickle.loads(standard_components_file.read_bytes())


@lru_cache
def get_standard_components(refresh=False):
    """Get the standard components for the project.

    Returns:
        A dictionary of standard components, each containing registered processing functions
    """
    return _get_standard_components_from_modules()
    return _get_standard_components_from_file(refresh=refresh)
