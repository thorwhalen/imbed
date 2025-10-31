"""Components for imbed applications.

This module provides lazy-loading access to component registries:
- segmenters: Text segmentation functions
- embedders: Vectorization/embedding functions
- planarizers: 2D projection functions for visualization
- clusterers: Clustering algorithms

Components are loaded on first access to minimize import time.

Examples:
    >>> from imbed.components import segmenters
    >>> list(segmenters.keys())[:3]  # doctest: +SKIP
    ['string_lines', 'jdict_to_segments', 'field_values']

    >>> from imbed.components import components
    >>> embedders = components.embedders  # Lazy load via attribute
    >>> planarizers = components['planarizers']  # Lazy load via dict access
    >>> 'constant_vectorizer' in embedders
    True
    >>> 'segmenters' in components  # Check if component type exists
    True
    >>> list(components)  # Iterate over component names
    ['segmenters', 'embedders', 'planarizers', 'clusterers']
"""


def __getattr__(name: str):
    """Lazy-load component registries on first access.

    This enables fast imports while deferring heavy dependencies until needed.

    Args:
        name: Attribute name to load

    Returns:
        The requested component registry

    Raises:
        AttributeError: If the requested component doesn't exist
    """
    _module_map = {
        'segmenters': ('imbed.components.segmentation', 'segmenters'),
        'embedders': ('imbed.components.vectorization', 'embedders'),
        'planarizers': ('imbed.components.planarization', 'planarizers'),
        'clusterers': ('imbed.components.clusterization', 'clusterers'),
    }

    if name in _module_map:
        from importlib import import_module

        module_path, attr_name = _module_map[name]
        module = import_module(module_path)
        value = getattr(module, attr_name)
        # Cache in module namespace for subsequent access
        globals()[name] = value
        return value

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    """Support for tab completion and dir()."""
    return [
        'segmenters',
        'embedders',
        'planarizers',
        'clusterers',
        'components',
        'ComponentRegistry',
    ]


from collections.abc import Mapping
from imbed.components.components_util import ComponentRegistry


class _LazyComponents(Mapping):
    """Lazy-loading container for component registries.

    Provides multiple access patterns:
        from imbed.components import components

        # Attribute access
        segmenters = components.segmenters

        # Dict-like access
        segmenters = components['segmenters']

        # Iteration
        for name in components:
            print(name)

        # Membership testing
        if 'segmenters' in components:
            ...

        # Length
        len(components)  # Number of component types

    Each access triggers lazy loading only when needed.
    """

    _registry_specs = {
        'segmenters': ('imbed.components.segmentation', 'segmenters'),
        'embedders': ('imbed.components.vectorization', 'embedders'),
        'planarizers': ('imbed.components.planarization', 'planarizers'),
        'clusterers': ('imbed.components.clusterization', 'clusterers'),
    }

    def __init__(self):
        self._cache = {}

    def _load_registry(self, name: str):
        """Load a registry if not already cached.

        Args:
            name: Name of the registry to load

        Returns:
            The loaded registry

        Raises:
            KeyError: If the registry name is not recognized
        """
        # Return from cache if already loaded
        if name in self._cache:
            return self._cache[name]

        # Load and cache if it's a known registry
        if name in self._registry_specs:
            from importlib import import_module

            module_path, attr_name = self._registry_specs[name]
            module = import_module(module_path)
            value = getattr(module, attr_name)
            self._cache[name] = value
            return value

        raise KeyError(f"Unknown component type: {name!r}")

    def __getattr__(self, name: str):
        """Load component registry on attribute access."""
        if name.startswith('_'):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        try:
            return self._load_registry(name)
        except KeyError as e:
            raise AttributeError(str(e)) from e

    def __getitem__(self, name: str):
        """Load component registry on dict-like access.

        Args:
            name: Name of the registry

        Returns:
            The requested registry

        Raises:
            KeyError: If the registry name is not recognized
        """
        return self._load_registry(name)

    def __iter__(self):
        """Iterate over component registry names."""
        return iter(self._registry_specs)

    def __len__(self):
        """Return the number of component types."""
        return len(self._registry_specs)

    def __contains__(self, name):
        """Check if a component type exists."""
        return name in self._registry_specs

    def __dir__(self):
        """Show available component registries."""
        return list(self._registry_specs.keys())

    def __repr__(self):
        """Show which registries are loaded."""
        loaded = ', '.join(
            f"{k}={'✓' if k in self._cache else '○'}" for k in self._registry_specs
        )
        return f"<LazyComponents({loaded})>"


# Create singleton instance
components = _LazyComponents()
