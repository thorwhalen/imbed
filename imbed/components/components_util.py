"""Utils for components"""

import pickle
from functools import lru_cache
import os
from imbed.util import pkg_files

component_files = pkg_files.joinpath("components")
standard_components_file = component_files.joinpath("standard_components.pickle")


def add_default_key(d: dict, default_key, enviornment_var=None):
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
