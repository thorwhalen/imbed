"""Utils for components"""

import os


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
