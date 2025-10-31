"""
Tests for the components module refactoring.

Tests cover:
- ComponentRegistry initialization and registration
- Lazy loading mechanism
- Registration validation
- Factory functions
- The unified components interface
"""

import pytest
from typing import Callable


# Test ComponentRegistry
class TestComponentRegistry:
    """Tests for the ComponentRegistry class."""

    def test_registry_creation(self):
        """Registry can be created with a name."""
        from imbed.components.components_util import ComponentRegistry

        registry = ComponentRegistry('test_components')
        assert registry.name == 'test_components'
        assert len(registry) == 0
        assert isinstance(registry, dict)

    def test_component_registry_importable_from_components(self):
        """ComponentRegistry can be imported from imbed.components."""
        from imbed.components import ComponentRegistry

        # Should be the same class
        from imbed.components.components_util import ComponentRegistry as DirectImport

        assert ComponentRegistry is DirectImport

        # Should work to create registries
        registry = ComponentRegistry('test_import')
        assert registry.name == 'test_import'

    def test_register_decorator_with_default_key(self):
        """Decorator registers function with its name as key."""
        from imbed.components.components_util import ComponentRegistry

        registry = ComponentRegistry('test_components')

        @registry.register()
        def my_function(x):
            return x * 2

        assert 'my_function' in registry
        assert registry['my_function'](5) == 10
        # Function still works as normal
        assert my_function(5) == 10

    def test_register_decorator_with_custom_key(self):
        """Decorator can register with a custom key."""
        from imbed.components.components_util import ComponentRegistry

        registry = ComponentRegistry('test_components')

        @registry.register('custom_name')
        def my_function(x):
            return x * 2

        assert 'custom_name' in registry
        assert registry['custom_name'](5) == 10

    def test_register_item_directly(self):
        """Items can be registered without decorator."""
        from imbed.components.components_util import ComponentRegistry

        registry = ComponentRegistry('test_components')

        def my_function(x):
            return x * 3

        registry.register_item('my_key', my_function)
        assert 'my_key' in registry
        assert registry['my_key'](5) == 15

    def test_registry_with_factory(self):
        """Registry can use a factory to wrap registered items."""
        from imbed.components.components_util import ComponentRegistry

        def wrapper_factory(func, **config):
            """Factory that adds metadata to functions."""

            def wrapped(*args, **kwargs):
                result = func(*args, **kwargs)
                return {'result': result, 'config': config}

            wrapped.__wrapped__ = func
            return wrapped

        registry = ComponentRegistry('test_components', factory=wrapper_factory)

        @registry.register(scale=2)
        def my_function(x):
            return x * 2

        result = registry['my_function'](5)
        assert result['result'] == 10
        assert result['config'] == {'scale': 2}

    def test_registry_repr(self):
        """Registry has a useful repr."""
        from imbed.components.components_util import ComponentRegistry

        registry = ComponentRegistry('my_registry')
        registry.register_item('item1', lambda: 1)
        registry.register_item('item2', lambda: 2)

        repr_str = repr(registry)
        assert 'my_registry' in repr_str
        assert 'item1' in repr_str
        assert 'item2' in repr_str


# Test lazy loading
class TestLazyLoading:
    """Tests for lazy loading mechanism."""

    def test_lazy_loading_via_getattr(self):
        """Components load lazily when accessed via module getattr."""
        # This test imports components for the first time
        from imbed.components import segmenters

        assert isinstance(segmenters, dict)
        assert len(segmenters) > 0

    def test_lazy_loading_from_statement(self):
        """Components can be imported normally."""
        from imbed.components import embedders

        assert isinstance(embedders, dict)
        assert len(embedders) > 0

    def test_lazy_loading_multiple_components(self):
        """Multiple components can be loaded."""
        from imbed.components import segmenters, embedders, planarizers, clusterers

        assert isinstance(segmenters, dict)
        assert isinstance(embedders, dict)
        assert isinstance(planarizers, dict)
        assert isinstance(clusterers, dict)

    def test_lazy_loading_via_components_object(self):
        """Components can be accessed via components object."""
        from imbed.components import components

        # Access should trigger lazy loading
        segmenters = components.segmenters
        assert isinstance(segmenters, dict)

        embedders = components.embedders
        assert isinstance(embedders, dict)

    def test_nonexistent_component_raises(self):
        """Accessing nonexistent component raises ImportError."""
        with pytest.raises(ImportError):
            from imbed.components import nonexistent_component

    def test_dir_includes_components(self):
        """dir() on components module shows available components."""
        import imbed.components

        component_names = dir(imbed.components)
        assert 'segmenters' in component_names
        assert 'embedders' in component_names
        assert 'planarizers' in component_names
        assert 'clusterers' in component_names


# Test LazyComponents as a Mapping
class TestLazyComponentsMapping:
    """Tests for LazyComponents Mapping interface."""

    def test_dict_like_access(self):
        """LazyComponents supports dict-like access."""
        from imbed.components import components

        # Access via __getitem__
        segmenters = components['segmenters']
        assert isinstance(segmenters, dict)
        assert len(segmenters) > 0

    def test_iteration(self):
        """Can iterate over component names."""
        from imbed.components import components

        names = list(components)
        assert 'segmenters' in names
        assert 'embedders' in names
        assert 'planarizers' in names
        assert 'clusterers' in names
        assert len(names) == 4

    def test_length(self):
        """len() returns number of component types."""
        from imbed.components import components

        assert len(components) == 4

    def test_membership(self):
        """'in' operator works for checking component types."""
        from imbed.components import components

        assert 'segmenters' in components
        assert 'embedders' in components
        assert 'planarizers' in components
        assert 'clusterers' in components
        assert 'nonexistent' not in components

    def test_keys_values_items(self):
        """Mapping methods work (keys, values, items)."""
        from imbed.components import components

        # keys()
        keys = list(components.keys())
        assert len(keys) == 4
        assert 'segmenters' in keys

        # values() - triggers loading
        values = list(components.values())
        assert len(values) == 4
        assert all(isinstance(v, dict) for v in values)

        # items()
        items = list(components.items())
        assert len(items) == 4
        assert all(
            isinstance(name, str) and isinstance(reg, dict) for name, reg in items
        )

    def test_getitem_raises_keyerror(self):
        """Accessing nonexistent component via [] raises KeyError."""
        from imbed.components import components

        with pytest.raises(KeyError):
            _ = components['nonexistent_component']

    def test_attribute_and_dict_access_same_object(self):
        """Attribute and dict access return the same cached object."""
        from imbed.components import components

        # Access via attribute
        seg1 = components.segmenters
        # Access via dict
        seg2 = components['segmenters']

        # Should be the exact same object (cached)
        assert seg1 is seg2

    def test_for_loop_iteration(self):
        """Can use in for loop."""
        from imbed.components import components

        component_types = []
        for name in components:
            component_types.append(name)

        assert len(component_types) == 4
        assert 'segmenters' in component_types


# Test actual component registries
class TestComponentRegistries:
    """Tests for the actual component registries."""

    def test_segmenters_registry(self):
        """Segmenters registry works correctly."""
        from imbed.components import segmenters
        from imbed.components.components_util import ComponentRegistry

        assert isinstance(segmenters, ComponentRegistry)
        assert segmenters.name == 'segmenters'

        # Check that known segmenters are registered
        assert 'string_lines' in segmenters
        assert 'jdict_to_segments' in segmenters

        # Default should be set
        assert 'default' in segmenters

    def test_embedders_registry(self):
        """Embedders registry works correctly."""
        from imbed.components import embedders
        from imbed.components.components_util import ComponentRegistry

        assert isinstance(embedders, ComponentRegistry)
        assert embedders.name == 'embedders'

        # Check that known embedders are registered
        assert 'constant_vectorizer' in embedders
        assert 'simple_text_embedder' in embedders

        # Default should be set
        assert 'default' in embedders

    def test_planarizers_registry(self):
        """Planarizers registry works correctly."""
        from imbed.components import planarizers
        from imbed.components.components_util import ComponentRegistry

        assert isinstance(planarizers, ComponentRegistry)
        assert planarizers.name == 'planarizers'

        # Check that known planarizers are registered
        assert 'constant_planarizer' in planarizers
        assert 'identity_planarizer' in planarizers

        # Default should be set
        assert 'default' in planarizers

    def test_clusterers_registry(self):
        """Clusterers registry works correctly."""
        from imbed.components import clusterers
        from imbed.components.components_util import ComponentRegistry

        assert isinstance(clusterers, ComponentRegistry)
        assert clusterers.name == 'clusterers'

        # Check that known clusterers are registered
        assert 'constant_clusterer' in clusterers
        assert 'random_clusterer' in clusterers

        # Default should be set
        assert 'default' in clusterers


# Test registration patterns
class TestRegistrationPatterns:
    """Tests for different registration patterns."""

    def test_function_registration(self):
        """Functions can be registered."""
        from imbed.components import segmenters

        # Save original state
        original_keys = set(segmenters.keys())

        @segmenters.register()
        def test_segmenter(text):
            return text.split(',')

        assert 'test_segmenter' in segmenters
        result = segmenters['test_segmenter']("a,b,c")
        assert list(result) == ['a', 'b', 'c']

        # Clean up
        del segmenters['test_segmenter']

    def test_custom_key_registration(self):
        """Components can be registered with custom keys."""
        from imbed.components import embedders

        @embedders.register('my_custom_embedder')
        def custom_embed(text):
            return [1.0, 2.0, 3.0]

        assert 'my_custom_embedder' in embedders
        assert embedders['my_custom_embedder']("test") == [1.0, 2.0, 3.0]

        # Clean up
        del embedders['my_custom_embedder']


# Test component functionality
class TestComponentFunctionality:
    """Tests that actual components work correctly."""

    def test_segmenter_works(self):
        """String lines segmenter works."""
        from imbed.components import segmenters

        text = "line 1\nline 2\nline 3"
        result = list(segmenters['string_lines'](text))
        assert result == ['line 1', 'line 2', 'line 3']

    def test_embedder_works(self):
        """Constant vectorizer works."""
        from imbed.components import embedders

        segments = ["a", "b", "c"]
        result = embedders['constant_vectorizer'](segments)
        assert len(result) == 3
        assert all(isinstance(v, list) for v in result)

    def test_planarizer_works(self):
        """Identity planarizer works."""
        from imbed.components import planarizers

        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        result = planarizers['identity_planarizer'](vectors)
        assert len(result) == 2
        assert result[0] == (1.0, 2.0)
        assert result[1] == (4.0, 5.0)

    def test_clusterer_works(self):
        """Constant clusterer works."""
        from imbed.components import clusterers

        vectors = [[1, 2], [3, 4], [5, 6]]
        result = clusterers['constant_clusterer'](vectors)
        assert len(result) == 3
        assert all(isinstance(c, int) for c in result)


# Test default key functionality
class TestDefaultKey:
    """Tests for default key functionality."""

    def test_default_key_exists(self):
        """Each registry has a default key."""
        from imbed.components import segmenters, embedders, planarizers, clusterers

        assert 'default' in segmenters
        assert 'default' in embedders
        assert 'default' in planarizers
        assert 'default' in clusterers

    def test_default_key_is_functional(self):
        """Default components work."""
        from imbed.components import segmenters

        default_segmenter = segmenters['default']
        assert callable(default_segmenter)


# Test backward compatibility
class TestBackwardCompatibility:
    """Tests that existing code still works."""

    def test_dict_like_access(self):
        """Registries still work like dicts."""
        from imbed.components import segmenters

        # Can iterate over keys
        keys = list(segmenters.keys())
        assert len(keys) > 0

        # Can get values
        for key in keys[:3]:  # Check first 3
            value = segmenters[key]
            assert callable(value)

        # Can check membership
        assert 'string_lines' in segmenters

    def test_existing_registration_pattern_works(self):
        """Old registration patterns still work."""
        from imbed.components.components_util import ComponentRegistry

        registry = ComponentRegistry('test')

        # Old pattern: direct dict assignment
        def my_func():
            return 42

        registry['direct_key'] = my_func
        assert registry['direct_key']() == 42


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
