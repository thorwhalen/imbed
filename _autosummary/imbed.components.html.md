# imbed.components

Components for imbed applications.

This module provides lazy-loading access to component registries:

- segmenters: Text segmentation functions
- embedders: Vectorization/embedding functions
- planarizers: 2D projection functions for visualization
- clusterers: Clustering algorithms

Components are loaded on first access to minimize import time.

### Examples

```pycon
>>> from imbed.components import segmenters
>>> list(segmenters.keys())[:3]
['string_lines', 'jdict_to_segments', 'field_values']
```

```pycon
>>> from imbed.components import components
>>> embedders = components.embedders  # Lazy load via attribute
>>> planarizers = components['planarizers']  # Lazy load via dict access
>>> 'constant_vectorizer' in embedders
True
>>> 'segmenters' in components  # Check if component type exists
True
>>> list(components)  # Iterate over component names
['segmenters', 'embedders', 'planarizers', 'clusterers']
```

### Modules

| [`clusterization`](imbed.components.clusterization.html.md#module-imbed.components.clusterization)   | Clusterization module for imbed.                          |
|----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| [`components_util`](imbed.components.components_util.html.md#module-imbed.components.components_util) | Utils for components                                      |
| [`planarization`](imbed.components.planarization.html.md#module-imbed.components.planarization)     | Planarization functions for embedding visualization.      |
| [`segmentation`](imbed.components.segmentation.html.md#module-imbed.components.segmentation)       | Segmentation functions to get text segments.              |
| [`vectorization`](imbed.components.vectorization.html.md#module-imbed.components.vectorization)     | Vectorization functions for converting text to embeddings |
