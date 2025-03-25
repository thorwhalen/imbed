# imbed

Tools to work with embeddings, easily an flexibily.

To install:	```pip install imbed```


# Introduction

As we all know, though RAG (Retrieval Augumented Generation) is hyper-popular at the moment, the R part, though around for decades 
(mainly under the names "information retrieval" (IR), "search", "indexing",...), has a lot to contribute towards the success, or failure, of the effort.
The [many characteristics of the retrieval part](https://arxiv.org/abs/2312.10997) need to be tuned to align with the final generation and business objectives. 
There's still a lot of science to do. 

So the last thing we want is to be slowed down by pedestrian aspects of the process. 
We want to be agile in getting data prepared and analyzed, so we spend more time doign science, and iterate our models quickly.

There are two major aspects the `imbed` wishes to contribute two that.
* search: getting from raw data to an iterface where we can search the information effectively
* visualize: exploring the data visually (which requires yet another kind of embedding, to 2D or 3D vectors)

What we're looking for here is a setup where with minimal **configuration** (not code), we can make pipelines where we can point to the original data, enter a few parameters, 
wait, and get a "search controller" (that is, an object that has all the methods we need to do retrieval stuff). Here's an example of the kind of interface we'd like to target.

```python
raw_docs = mk_text_store(doc_src_uri)  # the store used will depend on the source and format of where the docs are stored
segments = mk_segments_store(raw_docs, ...)  # will not copy any data over, but will give a key-value view of chunked (split) docs
search_ctrl = mk_search_controller(vectorDB, embedder, ...)
search_ctrl.fit(segments, doc_src_uri, ...)
search_ctrl.save(...)
```

# Basic Usage

## Text Segmentation

```python
from imbed.segmentation import fixed_step_chunker

# Create chunks of text with a specific size
text = "This is a sample text that will be divided into smaller chunks for processing."
chunks = list(fixed_step_chunker(text.split(), chk_size=3))
print(chunks)
# Output: [['This', 'is', 'a'], ['sample', 'text', 'that'], ['will', 'be', 'divided'], ['into', 'smaller', 'chunks'], ['for', 'processing.']]

# Create overlapping chunks with a step size
overlapping_chunks = list(fixed_step_chunker(text.split(), chk_size=4, chk_step=2))
print(overlapping_chunks)
# Output: [['This', 'is', 'a', 'sample'], ['a', 'sample', 'text', 'that'], ...]
```

## Working with Embeddings

```python
import numpy as np
from imbed.util import cosine_similarity, planar_embeddings, transpose_iterable

# Create some example embeddings
embeddings = {
    "doc1": np.array([0.1, 0.2, 0.3]),
    "doc2": np.array([0.2, 0.3, 0.4]),
    "doc3": np.array([0.9, 0.8, 0.7])
}

# Calculate cosine similarity between embeddings
similarity = cosine_similarity(embeddings["doc1"], embeddings["doc2"])
print(f"Similarity between doc1 and doc2: {similarity:.3f}")

# Project embeddings to 2D for visualization
planar_coords = planar_embeddings(embeddings)
print("2D coordinates for visualization:")
for doc_id, coords in planar_coords.items():
    print(f"  {doc_id}: {coords}")

# Get x, y coordinates separately for plotting
x_values, y_values = transpose_iterable(planar_coords.values())
```

## Creating a Search System

```python
from imbed.segmentation import SegmentStore

# Example document store
docs = {
    "doc1": "This is the first document about artificial intelligence.",
    "doc2": "The second document discusses neural networks and deep learning.",
    "doc3": "Document three covers natural language processing."
}

# Create segment keys (doc_id, start_position, end_position)
segment_keys = [
    ("doc1", 0, len(docs["doc1"])),
    ("doc2", 0, 27),  # First half
    ("doc2", 28, len(docs["doc2"])),  # Second half
    ("doc3", 0, len(docs["doc3"]))
]

# Create a segment store
segment_store = SegmentStore(docs, segment_keys)

# Get a segment
print(segment_store[("doc2", 28, len(docs["doc2"]))])
# Output: "neural networks and deep learning."

# Iterate over all segments
for key in segment_store:
    segment = segment_store[key]
    print(f"{key}: {segment[:20]}...")
```

## Storage Utilities

```python
import os
from imbed.util import extension_based_wrap
from dol import Files

# Create a directory for storing data
os.makedirs("./data_store", exist_ok=True)

# Create a store that handles encoding/decoding based on file extensions
store = extension_based_wrap(Files("./data_store"))

# Store different types of data with appropriate extensions
store["config.json"] = {"model": "text-embedding-3-small", "batch_size": 32}
store["embeddings.npy"] = np.random.random((10, 128))

# The data is automatically encoded/decoded based on file extension
config = store["config.json"]  # Decoded from JSON automatically
embeddings = store["embeddings.npy"]  # Loaded as numpy array automatically

# Check available codec mappings
from imbed.util import get_codec_mappings
print("Available codecs:", list(get_codec_mappings()[0].keys()))
```

# Advanced Pipeline

For more complex use cases, imbed enables a configuration-driven pipeline approach:

```py
# Example of the configuration-driven pipeline (conceptual)
raw_docs = mk_text_store("s3://my-bucket/documents/")
segments = mk_segments_store(raw_docs, chunk_size=512, overlap=128)
search_ctrl = mk_search_controller(vector_db="faiss", embedder="text-embedding-3-small")
search_ctrl.fit(segments)
search_ctrl.save("./search_model")

# Search using the controller
results = search_ctrl.search("How does machine learning work?")
```

# Working with Embeddings and Visualization in imbed

The imbed package provides powerful tools for working with embeddings, particularly for visualizing high-dimensional data and identifying meaningful clusters. Here are examples showing how to use the planarization and clusterization modules.

## Planarization for Embedding Visualization

Embedding models typically produce high-dimensional vectors (e.g., 384 or 1536 dimensions) that can't be directly visualized. The planarization module helps project these vectors to 2D space for visualization purposes.

```py
import numpy as np
import matplotlib.pyplot as plt
from imbed.components.planarization import planarizers, umap_planarizer

# Create some sample high-dimensional embeddings
np.random.seed(42)
embeddings = np.random.randn(100, 128)  # 100 documents with 128-dimensional embeddings

# Project embeddings to 2D using UMAP (great for preserving local relationships)
planar_points = umap_planarizer(
    embeddings,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)

# Convert to separate x and y coordinates for plotting
x_coords, y_coords = zip(*planar_points)

# Create a simple scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(x_coords, y_coords, alpha=0.7)
plt.title("Document Embeddings Visualization using UMAP")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(alpha=0.3)
plt.show()

# Available planarization algorithms
print(f"Available planarization methods: {list(planarizers.keys())}")
```

The planarization module offers multiple techniques for dimensionality reduction:

* `umap_planarizer`: Great for preserving both local and global relationships
* `tsne_planarizer`: Good for preserving local neighborhood relationships
* `pca_planarizer`: Linear projection that preserves global variance
* `force_directed_planarizer`: Physics-based approach for visualization

Each algorithm has different strengths - UMAP is generally excellent for embedding visualization, while t-SNE is better for highlighting local clusters.

## Clusterization for Content Organization

After projecting embeddings to 2D, you can cluster them to identify groups of related documents:

```py
import numpy as np
import matplotlib.pyplot as plt
from imbed.components.planarization import umap_planarizer
from imbed.components.clusterization import kmeans_clusterer, hierarchical_clusterer, clusterers

# Create some sample embeddings
np.random.seed(42)
# Create 3 distinct groups of embeddings
group1 = np.random.randn(30, 128) + np.array([2.0] * 128)
group2 = np.random.randn(40, 128) - np.array([2.0] * 128)
group3 = np.random.randn(30, 128) + np.array([0.0] * 128)
embeddings = np.vstack([group1, group2, group3])

# First project to 2D for visualization
planar_points = umap_planarizer(embeddings, random_state=42)
x_coords, y_coords = zip(*planar_points)

# Apply clustering to the original high-dimensional embeddings
cluster_ids = kmeans_clusterer(embeddings, n_clusters=3)

# Visualize the clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(x_coords, y_coords, c=cluster_ids, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster ID')
plt.title("Document Clusters Visualization")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(alpha=0.3)
plt.show()

# Try a different clustering algorithm
hierarchical_clusters = hierarchical_clusterer(embeddings, n_clusters=3, linkage='ward')

# Compare clustering results
agreement = sum(1 for a, b in zip(cluster_ids, hierarchical_clusters) if a == b) / len(cluster_ids)
print(f"Agreement between kmeans and hierarchical clustering: {agreement:.2%}")

# Available clustering algorithms
print(f"Available clustering methods: {list(clusterers.keys())}")
```


## Labeling clusters

A useful AI-based tool to label clusters.

```py
from imbed import cluster_labeler

import pandas as pd
import numpy as np
from typing import Callable, Union

# Sample data with text segments and cluster indices
data = {
    'segment': [
        "Machine learning models can be trained on large datasets to identify patterns.",
        "Neural networks are a subset of machine learning algorithms inspired by the human brain.",
        "Deep learning is a type of neural network with multiple hidden layers.",
        "Python is a versatile programming language used in data science and web development.",
        "JavaScript is primarily used for web development and creating interactive websites.",
        "HTML and CSS are markup languages used to structure and style web pages.",
        "SQL is a query language designed for managing and manipulating databases.",
        "NoSQL databases like MongoDB store data in flexible, JSON-like documents."
    ],
    'cluster_idx': [0, 0, 0, 1, 1, 1, 2, 2]  # 3 clusters
}

# Create the dataframe
df = pd.DataFrame(data)

# You can test with:
labels = cluster_labeler(df, context="Technical documentation")
labels
```

    {0: 'Neural Networks Overview',
    1: 'Web Development Language Comparisons',
    2: 'Database Management Comparisons'}


## Why This Matters for Embedding Visualization

Both planarization and clusterization are essential for making sense of embeddings:

Dimensionality Reduction: High-dimensional embeddings can't be directly visualized. Planarization techniques reduce them to 2D or 3D for plotting while preserving meaningful relationships.

Pattern Discovery: Clustering helps identify natural groupings within your data, revealing thematic structures that might not be obvious.

Content Organization: You can use clusters to automatically organize documents by topic, identify outliers, or create faceted navigation systems.

Relevance Evaluation: Visualizing embeddings lets you assess whether your embedding model is capturing meaningful semantic relationships.

Iterative Refinement: Visual inspection of embeddings and clusters helps you iterate on your data preparation, segmentation, and model selection strategies.

The imbed package makes these powerful techniques accessible through a simple, unified interface, allowing you to focus on the analysis rather than implementation details.
