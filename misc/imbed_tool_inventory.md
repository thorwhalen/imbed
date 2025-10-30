# Imbed & Imbed_Data_Prep Tool Inventory

Inventory compiled to map current assets for segmentation → vectorization → planarization → clusterization pipelines.

### Segmentation Component Registry - Segmentation

**Location**: `t/imbed/imbed/components/segmentation.py`

**Purpose**: Maintains a registry of text segmentation functions to normalize diverse sources into iterable segments.

**Interface Type**:
- Function registry (`dict[str, Callable]`) with decorator-based registration.

**Key Features**:
- Built-in `string_lines`, `jdict_to_segments`, and `field_values` adapters.
- `register_segmenter` decorator for incremental extension.
- Default alias resolved via `add_default_key` and environment override.

**Strengths** ✓:
- Registry pattern keeps segmentation plug-and-play.
- Default fallback prevents hard crashes when selector missing.
- Supports JSON-friendly inputs (str, dict, iterable) out of the box.

**Weaknesses/Limitations** ⚠:
- Only trivial segmenters shipped; no sentence/token chunkers yet.
- Registry stores callables but lacks metadata describing outputs.
- Missing parameterization hooks (e.g., chunk size, overlap).

**Integration Potential**:
- Use `segmenters[key]` inside Dacc classes to normalize sources before vectorization.
- Could expose default key via project config for runtime swapping.

**Dependencies**: `i2.register_object`, Python stdlib.

**Example Usage**:
```python
from imbed.components.segmentation import segmenters
segments = list(segmenters['jdict_to_segments']("Line 1\nLine 2\n"))
```

### SegmentStore - Segmentation

**Location**: `t/imbed/imbed/segmentation_util.py`

**Purpose**: Provides a mapping-like view bridging documents and their segment windows, allowing retrieval by key or `(doc_key, start, end)` tuple.

**Interface Type**:
- Class implementing custom `__iter__`, `__getitem__`, `__setitem__`, and `__len__`.

**Key Features**:
- Accepts dict of documents plus explicit segment key tuples.
- Handles slice-based retrieval when keys are `(doc_key, start, end)`.
- Supports additive merge via `__add__` and exposes `values()` iterator.

**Strengths** ✓:
- Encapsulates text slicing so downstream code treats segments as mapping entries.
- Supports incremental updates, enabling caching of new segments.
- Works alongside chunking utilities for alignment.

**Weaknesses/Limitations** ⚠:
- Not a full Mapping implementation (no `.keys()` / `.items()`), so some APIs break.
- No validation that segment ranges stay within document bounds.
- Mutation semantics on overlapping keys are undocumented.

**Integration Potential**:
- Use as the backing store for segmentation stage outputs prior to vectorization.
- Could be wrapped with `dol` stores to persist segment selections.

**Dependencies**: Python stdlib only.

**Example Usage**:
```python
from imbed.segmentation_util import SegmentStore
docs = {'doc1': "abcdefgh"}
segments = [('doc1', 0, 4), ('doc1', 4, 8)]
store = SegmentStore(docs, segments)
snippet = store[('doc1', 0, 4)]
```

### Chunking Utilities - Segmentation

**Location**: `t/imbed/imbed/segmentation_util.py`

**Purpose**: Supplies reusable chunking helpers (`fixed_step_chunker`, `wrapped_chunker`, `chunk_mapping`, `chunk_dataframe`) to batch iterables, mappings, and DataFrames.

**Interface Type**:
- Functions returning generators over chunked collections.

**Key Features**:
- `fixed_step_chunker` supports step size, start/stop bounds, and tail handling.
- `wrapped_chunker` adapts any iterable source using ingress/egress transforms.
- `chunk_mapping` and `chunk_dataframe` preserve key/index alignment in chunk output.

**Strengths** ✓:
- Covers both sequential and mapping sources without duplicating logic.
- Doctested behavior simplifies validation for edge cases.
- Plays nicely with vector batching and OpenAI batch limits.

**Weaknesses/Limitations** ⚠:
- No built-in telemetry (chunk sizes, counters) for monitoring pipelines.
- Error messages minimal when chunker spec is invalid.
- Lacks overlap chunking strategy for long-context models.

**Integration Potential**:
- Feed directly into embedding functions (e.g., `compute_and_save_embeddings` batching).
- Could be wrapped into pipeline stage configs to standardize chunk sizes.

**Dependencies**: Python stdlib (`functools`, `itertools`, `operator`).

**Example Usage**:
```python
from imbed.segmentation_util import chunk_mapping, fixed_step_chunker
chunker = lambda data: fixed_step_chunker(data, chk_size=2)
batched = list(chunk_mapping({1: 'a', 2: 'b', 3: 'c'}, chunker))
```

### Vectorization Component Registry - Vectorization

**Location**: `t/imbed/imbed/components/vectorization.py`

**Purpose**: Central registry of embedding functions that turn text segments into vectors.

**Interface Type**:
- Function registry (`dict[str, Callable]`) with decorator-based registration and default alias.

**Key Features**:
- Bundles lightweight placeholders (`constant_vectorizer`, `simple_text_embedder`).
- Dynamically adds OpenAI embeddings via `oa.embeddings` when available.
- Default selection controlled by `DEFAULT_IMBED_VECTORIZER_KEY`.

**Strengths** ✓:
- Registry abstraction keeps pipeline pluggable across embedding providers.
- Placeholder vectorizers enable offline testing without API calls.
- Works with mappings and iterables, returning same structure shape.

**Weaknesses/Limitations** ⚠:
- No configuration metadata (e.g., dimensionality) stored alongside functions.
- Placeholder vectorizers lack normalization or token limits awareness.
- Fails silently if OpenAI import missing; users must inspect registry manually.

**Integration Potential**:
- Reference through `Project.embedders` or dataset classes for consistent embedding choice.
- Extend via `embedders['my_model'] = partial(...)` to wrap new providers.

**Dependencies**: `functools.partial`, optional `oa` package.

**Example Usage**:
```python
from imbed.components.vectorization import embedders
vectors = embedders['simple_text_embedder'](["short text", "longer segment"])
```

### compute_and_save_embeddings - Vectorization

**Location**: `t/imbed/imbed/base.py`

**Purpose**: Batch-computes embeddings for a DataFrame of segments and persists each chunk to a mutable store.

**Interface Type**:
- Function `compute_and_save_embeddings(df, save_store, **opts)`.

**Key Features**:
- Streams rows in configurable batches via `segmentation_util.batches`.
- Skips already-processed chunks unless `overwrite_chunks` is set.
- Accepts callable `key_for_chunk_index` to control storage keys.

**Strengths** ✓:
- Minimizes API costs by resuming from partially computed batches.
- Works with any `MutableMapping` backend (local Files, in-memory dict).
- Verbose logging hooks simplify monitoring long runs.

**Weaknesses/Limitations** ⚠:
- Hard-coded to OpenAI embeddings; lacks abstraction for other providers.
- Error handling logs but swallows exceptions per chunk without retries.
- Requires full DataFrame in memory before batching.

**Integration Potential**:
- Use in dataset Dacc classes to persist embeddings to disk.
- Could be wrapped in async job orchestrators for large corpora.

**Dependencies**: `pandas`, `numpy`, `oa.embeddings`, `dol.Files`.

**Example Usage**:
```python
from imbed.base import compute_and_save_embeddings
embeddings_store = {}
compute_and_save_embeddings(df, embeddings_store, text_col='segment')
```

### EmbeddingBatchManager - Vectorization

**Location**: `t/imbed/imbed/tools.py`

**Purpose**: Orchestrates OpenAI Batch API workflows for large embedding jobs, from submission to result aggregation.

**Interface Type**:
- Class with `run()`, plus helper generator methods; `compute_embeddings_in_bulk` facade.

**Key Features**:
- Converts mappings or iterables into batches using chunking utilities.
- Uploads tasks via `oa.stores.OaDacc`, tracks status, and polls completion.
- Aggregates output files into lists or DataFrames for downstream use.

**Strengths** ✓:
- Encapsulates the full batch lifecycle (submit, poll, collect) in one place.
- Stores intermediate artifacts (`submitted_batches`, `completed_batches`) for auditing.
- Pluggable `batcher` spec (int or callable) adapts to provider limits.

**Weaknesses/Limitations** ⚠:
- Hard-wired to OpenAI-specific DTOs; little abstraction for other vendors.
- Minimal error recovery if batches fail or poll cycles exhausted.
- Requires synchronous run; no callback integration for external schedulers.

**Integration Potential**:
- Wrap inside pipeline stage when using OpenAI Batch for embeddings.
- Could expose hooks to `Project` to auto-trigger remote embedding jobs.

**Dependencies**: `oa` client libraries, `dol`, `numpy`, `pandas`.

**Example Usage**:
```python
from imbed.tools import compute_embeddings_in_bulk
result = compute_embeddings_in_bulk({'id1': 'text'}, batcher=500)
```

### Embeddings Utility Suite - Vectorization Support

**Location**: `t/imbed/imbed/util.py`

**Purpose**: Provides helper classes and functions (`Embeddings`, `cosine_similarity`) for vector storage, search, and comparison.

**Interface Type**:
- Class `Embeddings` and supporting numeric functions.

**Key Features**:
- `Embeddings.search` performs cosine-similarity nearest neighbor queries.
- Factory constructors ingest mappings or DataFrames while preserving keys.
- `cosine_similarity` handles 1D/2D inputs with broadcast or cartesian modes.

**Strengths** ✓:
- Keeps lightweight similarity tooling close to embedding pipeline.
- Handles metadata functions via optional `meta` callable.
- Comprehensive doctests illustrate edge cases for row-wise vs cartesian comparisons.

**Weaknesses/Limitations** ⚠:
- `Embeddings` stores vectors in memory; no persistence or ANN acceleration.
- `cosine_similarity` raises on mismatched shapes instead of aligning automatically.
- No GPU or batching support for large matrices.

**Integration Potential**:
- Use for quick QA on embedding quality before committing to planarization.
- Plug into cluster labeling to fetch exemplar segments.

**Dependencies**: `numpy`, `sklearn.metrics.pairwise`.

**Example Usage**:
```python
from imbed.util import Embeddings
E = Embeddings.from_mapping({'a': [1, 0], 'b': [0, 1]}, meta=lambda k: {'label': k})
hits = E.search([0.8, 0.2], n=1)
```

### Planarization Component Registry - Planarization

**Location**: `t/imbed/imbed/components/planarization.py`

**Purpose**: Catalogs planarization functions that project high-dimensional embeddings into 2D coordinates.

**Interface Type**:
- Function registry (`dict[str, Callable]`) managed via `register_planarizer`.

**Key Features**:
- Ships dozens of algorithms (PCA, t-SNE, UMAP, Isomap, LLE, spectral, SVD, graph layouts).
- Default `constant_planarizer` plus random/grid fallbacks for testing.
- Environment override through `DEFAULT_IMBED_PLANARIZER_KEY`.

**Strengths** ✓:
- Broad coverage lets experiments swap dimensionality reducers quickly.
- Decorator pattern simplifies adding tuned variants (e.g., `umap_planarizer` with kwargs).
- Defensive guards handle small sample edge cases (single vector fallback).

**Weaknesses/Limitations** ⚠:
- Dependency heavy; many algorithms require optional imports that may fail at runtime.
- Parameter tuning requires partials or wrappers; registry stores bare callables.
- No metadata describing computational cost or suitability per dataset size.

**Integration Potential**:
- Connect to `Project.planarizers` to expose algorithm choice in UI/workflows.
- Could pair with `planar_embeddings` helper to standardize output format.

**Dependencies**: `numpy`, `scikit-learn`, `umap-learn`, `networkx` (optional).

**Example Usage**:
```python
from imbed.components.planarization import planarizers
points = planarizers['umap_planarizer']([[0.1, 0.2], [0.3, 0.4]])
```

### planar_embeddings Helpers - Planarization

**Location**: `t/imbed/imbed/util.py`

**Purpose**: High-level helpers that transform embedding mappings into 2D layouts and DataFrame-friendly outputs.

**Interface Type**:
- Functions `planar_embeddings`, `planar_embeddings_dict_to_df`, `umap_2d_embeddings_df`.

**Key Features**:
- Applies optional preprocessing pipeline (StandardScaler + PCA by default).
- Accepts callables or string specs to choose projection algorithm (umap, tsne, pca, ncvis).
- Converts results into dict or DataFrame with named columns and index retention.

**Strengths** ✓:
- Abstracts away shape conversions and preprocessing for consistent planar outputs.
- Supports environment-based default kind while allowing custom callables.
- Integrates seamlessly with pandas for downstream visualization.

**Weaknesses/Limitations** ⚠:
- Assumes embeddings fit in memory; no streaming projection.
- Limited error messaging when projection libraries unavailable.
- Preprocess pipeline fixed; customizing requires passing alternative function.

**Integration Potential**:
- Use inside dataset classes after embeddings to produce XY coordinates.
- Feed output directly to clustering or visualization stores.

**Dependencies**: `numpy`, `pandas`, `scikit-learn`, optional `umap`, `ncvis`.

**Example Usage**:
```python
from imbed.util import planar_embeddings, planar_embeddings_dict_to_df
xy = planar_embeddings({'a': [0.1, 0.3, 0.5], 'b': [0.2, 0.4, 0.6]})
df = planar_embeddings_dict_to_df(xy, key_col=True)
```

### Clusterization Component Registry - Clusterization

**Location**: `t/imbed/imbed/components/clusterization.py`

**Purpose**: Maintains a catalog of clustering functions for assigning embeddings to cluster IDs.

**Interface Type**:
- Function registry (`dict[str, Callable]`) populated via `register_clusterer`.

**Key Features**:
- Includes scratch implementations (threshold, kmeans_lite) and scikit-learn wrappers (KMeans, DBSCAN, OPTICS, HDBSCAN, bisecting).
- Default alias `constant_clusterer` for testing and pipeline sanity checks.
- Helper utilities `get_clusterer`, `list_available_clusterers`, `scan_for_clusterers`.

**Strengths** ✓:
- Offers rich algorithm variety covering centroidal, density, spectral, and hybrid flows.
- Consistent signature simplifies experiment automation.
- Handles missing optional dependencies gracefully via `suppress_import_errors`.

**Weaknesses/Limitations** ⚠:
- Some algorithms assume dense numpy arrays; no sparse support.
- Parameter defaults are generic; no heuristics per dataset size.
- Lack of metadata makes comparing algorithm trade-offs manual.

**Integration Potential**:
- Reference from `Project.clusterers` or dataset classes to compute label sets per experiment.
- Pair with `clusters_df` to produce multi-resolution cluster tables.

**Dependencies**: `numpy`, `scikit-learn`, optional `umap`, `hdbscan`.

**Example Usage**:
```python
from imbed.components.clusterization import clusterers
labels = clusterers['dbscan_clusterer']([[0, 0], [1, 1], [10, 10]], eps=2.0)
```

### kmeans_cluster_indices - Clusterization

**Location**: `t/imbed/imbed/data_prep.py`

**Purpose**: Generates cluster labels using KMeans or MiniBatchKMeans depending on the input shape or streaming mode.

**Interface Type**:
- Function accepting numpy arrays, iterables of batches, or batch factories.

**Key Features**:
- Automatically switches to `MiniBatchKMeans` when provided batches or generators.
- Supports additional learner kwargs forwarded to scikit-learn estimators.
- Works with lambda returning generator to allow multiple passes for prediction.

**Strengths** ✓:
- Flexible input contract suits both in-memory and streaming embeddings.
- Docstring examples cover both classic and minibatch workflows.
- Simplifies common clustering step for multi-resolution analyses.

**Weaknesses/Limitations** ⚠:
- Defined twice in module; initial simple version is shadowed, hinting at refactor debt.
- No partial-fit progress logging, making long runs opaque.
- Assumes numeric numpy arrays; lacks support for pandas DataFrames directly.

**Integration Potential**:
- Use as the default `mk_cluster_learner` in dataset prep classes.
- Could be wrapped by `Project` cluster stage to pipeline chunked embeddings.

**Dependencies**: `numpy`, `scikit-learn`.

**Example Usage**:
```python
import numpy as np
from imbed.data_prep import kmeans_cluster_indices
data = np.random.rand(100, 5)
labels = kmeans_cluster_indices(data, n_clusters=5, random_state=42)
```

### clusters_df - Clusterization

**Location**: `t/imbed/imbed/data_prep.py`

**Purpose**: Builds a DataFrame aggregating cluster assignments across multiple cluster sizes for the same embedding set.

**Interface Type**:
- Function `clusters_df(embeddings, n_clusters=sequence)` returning `pandas.DataFrame`.

**Key Features**:
- Accepts pandas DataFrame, mapping, or iterable of vectors, preserving keys where possible.
- Defaults to Fibonacci cluster counts `(5, 8, 13, 21, 34)` for multi-scale exploration.
- Returns wide DataFrame with `cluster_{k}` columns keyed by original index.

**Strengths** ✓:
- Simplifies production of multi-resolution clustering artifacts for dashboards.
- Handles DataFrame input with `embedding` column seamlessly.
- Pairs naturally with planar outputs for quick plotting.

**Weaknesses/Limitations** ⚠:
- Forces embeddings into numpy array; large datasets may exhaust memory.
- Relies on `kmeans_cluster_indices` without exposing alternative clusterers.
- No caching or persistence built in.

**Integration Potential**:
- Use inside dataset modules after embeddings to enrich artifacts.
- Could feed results into `ClusterLabeler` for labeling.

**Dependencies**: `numpy`, `pandas`.

**Example Usage**:
```python
from imbed.data_prep import clusters_df
table = clusters_df({'a': [0, 1], 'b': [1, 0]}, n_clusters=(2, 3))
```

### ClusterLabeler - Metadata Management

**Location**: `t/imbed/imbed/tools.py`

**Purpose**: Automates cluster naming by sampling cluster members and prompting an LLM for concise titles.

**Interface Type**:
- Class with `label_clusters(df)` producing dict of `cluster_idx -> title`.

**Key Features**:
- Configurable sample size, context prompt, and max cluster count guard.
- Supports callable or column name for retrieving segment text per row.
- Uses `oa.prompt_function` to construct reusable prompt template.

**Strengths** ✓:
- Bridges clustering output with human-friendly summaries quickly.
- Sampling/truncation controls prevent token overflows.
- Designed for reuse across datasets via parameterized init.

**Weaknesses/Limitations** ⚠:
- Depends on OpenAI completion; no retry/backoff or provider abstraction.
- Assumes DataFrame structure with specific column names.
- Limited evaluation feedback (no confidence scores or examples returned).

**Integration Potential**:
- Apply after `clusters_df` to decorate cluster tables before visualization.
- Could be wrapped in pipeline step to store titles alongside indices.

**Dependencies**: `oa`, `numpy`, `pandas`.

**Example Usage**:
```python
from imbed.tools import ClusterLabeler
labeler = ClusterLabeler(context="GitHub repos", cluster_idx_col='cluster_05')
titles = labeler.label_clusters(df_with_segments)
```

### Project - Pipeline/Workflow Management

**Location**: `t/imbed/imbed/imbed_project.py`

**Purpose**: Facade class coordinating segments, embeddings, planar projections, and cluster labels across configurable stores.

**Interface Type**:
- Dataclass `Project` with methods `add_segments`, `_compute_embeddings_async`, and store accessors.

**Key Features**:
- Integrates component registries (`embedders`, `planarizers`, `clusterers`) via malls.
- Supports RAM or local storage through `get_mall` and `mk_mall_kinds`.
- Optional async embedding computation backed by `au` framework.

**Strengths** ✓:
- Provides single point of coordination for pipeline state and invalidation.
- Mall concept separates storage concerns from computation logic.
- Extensible via `from_mall` factory to plug in custom store layouts.

**Weaknesses/Limitations** ⚠:
- Async path appears unfinished; broader API surface still under construction.
- Limited documentation on invalidation cascade and auto-compute toggles.
- Lack of tests leaves behavior (especially error handling) uncertain.

**Integration Potential**:
- Ideal skeleton for the desired pipeline framework once segment/vector stages plug in.
- Could orchestrate dataset Dacc outputs by binding their stores to project mall.

**Dependencies**: `au`, `dol`, Python stdlib.

**Example Usage**:
```python
from imbed.imbed_project import Project
project = Project.from_mall(mk_mall='ram')
project.add_segments({'seg-1': "Hello world"})
```

### HugfaceDaccBase - Data Acquisition

**Location**: `t/imbed/imbed/base.py`

**Purpose**: Base class for dataset acquisition and caching when sourcing data from Hugging Face datasets.

**Interface Type**:
- Dataclass mixing `LocalSavesMixin` with dataset-specific properties.

**Key Features**:
- Lazily loads dataset splits via `datasets.load_dataset`.
- Provides `train_data`, `test_data`, and `all_data` pandas DataFrames.
- Manages per-dataset save directories with config-driven defaults.

**Strengths** ✓:
- Centralizes huggingface download and caching logic for reuse across datasets.
- Integrates with dol stores for persisted artifacts.
- Offers easy extension by overriding methods in child Dacc classes.

**Weaknesses/Limitations** ⚠:
- Does not define segmentation or embedding behaviors directly.
- Hardcodes expectation of `train`/`test` splits; other schemas require overrides.
- Limited error handling if dataset missing or corrupted.

**Integration Potential**:
- Extend for each dataset in `imbed_data_prep` to align with pipeline interfaces.
- Could feed `Project` segments by exposing `text_segments` property.

**Dependencies**: `datasets`, `pandas`, `dol`.

**Example Usage**:
```python
from imbed.base import HugfaceDaccBase
class TinyDacc(HugfaceDaccBase):
    pass
dataset = TinyDacc(huggingface_data_stub="allenai/WildChat-1M")
df = dataset.train_data.head()
```

### Store Mall Builders - Caching/Persistence

**Location**: `t/imbed/imbed/stores_util.py`

**Purpose**: Factory utilities to construct hierarchical storage "malls" backed by filesystem directories with extension-aware codecs.

**Interface Type**:
- Functions `mk_blob_store_for_path` and `extension_based_mall_maker`.

**Key Features**:
- Generates nested directory structures (`spaces/<space>/stores/<kind>`) automatically.
- Applies codec pipelines (json, dill, pickle, plaintext) based on filename suffix.
- Provides ready-made local store makers (`local_store_makers`) for common formats.

**Strengths** ✓:
- Encapsulates repetitive directory/coding logic for persistent artifacts.
- Plays nicely with `dol` wrappers for filtering and key completion.
- Flexible parameters allow customizing prefixes, suffixes, and base store wrappers.

**Weaknesses/Limitations** ⚠:
- API surface large; documentation sparse, making correct usage non-trivial.
- No validation that requested store kind already exists in mall.
- Lacks remote storage options (S3, SQLite) out of the box.

**Integration Potential**:
- Use to materialize persistence layers for each pipeline stage (segments, embeddings, clusters).
- Pair with `Project.get_mall` to wire custom stores for specific projects.

**Dependencies**: `dol`, `tabled.extension_based_wrap`, `dill`, `pickle`, `json`.

**Example Usage**:
```python
from imbed.stores_util import extension_based_mall_maker
makers = extension_based_mall_maker()
json_store = makers['json']("/tmp/imbed", space='demo')
json_store['example'] = {'status': 'ok'}
```

### PartializedFuncs - Pipeline Composition

**Location**: `t/imbed/imbed/imbed_project.py`

**Purpose**: Mapping wrapper that returns either raw functions or partially applied variants based on dict keys, enabling lightweight dependency injection.

**Interface Type**:
- Class implementing `Mapping[str | dict, Callable]`.

**Key Features**:
- Accepts dict key `{'func_name': {'param': value}}` to produce partial function.
- Preserves original function name for readability via `named_partial`.
- Integrates with component registries to offer parameterized variants without extra storage.

**Strengths** ✓:
- Enables configuration-driven component selection without writing new functions.
- Reduces boilerplate when exposing algorithm knobs through `mall` stores.
- Works seamlessly with `Project` to interpret requests for tuned components.

**Weaknesses/Limitations** ⚠:
- Supports only a single function per dict key; nested configs unsupported.
- No validation of kwargs against function signature (errors deferred to call time).
- Lacks caching for repeated partial creations.

**Integration Potential**:
- Use in `get_mall` to expose ready-to-run component variants to UI or CLI.
- Could power a configuration schema translating JSON into pipeline components.

**Dependencies**: Python stdlib (`functools.partial`, `collections.abc.Mapping`).

**Example Usage**:
```python
from imbed.imbed_project import PartializedFuncs
store = PartializedFuncs({'kmeans': lambda data, n_clusters=2: n_clusters})
kmeans5 = store[{'kmeans': {'n_clusters': 5}}]
```

### WildchatDacc - Dataset Preparation

**Location**: `t/imbed_data_prep/imbed_data_prep/wildchat.py`

**Purpose**: Dataset access class for the WildChat corpus, handling flattening, token validation, and embedding readiness.

**Interface Type**:
- Dataclass `WildchatDacc(HugfaceDaccBase)` with cached properties and methods.

**Key Features**:
- `expanded_train` flattens nested conversations and moderation metadata.
- `embeddable_df` filters segments based on token limits and enriches with counts.
- Provides language-specific stats (`language_conversation_counts`, `language_turn_counts`).

**Strengths** ✓:
- Leverages caching decorators to avoid recomputing expensive transformations.
- Integrates token validity checks to ensure embeddings pipeline safety.
- Offers quick access to subset (e.g., English segments) for targeted prep.

**Weaknesses/Limitations** ⚠:
- Several commented-out methods hint at incomplete planarization/cluster steps.
- Hard-coded huggingface stub and cache env var names reduce reuse.
- Embedding pipeline still synchronous; no batching integration yet.

**Integration Potential**:
- Feed `embeddable_df` into vectorization stage, then pipe results to `Project`.
- Could expose `expanded_train` as segmentation source for other tasks.

**Dependencies**: `pandas`, `numpy`, `dol.cache_this`, `tabled`, `oa`, `imbed.util`.

**Example Usage**:
```python
from imbed_data_prep.imbed_data_prep.wildchat import WildchatDacc
dacc = WildchatDacc(model='text-embedding-3-small')
english = dacc.expanded_en.head()
```

### GithubReposData - Dataset Preparation

**Location**: `t/imbed_data_prep/imbed_data_prep/github_repos.py`

**Purpose**: Manages download, caching, planarization, and clustering for GitHub repository metadata embeddings.

**Interface Type**:
- Dataclass `GithubReposData` with cached methods (`raw_data`, `planar_embeddings`).

**Key Features**:
- Downloads parquet via `graze` with progress logging and dedupes repositories.
- Provides convenience accessors (`text_segments`, `embeddings`, `embeddings_matrix`).
- Computes planar embeddings and clusters with configurable learners.

**Strengths** ✓:
- Encapsulates end-to-end prep flow (load → planarize → cluster).
- Uses `CacheSpec` and `ensure_cache` to standardize storage directories.
- Logging hooks (`clog`, `log_calls`) aid observability.

**Weaknesses/Limitations** ⚠:
- Embedding computation step currently commented out (assumes embeddings precomputed).
- No error handling for download failures or schema drift.
- Cluster persistence limited to parquet via `cache_this`.

**Integration Potential**:
- Plug into visualization pipeline by merging planar embeddings with metadata.
- Replace `mk_cluster_learner` to experiment with alternative clustering algorithms.

**Dependencies**: `pandas`, `numpy`, `graze`, `dol`, `imbed.util`, `imbed.data_prep`.

**Example Usage**:
```python
from imbed_data_prep.imbed_data_prep.github_repos import GithubReposData
repos = GithubReposData()
df = repos.raw_data().head()
```

### LmsysChatDacc - Dataset Preparation

**Location**: `t/imbed_data_prep/imbed_data_prep/lmsys_ai_conversations.py`

**Purpose**: Handles loading and flattening of the LMSYS Chat-1M dataset with utilities for embeddings aggregation.

**Interface Type**:
- Class `Dacc` with cached properties and stores.

**Key Features**:
- Loads huggingface dataset, caches to `Files` store with key completion.
- Produces flattened English conversation DataFrame with expanded moderation fields.
- Offers iterators for chunked embeddings (`flat_en_embeddings_store`).

**Strengths** ✓:
- Extensive flattening pipeline simplifies downstream segmentation.
- Counts helpers (`language_count`, `role_count`) support exploratory analysis.
- Reuses `imbed.util.counts` and `tabled` utilities to manage nested data.

**Weaknesses/Limitations** ⚠:
- Many steps assume cached parquet files already written (tight coupling to saves).
- Embedding computation not encapsulated; expects external tooling.
- Mixed return types (DataFrame vs dict) without consistent interfaces.

**Integration Potential**:
- Use `flat_en_embeddings_iter` to stream vectors into planarization/clustering.
- Could adapt to `HugfaceDaccBase` to align with other dataset classes.

**Dependencies**: `datasets`, `pandas`, `numpy`, `dol`, `tabled`, `imbed.util`.

**Example Usage**:
```python
from imbed_data_prep.imbed_data_prep.lmsys_ai_conversations import Dacc
dacc = Dacc()
tall = dacc.flat_en.head()
```

## Summary Analysis

- **Coverage Map**: Planarization and clusterization are richly covered (large registries, labeling utilities); vectorization has mid-level support through registries and batch tooling; segmentation remains sparse with only baseline chunkers/registries.
- **Design Consistency**: Component registries share a coherent decorator + default-key pattern, but dataset modules diverge in style and inheritance (some use `HugfaceDaccBase`, others standalone).
- **Integration Readiness**: Registries (`segmenters`, `embedders`, `planarizers`, `clusterers`) and store builders are ready to slot into a framework; `Project` and dataset Daccs need polish before production use.
- **Gaps**: Missing advanced text segmenters, vector store abstractions, provenance tracking, and standardized metadata schema; async embedding path incomplete.
- **Redundancy**: Multiple clustering algorithms overlap (three K-means variants, both sklearn and scratch implementations) without guidance on selection; duplicate `kmeans_cluster_indices` definitions add confusion.
- **Interoperability**: Registries can be wired together, but dataset modules and the `Project` facade are not yet connected via common interfaces or malls.

## Recommendations

- **Quick Wins**: 1) Remove the duplicate `kmeans_cluster_indices` definition and surface registry listings for discoverability; 2) Expose default component keys through config so datasets can opt-in without code changes; 3) Wrap dataset Dacc outputs in `SegmentStore` and `clusters_df` to exercise the full pipeline.
- **Refactoring Priorities**: 1) Normalize dataset classes around `HugfaceDaccBase` plus explicit segmentation/vectorization hooks; 2) Add metadata descriptors (dimension, cost) to component registries; 3) Tighten error handling/logging in `EmbeddingBatchManager` and `compute_and_save_embeddings`.
- **Build vs Integrate**: 1) Build richer segmentation utilities (tokenizers, sliding windows); 2) Integrate existing ANN/vector DB libraries into `vector_db.py` instead of bespoke implementations; 3) Evaluate third-party workflow tools before extending `Project`.
- **Architecture Suggestions**: 1) Define a formal pipeline spec (`SegmenterSpec`, `VectorizerSpec`, etc.) tied to malls; 2) Treat each stage output as a `Store` with provenance metadata; 3) Align dataset modules to emit standardized keys so `Project` can orchestrate optional/skip stages cleanly.
