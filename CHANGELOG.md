# Changelog

All notable changes to this project are documented in this file.

The format is inspired by [Keep a Changelog](https://keepachangelog.com/);
each section corresponds to a git version tag (which is also the release
published to PyPI). Entries are commit subjects and PR titles, verbatim.

## [0.0.24] - 2026-05-16

- Add fit_kmeans returning a re-applicable codebook ([#13](https://github.com/thorwhalen/imbed/pull/13))
- Refactor components module to use ComponentRegistry for registration

### Added

- feat: Enhance planar_embeddings function with egress parameter and default preprocess handling
- feat: Enhance clustering and planarization with cosine distance and PCA optimizations for semantic embeddings
- feat: Add comprehensive tool inventory documentation for imbed components

### Changed

- refactor: Update type hints to use built-in collections.abc types and simplify annotations (Modernizing python style)
- refactor: Rename get_app_data_folder to get_app_config_folder

### Fixed

- fix: Add missing comma in t-SNE implementation parameters

## [0.0.23] - 2025-10-02

- get_app_folder -> get_data_app_folder

## [0.0.22] - 2025-08-22

- edit setup.cfg

## [0.0.21] - 2025-07-20

- Merge tag '0.0.20'
- chore: remove obsolete typical functionality test file

### Changed

- refactor: reorganize test functions and add test documents for search functionality

## [0.0.20] - 2025-07-20

### Added

- feat: add comprehensive tests for search functionality and document retrieval

### Fixed

- fix: made test_typical_functionality.py module helper function be check_ functions instead of test_ functions (which was tripping pytest)
- fix: imports

## [0.0.19] - 2025-07-09

### Added

- feat: add K-means clustering implementation using scikit-learn and rename existing kmeans_clusterer to kmeans_lite_clusterer

## [0.0.18] - 2025-07-09

### Added

- feat: add __repr__ method to PartializedFuncs for better representation and debug output

## [0.0.17] - 2025-07-09

### Added

- feat: enhance planarizer with dynamic projection and add cluster_labels data store

## [0.0.16] - 2025-07-01

### Added

- feat: implement component store management and standard component loading

## [0.0.15] - 2025-07-01

### Added

- feat: enhance store utilities with optional base store wrapping and extension-based mall creation

### Changed

- refactor: rename parameter in simple_text_embedder function for clarity; remove imbed_dog.py example file

### Fixed

- fix: add 'segments' to data_store_names and improve local mall validation

## [0.0.14] - 2025-06-28

### Fixed

- fix: update import path for DOG and ADOG classes in imbed_dog.py

## [0.0.13] - 2025-06-24

- Ignore tests/test_imbed_project.py in pytest
- Ignore test_imbed_project.py in tests
- imbed_project_w_updateble_segments
- Add initial implementation of embedding tests, segmentation utilities, and project integration tests

### Added

- feat: add 'au' to install_requires for enhanced functionality
- feat: update get_local_mall to use mk_json_local_store for segments and improve mall structure; add imbed_dog.py for DOG/ADOG instance creation
- feat: enhance constant_vectorizer with presleep option and update Project class for improved mall handling
- feat: update Project class to use mk_dill_local_store for segments and clusters, modify demo notebook for improved output clarity
- feat: update constant_vectorizer to handle dict and list inputs, default async mode to sync for reliability
- feat: add async backend support and improve embedding computation in Project class See https://github.com/thorwhalen/imbed/discussions/12#discussioncomment-13483023 for more info
- feat: enhance Project and utility functions for improved store management and async support
- feat: enhance Project class with async computation support and improve embedding management

### Changed

- refactor: remove Sphinx documentation files and update Project class for async embedding

## [0.0.12] - 2025-03-28

### Added

- feat: Some iterfaces and examples for batch embeddings

## [0.0.11] - 2025-03-25

### Docs

- docs: add docstrings for ClusterLabeler class and cluster_labeler function

## [0.0.10] - 2025-03-25

### Added

- feat: cluster_labeler

### Changed

- refactor: planarize

## [0.0.9] - 2025-03-24

### Added

- feat: add_default_key

## [0.0.8] - 2025-03-21

### Added

- feat: add module docstrings for clusterization, planarization, and vectorization components
- feat: add components module for imbed applications

## [0.0.7] - 2025-03-21

### Added

- feat: implement vectorization components and async utility functions

### Fixed

- fix: update doctest examples to use correct function names and suppress import errors

## [0.0.6] - 2025-03-18

- merged
- Update ci.yml
- chore: refactoring imbed stuff
- 0.0.4:
- chore: add scikit-learn deps and misc
- 0.0.3:
- more cleanup
- chore: use lkj instead of vectorizing utils
- chore: add lkj to dependencies
- chore: moving prep stuff to imbed_data_prep
- chore: move mdat dataprep modules to imbed_data_prep
- chore: various updates
- developing imbed further...
- chore: preget->key_ingress
- on going WIP
- 0.0.2: feat: first commit

### Added

- feat: more flexible and only numpy-dependent cosine_similarity
- feat: add tsne
- feat: add transpose_iterable utility and update planar_embeddings to support preprocessing
- feat: add distance metric parameter to planar_embeddings_func
- feat: add fullpath_factory utility and update saves_join implementation
- feat: more EmbeddingBatchManager methods
- feat: more on batch embeddings interface
- feat: compute_embeddings_in_bulk
- feat: simple_embedding_vectorizer = simple_semantic_features  alias
- feat: work on add_extension_codec and add tabled as dep
- feat: ClusterLabeler
- feat: Changed default in fixed_step_chunker: now, return_tail = True
- feat: clustering tools
- feat: alias_based_mapping
- feat: using_ai_to_get_data_descriptions
- feat: misc refactoring
- feat: more type aliases
- feat: compute_and_save_embeddings_pca and compute_and_save_dbscan
- feat: prompt_injections
- feat: lmsys ai
- feat: segmentMapping and pipelines
- feat: umap_2d_embeddings
- feat: included download urls
- feat: arxiv
- feat: add gitignore

### Changed

- refactor: Renamed SegmentMapping (class) to SegmentStore
- refactor: extension_base_wrap -> extension_based_wrap
- refactor: alias_based_mapping -> match_aliases and other variable name changes
- refactor: SegmentKey -> KeyAndIntervalSegmentKey
- refactor: move arxiv to xv
- refactor: arxiv to it's own notebook

### Fixed

- fix: ensure cosine_similarity returns a float in example usage
- fix: import error
- fix: .npy extension encoder
- fix: clog

### Docs

- docs: add todos for future work
