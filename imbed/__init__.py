"""Tools for imbeddings"""

from imbed.segmentation import fixed_step_chunker, SegmentMapping
from imbed.util import (
    cosine_similarity,
    planar_embeddings,
    umap_2d_embeddings,
    extension_base_wrap,
    add_extension_codec,
    match_aliases,
    get_codec_mappings,
)
