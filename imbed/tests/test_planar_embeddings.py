"""``planar_embeddings`` on each kind of embedding function (thorwhalen/imbed#15)."""

import numpy as np
import pytest

from imbed.util import planar_embeddings


def _embeddings(n, *, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    return {f"k{i}": rng.random(dim) for i in range(n)}


@pytest.mark.parametrize("kind", ["pca", "umap"])
def test_planar_embeddings_maps_every_key_to_a_pair(kind):
    if kind == "umap":
        pytest.importorskip("umap")
    kd = _embeddings(40)
    xy = planar_embeddings(kd, embeddings_func=kind)
    assert list(xy) == list(kd)
    assert all(len(pair) == 2 for pair in xy.values())


def test_pca_is_deterministic():
    kd = _embeddings(10)
    first = planar_embeddings.pca(kd)
    assert planar_embeddings.pca(kd) == first
