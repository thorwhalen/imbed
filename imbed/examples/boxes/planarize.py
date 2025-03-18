"""
Planarization tools.

A bunch of tools to make planar projectors and manage projections.
"""

from functools import partial
from operator import itemgetter, attrgetter
from typing import Callable, Union, Mapping, MutableMapping, KT, VT, Iterable, Tuple

from i2 import Sig, FuncFactory, Pipe
from lkj import CallOnError
from imbed.imbed_types import (
    SegmentKey,
    Vector,
    Vectors,
    PlanarVector,
    PlanarVectors,
    SingularPlanarProjector,
    BatchPlanarProjector,
    PlanarProjector,
)


# -------------------------------------------------------------------------------------
# General Utils

warn_about_import_errors = CallOnError(ImportError, ModuleNotFoundError, on_error=print)


def _parametrized_fit_transform(cls, **kwargs):
    return cls(**kwargs).fit_transform


def mk_parametrized_fit_transform(cls, **kwargs):
    """
    Create a function that instantiates a class with the given keyword arguments and
    calls fit_transform on it. Makes sure that the signature is specific and correct.
    """
    sig = Sig(cls).ch_defaults(**kwargs)
    return sig(partial(_parametrized_fit_transform, cls, **kwargs))


def mk_parametrized_fit_transform_factory(cls):
    """
    Create a factory of parametrized_fit_transform functions.
    """
    return partial(mk_parametrized_fit_transform, cls)


TargetMapping = MutableMapping[KT, VT]


def mk_overwrite_boolean_function(overwrite: bool):
    if callable(overwrite):

        def should_overwrite(k):
            return overwrite(k)

    else:
        assert isinstance(overwrite, bool), f"Invalid overwrite value: {overwrite}"

        def should_overwrite(k):
            return overwrite

    return should_overwrite


def conditional_update(
    update_this: MutableMapping[KT, VT],
    with_this: Mapping[KT, VT],
    *,
    overwrite: Union[bool, Callable[[TargetMapping, KT], bool]] = False,
):
    """
    Update a dictionary with another dictionary, with more control over the update.

    For example, if overwrite is False, then only keys that are not already in the
    dictionary will be added. If overwrite is True, then all keys will be added.
    If overwrite is a Callable, then it will be called with the key to determine
    whether to overwrite the key or not.
    """
    if callable(overwrite):

        def should_overwrite(k):
            return overwrite(k)

    else:
        assert isinstance(overwrite, bool), f"Invalid overwrite value: {overwrite}"

        def should_overwrite(k):
            return overwrite

    for k, v in with_this.items():
        if k not in update_this or should_overwrite(k):
            update_this[k] = v


def conditional_update_with_factory_commands(
    update_this: MutableMapping[KT, VT],
    with_these_commands: Iterable[Tuple[KT, Callable, dict]],
    *,
    overwrite: Union[bool, Callable[[KT], bool]] = False,
):
    """
    Update a dictionary with a sequence of key, factory, kwargs commands.
    """
    for name, factory, factory_kwargs in with_these_commands:
        if name not in update_this:
            update_this[name] = factory(**factory_kwargs)


# -------------------------------------------------------------------------------------
# setup stores


DFLT_DISTANCE_METRIC = 'cosine'


def fill_planarizer_stores(*, planarizer_factories, planarizers):
    """
    Fill the planarizer mall with planarizers from various libraries.
    """
    with warn_about_import_errors:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import normalize, FunctionTransformer
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.manifold import MDS
        from sklearn.pipeline import Pipeline

        # Update with default factories
        f = mk_parametrized_fit_transform_factory

        @Sig(PCA)
        def mk_normalized_pca(**kwargs):
            return Pipeline([('normalize', l2_normalization), ('pca', PCA(**kwargs))])

        default_factory_commands = [
            ('pca', f(PCA), {'n_components': 2}),
            ('normalized_pca', mk_normalized_pca, {'n_components': 2}),
            ('tsne', f(TSNE), {'n_components': 2, 'metric': DFLT_DISTANCE_METRIC}),
            ('lda', f(LinearDiscriminantAnalysis), {'n_components': 2}),
            ('mds', f(MDS), {'n_components': 2, 'metric': DFLT_DISTANCE_METRIC}),
        ]

        # Note: Here, the func are factory factories. They take func_kwargs and return a fit_transform_factory
        for name, func, func_kwargs in default_factory_commands:
            if name not in planarizer_factories:
                planarizer_factories[name] = func(**func_kwargs)

        # Update with default planarizers
        default_planarizer_commands = [
            (name, func(**func_kwargs), {})
            for name, func, func_kwargs in default_factory_commands
        ]

        for name, func, func_kwargs in default_planarizer_commands:
            if name not in planarizers:
                planarizers[name] = func

        l2_normalization = FunctionTransformer(
            lambda X: normalize(X, norm='l2'), validate=True
        )

        normalize_pca = mk_normalized_pca(n_components=2).fit_transform

        planarizers.update(
            normalized_pca=normalize_pca,
            tsne=TSNE(n_components=2, metric=DFLT_DISTANCE_METRIC).fit_transform,
        )

        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        planarizers.update(
            lda=LinearDiscriminantAnalysis(n_components=2).fit_transform,
        )

        from sklearn.manifold import MDS

        planarizers.update(
            mds=MDS(n_components=2, metric=DFLT_DISTANCE_METRIC).fit_transform,
        )

        default_planarizer_commands = [
            ('normalized_pca', PCA, {'n_components': 2}),
        ]

        for name, func, func_kwargs in default_planarizer_commands:
            if name not in planarizers:
                planarizers[name] = func(**func_kwargs)

    with warn_about_import_errors:
        from umap import UMAP

        planarizers.update(
            umap=UMAP(n_components=2, metric=DFLT_DISTANCE_METRIC).fit_transform,
        )

    with warn_about_import_errors:
        import ncvis

        planarizers.update(
            ncvis=ncvis.NCVis(d=2, distance=DFLT_DISTANCE_METRIC).fit_transform,
        )


# -------------------------------------------------------------------------------------
# Example Usage with dict stores


def get_dict_mall():
    def dflt_named_store_factory(name=None):
        return dict()

    planarizer_mall = dict(
        planarizer_factories=dflt_named_store_factory('planarize_factories'),
        planarizers=dflt_named_store_factory('planarizers'),
    )

    planarizer_factories = planarizer_mall['planarizer_factories']
    planarizers = planarizer_mall['planarizers']

    fill_planarizer_stores(
        planarizers=planarizers,
        planarizer_factories=planarizer_factories,
    )

    return planarizer_mall
