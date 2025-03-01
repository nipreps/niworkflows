# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""NeuroImaging Workflows (NIWorkflows) is a selection of image processing workflows."""

import logging

from acres import Loader

from .__about__ import __copyright__, __credits__, __packagename__

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = '0+unknown'


__all__ = [
    'NIWORKFLOWS_LOG',
    '__copyright__',
    '__credits__',
    '__packagename__',
    '__version__',
    'load_resource',
]

NIWORKFLOWS_LOG = logging.getLogger(__packagename__)
NIWORKFLOWS_LOG.setLevel(logging.INFO)

try:
    import matplotlib as mpl

    mpl.use('Agg')
except ImportError:
    pass

load_resource = Loader(__package__)
