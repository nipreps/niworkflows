# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""NeuroImaging Workflows (NIWorkflows) is a selection of image processing workflows."""
import logging

from .__about__ import __packagename__, __copyright__, __credits__
from .data import Loader
try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0+unknown"


__all__ = [
    "__version__",
    "__packagename__",
    "__copyright__",
    "__credits__",
    "NIWORKFLOWS_LOG",
    "load_resource",
]

NIWORKFLOWS_LOG = logging.getLogger(__packagename__)
NIWORKFLOWS_LOG.setLevel(logging.INFO)

try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    pass

load_resource = Loader(__package__)
