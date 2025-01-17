# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import warnings

from .plots import plot_carpet
from .utils import SVGNS

msg = (
    'Niworkflows will be deprecating visualizations in favor of a standalone library "nireports".'
)

warnings.warn(msg, PendingDeprecationWarning, stacklevel=2)

__all__ = ['SVGNS', 'plot_carpet']
