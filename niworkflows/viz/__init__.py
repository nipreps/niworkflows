# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from .._deprecated import raise_moved_to_nireports


raise_moved_to_nireports(
    'niworkflows.viz',
    (
        'nireports.reportlets.modality.func.plot_carpet',
        'nireports.reportlets.utils.SVGNS',
        'nireports.reportlets',
    ),
)
