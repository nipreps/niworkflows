# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from ..._deprecated import raise_moved_to_nireports


raise_moved_to_nireports(
    'niworkflows.interfaces.reportlets.segmentation',
    (
        'nireports.interfaces.reporting.segmentation.FASTRPT',
        'nireports.interfaces.reporting.segmentation.ReconAllRPT',
        'nireports.interfaces.reporting.segmentation.MELODICRPT',
        'nireports.interfaces.reporting.segmentation.ICA_AROMARPT',
    ),
)
