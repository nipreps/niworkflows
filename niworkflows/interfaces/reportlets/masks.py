# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from ..._deprecated import raise_moved_to_nireports


raise_moved_to_nireports(
    'niworkflows.interfaces.reportlets.masks',
    (
        'nireports.interfaces.reporting.masks.BETRPT',
        'nireports.interfaces.reporting.masks.BrainExtractionRPT',
        'nireports.interfaces.reporting.masks.ACompCorRPT',
        'nireports.interfaces.reporting.masks.TCompCorRPT',
        'nireports.interfaces.reporting.masks.SimpleShowMaskRPT',
        'nireports.interfaces.reporting.masks.ROIsPlot',
    ),
)
