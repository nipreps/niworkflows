# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from .._deprecated import raise_moved_to_nireports

raise_moved_to_nireports(
    'niworkflows.viz.utils',
    (
        'nireports.reportlets.utils.compose_view',
        'nireports.reportlets.utils.cuts_from_bbox',
        'nireports.reportlets.utils.extract_svg',
        'nireports.reportlets.utils.robust_set_limits',
        'nireports.reportlets.utils.svg2str',
        'nireports.reportlets.utils.svg_compress',
        'nireports.reportlets.utils.transform_to_2d',
        'nireports.reportlets.utils.SVGNS',
        'nireports.reportlets.mosaic.plot_registration',
        'nireports.reportlets.mosaic.plot_segs',
        'nireports.reportlets.xca.plot_melodic_components',
    ),
)
