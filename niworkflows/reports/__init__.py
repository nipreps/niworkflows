# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from .._deprecated import raise_moved_to_nireports


raise_moved_to_nireports(
    'niworkflows.reports',
    (
        'nireports.assembler.report.Report',
        'nireports.assembler.report.SubReport',
        'nireports.assembler.reportlet.Reportlet',
        'nireports.assembler.misc.Element',
        'nireports.assembler.tools.generate_reports',
        'nireports.assembler.tools.run_reports',
    ),
)
