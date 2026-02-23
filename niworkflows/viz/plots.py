# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from .._deprecated import raise_moved_to_nireports


raise_moved_to_nireports(
    'niworkflows.viz.plots',
    (
        'nireports.reportlets.nuisance.confoundplot',
        'nireports.reportlets.nuisance.confounds_correlation_plot',
        'nireports.reportlets.nuisance.plot_carpet',
        'nireports.reportlets.nuisance.spikesplot',
        'nireports.reportlets.nuisance.spikesplot_cb',
        'nireports.reportlets.surface.cifti_surfaces_plot',
        'nireports.reportlets.xca.compcor_variance_plot',
        'nireports.reportlets.modality.func.fMRIPlot',
    ),
)
