# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Visualization tools

"""
import numpy as np

from ..nipype.utils.filemanip import fname_presuffix
from ..nipype.interfaces.base import (
    File, BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
)
from ..viz.plots import fMRIPlot


class FMRISummaryInputSpec(BaseInterfaceInputSpec):
    in_func = File(exists=True, mandatory=True, desc='')
    in_mask = File(exists=True, mandatory=True, desc='')
    in_segm = File(exists=True, mandatory=True, desc='')
    in_spikes_bg = File(exists=True, mandatory=True, desc='')
    fd = File(exists=True, mandatory=True, desc='')
    fd_thres = traits.Float(0.2, usedefault=True, desc='')
    dvars = File(exists=True, mandatory=True, desc='')
    outliers = File(exists=True, mandatory=True, desc='')


class FMRISummaryOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='written file path')


class FMRISummary(SimpleInterface):
    """
    Copy the x-form matrices from `hdr_file` to `out_file`.
    """
    input_spec = FMRISummaryInputSpec
    output_spec = FMRISummaryOutputSpec

    def _run_interface(self, runtime):
        out_name = fname_presuffix(self.inputs.in_func,
                                   suffix='_fmriplot.svg',
                                   use_ext=False,
                                   newpath=runtime.cwd)

        # Compose plot
        self._results['out_file'] = _big_plot(
            self.inputs.in_func,
            self.inputs.in_mask,
            self.inputs.in_segm,
            self.inputs.in_spikes_bg,
            self.inputs.fd,
            self.inputs.fd_thres,
            self.inputs.dvars,
            self.inputs.outliers,
            out_name,
        )
        return runtime


def _big_plot(in_func, in_mask, in_segm, in_spikes_bg,
              fd, fd_thres, dvars, outliers, out_file,
              title='fMRI Summary plot'):
    myplot = fMRIPlot(
        in_func, in_mask, in_segm, title=title)
    myplot.add_spikes(np.loadtxt(in_spikes_bg), zscored=False)

    # Add AFNI outliers plot
    myplot.add_confounds([np.nan] + np.loadtxt(outliers, usecols=[0]).tolist(),
                         {'name': 'outliers', 'units': '%', 'normalize': False,
                          'ylims': (0.0, None)})

    # Pick non-standardize dvars
    myplot.add_confounds([np.nan] + np.loadtxt(dvars, skiprows=1,
                                               usecols=[1]).tolist(),
                         {'name': 'DVARS', 'units': None, 'normalize': False})

    # Add FD
    myplot.add_confounds([np.nan] + np.loadtxt(fd, skiprows=1,
                                               usecols=[0]).tolist(),
                         {'name': 'FD', 'units': 'mm', 'normalize': False,
                          'cutoff': [fd_thres], 'ylims': (0.0, fd_thres)})
    myplot.plot()
    myplot.fig.savefig(out_file, bbox_inches='tight')
    myplot.fig.clf()
    return out_file
