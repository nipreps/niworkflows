# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Visualization tools

"""
import numpy as np
import pandas as pd

from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    File, BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
)
from ..viz.plots import (
    fMRIPlot, compcor_variance_plot, confounds_correlation_plot
)


class FMRISummaryInputSpec(BaseInterfaceInputSpec):
    in_func = File(exists=True, mandatory=True, desc='')
    in_mask = File(exists=True, mandatory=True, desc='')
    in_segm = File(exists=True, mandatory=True, desc='')
    in_spikes_bg = File(exists=True, mandatory=True, desc='')
    fd = File(exists=True, mandatory=True, desc='')
    fd_thres = traits.Float(0.2, usedefault=True, desc='')
    dvars = File(exists=True, mandatory=True, desc='')
    outliers = File(exists=True, mandatory=True, desc='')
    tr = traits.Either(None, traits.Float, usedefault=True,
                       desc='the TR')


class FMRISummaryOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='written file path')


class FMRISummary(SimpleInterface):
    """
    Prepare a fMRI summary plot for the report.
    """
    input_spec = FMRISummaryInputSpec
    output_spec = FMRISummaryOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = fname_presuffix(
            self.inputs.in_func,
            suffix='_fmriplot.svg',
            use_ext=False,
            newpath=runtime.cwd)

        dataframe = pd.DataFrame({
            'outliers': np.loadtxt(
                self.inputs.outliers, usecols=[0]).tolist(),
            # Pick non-standardize dvars (col 1)
            # First timepoint is NaN (difference)
            'DVARS': [np.nan] + np.loadtxt(
                self.inputs.dvars, skiprows=1, usecols=[1]).tolist(),
            # First timepoint is zero (reference volume)
            'FD': [0.0] + np.loadtxt(
                self.inputs.fd, skiprows=1, usecols=[0]).tolist(),
        })

        fig = fMRIPlot(
            self.inputs.in_func,
            mask_file=self.inputs.in_mask,
            seg_file=self.inputs.in_segm,
            spikes_files=[self.inputs.in_spikes_bg],
            tr=self.inputs.tr,
            data=dataframe[['outliers', 'DVARS', 'FD']],
            units={'outliers': '%', 'FD': 'mm'},
            vlines={'FD': [self.inputs.fd_thres]},
        ).plot()
        fig.savefig(self._results['out_file'], bbox_inches='tight')
        return runtime


class CompCorVariancePlotInputSpec(BaseInterfaceInputSpec):
    metadata_files = traits.List(File(exists=True), mandatory=True,
                                 desc='List of files containing component '
                                      'metadata')
    metadata_sources = traits.List(traits.Str,
                                   desc='List of names of decompositions '
                                        '(e.g., aCompCor, tCompCor) yielding '
                                        'the arguments in `metadata_files`')
    variance_thresholds = traits.Tuple(
        traits.Float(0.5), traits.Float(0.7), traits.Float(0.9),
        usedefault=True, desc='Levels of explained variance to include in '
                              'plot')
    out_file = traits.Either(None, File, value=None, usedefault=True,
                             desc='Path to save plot')


class CompCorVariancePlotOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Path to saved plot')


class CompCorVariancePlot(SimpleInterface):
    """
    Plot the number of components necessary to explain the specified levels
    of variance in the data.
    """
    input_spec = CompCorVariancePlotInputSpec
    output_spec = CompCorVariancePlotOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.out_file is None:
            self._results['out_file'] = fname_presuffix(
                self.inputs.metadata_files[0],
                suffix='_compcor.svg',
                use_ext=False,
                newpath=runtime.cwd)
        else:
            self._results['out_file'] = self.inputs.out_file
        compcor_variance_plot(
            metadata_files=self.inputs.metadata_files,
            metadata_sources=self.inputs.metadata_sources,
            output_file=self._results['out_file'],
            varexp_thresh=self.inputs.variance_thresholds
        )
        return runtime
