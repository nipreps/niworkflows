# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Visualization tools."""
import numpy as np
import pandas as pd

from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    File,
    BaseInterfaceInputSpec,
    TraitedSpec,
    SimpleInterface,
    traits,
    isdefined,
)
from ..viz.plots import fMRIPlot, compcor_variance_plot, confounds_correlation_plot


class _FMRISummaryInputSpec(BaseInterfaceInputSpec):
    in_func = File(exists=True, mandatory=True, desc="")
    in_mask = File(exists=True, desc="")
    in_segm = File(exists=True, desc="")
    in_spikes_bg = File(exists=True, mandatory=True, desc="")
    fd = File(exists=True, mandatory=True, desc="")
    fd_thres = traits.Float(0.2, usedefault=True, desc="")
    dvars = File(exists=True, mandatory=True, desc="")
    outliers = File(exists=True, mandatory=True, desc="")
    tr = traits.Either(None, traits.Float, usedefault=True, desc="the TR")


class _FMRISummaryOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="written file path")


class FMRISummary(SimpleInterface):
    """Prepare an fMRI summary plot for the report."""

    input_spec = _FMRISummaryInputSpec
    output_spec = _FMRISummaryOutputSpec

    def _run_interface(self, runtime):
        self._results["out_file"] = fname_presuffix(
            self.inputs.in_func,
            suffix="_fmriplot.svg",
            use_ext=False,
            newpath=runtime.cwd,
        )

        dataframe = pd.DataFrame(
            {
                "outliers": np.loadtxt(self.inputs.outliers, usecols=[0]).tolist(),
                # Pick non-standardize dvars (col 1)
                # First timepoint is NaN (difference)
                "DVARS": [np.nan]
                + np.loadtxt(self.inputs.dvars, skiprows=1, usecols=[1]).tolist(),
                # First timepoint is zero (reference volume)
                "FD": [0.0]
                + np.loadtxt(self.inputs.fd, skiprows=1, usecols=[0]).tolist(),
            }
        )

        fig = fMRIPlot(
            self.inputs.in_func,
            mask_file=self.inputs.in_mask if isdefined(self.inputs.in_mask) else None,
            seg_file=self.inputs.in_segm if isdefined(self.inputs.seg_file) else None,
            spikes_files=[self.inputs.in_spikes_bg],
            tr=self.inputs.tr,
            data=dataframe[["outliers", "DVARS", "FD"]],
            units={"outliers": "%", "FD": "mm"},
            vlines={"FD": [self.inputs.fd_thres]},
        ).plot()
        fig.savefig(self._results["out_file"], bbox_inches="tight")
        return runtime


class _CompCorVariancePlotInputSpec(BaseInterfaceInputSpec):
    metadata_files = traits.List(
        File(exists=True),
        mandatory=True,
        desc="List of files containing component " "metadata",
    )
    metadata_sources = traits.List(
        traits.Str,
        desc="List of names of decompositions "
        "(e.g., aCompCor, tCompCor) yielding "
        "the arguments in `metadata_files`",
    )
    variance_thresholds = traits.Tuple(
        traits.Float(0.5),
        traits.Float(0.7),
        traits.Float(0.9),
        usedefault=True,
        desc="Levels of explained variance to include in " "plot",
    )
    out_file = traits.Either(
        None, File, value=None, usedefault=True, desc="Path to save plot"
    )


class _CompCorVariancePlotOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Path to saved plot")


class CompCorVariancePlot(SimpleInterface):
    """Plot the number of components necessary to explain the specified levels of variance."""

    input_spec = _CompCorVariancePlotInputSpec
    output_spec = _CompCorVariancePlotOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.out_file is None:
            self._results["out_file"] = fname_presuffix(
                self.inputs.metadata_files[0],
                suffix="_compcor.svg",
                use_ext=False,
                newpath=runtime.cwd,
            )
        else:
            self._results["out_file"] = self.inputs.out_file
        compcor_variance_plot(
            metadata_files=self.inputs.metadata_files,
            metadata_sources=self.inputs.metadata_sources,
            output_file=self._results["out_file"],
            varexp_thresh=self.inputs.variance_thresholds,
        )
        return runtime


class _ConfoundsCorrelationPlotInputSpec(BaseInterfaceInputSpec):
    confounds_file = File(
        exists=True, mandatory=True, desc="File containing confound regressors"
    )
    out_file = traits.Either(
        None, File, value=None, usedefault=True, desc="Path to save plot"
    )
    reference_column = traits.Str(
        "global_signal",
        usedefault=True,
        desc="Column in the confound file for "
        "which all correlation magnitudes "
        "should be ranked and plotted",
    )
    columns = traits.List(
        traits.Str,
        desc="Filter out all regressors not found in this list."
    )
    max_dim = traits.Int(
        20,
        usedefault=True,
        desc="Maximum number of regressors to include in "
        "plot. Regressors with highest magnitude of "
        "correlation with `reference_column` will be "
        "selected.",
    )


class _ConfoundsCorrelationPlotOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Path to saved plot")


class ConfoundsCorrelationPlot(SimpleInterface):
    """Plot the correlation among confound regressors."""

    input_spec = _ConfoundsCorrelationPlotInputSpec
    output_spec = _ConfoundsCorrelationPlotOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.out_file is None:
            self._results["out_file"] = fname_presuffix(
                self.inputs.confounds_file,
                suffix="_confoundCorrelation.svg",
                use_ext=False,
                newpath=runtime.cwd,
            )
        else:
            self._results["out_file"] = self.inputs.out_file
        confounds_correlation_plot(
            confounds_file=self.inputs.confounds_file,
            columns=self.inputs.columns if isdefined(self.inputs.columns) else None,
            max_dim=self.inputs.max_dim,
            output_file=self._results["out_file"],
            reference=self.inputs.reference_column,
        )
        return runtime
