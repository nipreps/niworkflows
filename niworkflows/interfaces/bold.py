# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Utilities for BOLD fMRI imaging."""
import numpy as np
import nibabel as nb
from nipype import logging
from nipype.interfaces.base import (
    traits,
    TraitedSpec,
    BaseInterfaceInputSpec,
    SimpleInterface,
    File,
    isdefined,
)

LOGGER = logging.getLogger("nipype.interface")


class _NonsteadyStatesDetectorInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="BOLD fMRI timeseries")
    n_dummy_scans = traits.Int(
        desc="override detection and just return a mask with n_dummy_scans masked in the beginning"
    )
    n_volumes = traits.Range(
        value=50,
        low=10,
        high=200,
        usedefault=True,
        desc="drop volumes in 4D image beyond this timepoint",
    )


class _NonsteadyStatesDetectorOutputSpec(TraitedSpec):
    t_mask = traits.List(
        traits.Bool, desc="list of nonsteady-states (True) and stable (False) volumes"
    )


class NonsteadyStatesDetector(SimpleInterface):
    """Detect initial non-steady states in BOLD fMRI timeseries."""

    input_spec = _NonsteadyStatesDetectorInputSpec
    output_spec = _NonsteadyStatesDetectorOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)

        ntotal = img.shape[-1] if img.dataobj.ndim == 4 else 1

        self._results["t_mask"] = [False] * ntotal

        if ntotal == 1:
            self._results["t_mask"] = [True]
            return runtime

        if isdefined(self.inputs.n_dummy_scans):
            ndummy = min(ntotal, self.inputs.n_dummy_scans)
            self._results["t_mask"][:ndummy] = [True] * ndummy
            return runtime

        from nipype.algorithms.confounds import is_outlier

        global_signal = np.mean(
            np.asanyarray(img.dataobj[..., : self.inputs.n_volumes]), axis=(0, 1, 2)
        )
        self._results["t_mask"] = [bool(i) for i in is_outlier(global_signal)]
        return runtime
