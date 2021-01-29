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
)

LOGGER = logging.getLogger("nipype.interface")


class _NonsteadyStatesDetectorInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="BOLD fMRI timeseries")
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
    n_dummy = traits.Int(desc="number of volumes identified as nonsteady states")


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
            self._results["n_dummy"] = 1
            return runtime

        from nipype.algorithms.confounds import is_outlier

        global_signal = np.mean(
            np.asanyarray(img.dataobj[..., : self.inputs.n_volumes]), axis=(0, 1, 2)
        )

        n_discard = is_outlier(global_signal)
        self._results["t_mask"][:n_discard] = [True] * n_discard
        self._results["n_dummy"] = n_discard

        return runtime
