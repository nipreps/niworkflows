# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
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
    nonnegative = traits.Bool(True, usedefault=True,
                              desc="whether image voxels must be nonnegative")
    n_volumes = traits.Range(
        value=40,
        low=10,
        high=200,
        usedefault=True,
        desc="drop volumes in 4D image beyond this timepoint",
    )
    zero_dummy_masked = traits.Range(
        value=20,
        low=2,
        high=40,
        usedefault=True,
        desc="number of timepoints to average when the number of dummies is zero"
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
        t_mask = np.zeros((ntotal,), dtype=bool)

        if ntotal == 1:
            self._results["t_mask"] = [True]
            self._results["n_dummy"] = 1
            return runtime

        from nipype.algorithms.confounds import is_outlier

        data = img.get_fdata(dtype="float32")[..., :self.inputs.n_volumes]
        # Data can come with outliers showing very high numbers - preemptively prune
        data = np.clip(
            data,
            a_min=0.0 if self.inputs.nonnegative else np.percentile(data, 0.2),
            a_max=np.percentile(data, 99.8),
        )
        self._results["n_dummy"] = is_outlier(np.mean(data, axis=(0, 1, 2)))

        start = 0
        stop = self._results["n_dummy"]
        if stop < 2:
            stop = min(ntotal, self.inputs.n_volumes)
            start = max(0, stop - self.inputs.zero_dummy_masked)

        t_mask[start:stop] = True
        self._results["t_mask"] = t_mask.tolist()

        return runtime
