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
"""GradUnwarp interface."""
import numpy as np
from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits,
    TraitedSpec,
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    OutputMultiObject,
    InputMultiObject,
)


class _GradUnwarpInputSpec(BaseInterfaceInputSpec):
    infile = File(exists=True, mandatory=True, desc="input image to be corrected")
    gradfile = File(exists=True, default=None, desc="gradient file")
    coeffile = File(exists=True, default=None, desc="coefficients file")
    outfile = File(desc="output corrected image")
    vendor = traits.Enum("siemens", "ge", usedefault=True, desc="scanner vendor")
    warp = traits.Bool(desc="warp a volume (as opposed to unwarping)")
    nojac = traits.Bool(desc="Do not perform Jacobian intensity correction")

    fovmin = traits.Float(desc="the minimum extent of harmonics evaluation grid in meters")
    fovmax = traits.Float(desc="the maximum extent of harmonics evaluation grid in meters")
    order = traits.Int(min=1, max=4, usedefault=True, desc="the order of interpolation(1..4) where 1 is linear - default")

class _GradUnwarpOutputSpec(TraitedSpec):
    corrected_file = File(desc="input images corrected")
    warp_file = File(desc="absolute warp file")


class GradUnwarp(SimpleInterface):
    input_spec = _GradUnwarpInputSpec
    output_spec = _GradUnwarpOutputSpec

    def _run_interface(self, runtime):

        from gradunwarp.core.gradient_unwarp import GradientUnwarpRunner
        if not self.inputs.outfile:
            self.inputs.outfile = fname_presuffix(
                self.inputs.infile,
                suffix='_gradunwarped',
                newpath=runtime.cwd
                )
        gur = GradientUnwarpRunner(self.inputs)
        gur.run()
        gur.write()
        del gur

        self._results["corrected_file"] = self.inputs.outfile
        self._results["warp_file"] = fname_presuffix(
            "fullWarp_abs.nii.gz",
            newpath=runtime.cwd
            )
        return runtime
