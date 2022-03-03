# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
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
"""Mathematical morphology operations as nipype interfaces."""
from pathlib import Path
import numpy as np
import nibabel as nb

from nipype.interfaces.base import (
    traits,
    TraitedSpec,
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
)


class _BinaryDilationInputSpec(BaseInterfaceInputSpec):
    in_mask = File(exists=True, mandatory=True, desc="input mask")
    radius = traits.Int(2, usedefault=True, desc="Radius of dilation")


class _BinaryDilationOutputSpec(TraitedSpec):
    out_mask = File(exists=False, desc="dilated mask")


class BinaryDilation(SimpleInterface):
    """Binary dilation of a mask."""

    input_spec = _BinaryDilationInputSpec
    output_spec = _BinaryDilationOutputSpec

    def _run_interface(self, runtime):
        # Open files
        mask_img = nb.load(self.inputs.in_mask)
        maskdata = np.bool_(mask_img.dataobj)

        # Obtain dilated brainmask
        dilated = image_binary_dilation(
            maskdata,
            radius=self.inputs.radius,
        )
        out_file = str((Path(runtime.cwd) / "dilated_mask.nii.gz").absolute())
        out_img = mask_img.__class__(dilated, mask_img.affine, mask_img.header)
        out_img.set_data_dtype("uint8")
        out_img.to_filename(out_file)
        self._results["out_mask"] = out_file
        return runtime


class _BinarySubtractInputSpec(BaseInterfaceInputSpec):
    in_base = File(exists=True, mandatory=True, desc="input base mask")
    in_subtract = File(exists=True, mandatory=True, desc="input subtract mask")


class _BinarySubtractionOutputSpec(TraitedSpec):
    out_mask = File(exists=False, desc="subtracted mask")


class BinarySubtraction(SimpleInterface):
    """Binary subtraction of two masks."""

    input_spec = _BinarySubtractInputSpec
    output_spec = _BinarySubtractionOutputSpec

    def _run_interface(self, runtime):
        # Subtract mask from base
        base_img = nb.load(self.inputs.in_base)
        data = np.bool_(base_img.dataobj)
        data[np.bool_(nb.load(self.inputs.in_subtract).dataobj)] = False

        out_file = str((Path(runtime.cwd) / "subtracted_mask.nii.gz").absolute())
        out_img = base_img.__class__(
            data,
            base_img.affine,
            base_img.header
        )
        out_img.set_data_dtype("uint8")
        out_img.to_filename(out_file)
        self._results["out_mask"] = out_file
        return runtime


def image_binary_dilation(in_mask, radius=2):
    """
    Dilate the input binary mask.

    Parameters
    ----------
    in_mask: :obj:`numpy.ndarray`
        A 3D binary array.
    radius: :obj:`int`, optional
        The radius of the ball-shaped footprint for dilation of the mask.
    """
    from scipy import ndimage as ndi
    from skimage.morphology import ball

    return ndi.binary_dilation(in_mask.astype(bool), ball(radius)).astype(int)
