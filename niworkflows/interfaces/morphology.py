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
""" Handling brain mask"""

from nipype.interfaces.base import (
    traits,
    TraitedSpec,
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
)


class _CrownMaskInputSpec(BaseInterfaceInputSpec):
    in_segm = File(
        exists=True, mandatory=True, position=0, desc="Input discrete segmentation of the brain."
    )
    in_brainmask = File(exists=True, mandatory=True, position=1, desc="Brain mask.")
    radius = traits.Int(default_value=2, usedefault=True, desc="Radius of dilation")


class _CrownMaskOutputSpec(TraitedSpec):
    out_mask = File(exists=False, desc="Crown mask")


class CrownMask(SimpleInterface):
    """Dilate brain mask for computing the crown mask."""

    input_spec = _CrownMaskInputSpec
    output_spec = _CrownMaskOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nb
        import numpy as np
        from pathlib import Path

        # Open files
        segm_img = nb.load(self.inputs.in_segm)
        brainmask_img = nb.load(self.inputs.in_brainmask)

        segm = np.bool_(segm_img.dataobj)
        brainmask = np.bool_(brainmask_img.dataobj)

        # Obtain dilated brainmask
        crown_mask, func_seg_mask = get_dilated_brainmask(
            seg_mask=segm,
            brainmask=brainmask,
            radius=self.inputs.radius,
        )
        # Remove the brain from the crown mask
        crown_mask[func_seg_mask] = False
        crown_file = str((Path(runtime.cwd) / "crown_mask.nii.gz").absolute())
        nii = nb.Nifti1Image(
            crown_mask, brainmask_img.affine, brainmask_img.header
        )
        nii.set_data_dtype("uint8")
        nii.to_filename(crown_file)
        self._results["out_mask"] = crown_file

        return runtime


def get_dilated_brainmask(seg_mask, brainmask, radius=2):
    """Obtain the brain mask dilated
    Parameters
    ----------
    atlaslabels: ndarray
        A 3D binary array, resampled into ``img`` space.
    brainmask: ndarray
        A 3D binary array, resampled into ``img`` space.
    radius: int, optional
        The radius of the ball-shaped footprint for dilation of the mask.
    """
    from scipy import ndimage as ndi
    from skimage.morphology import ball

    # Union of functionally and anatomically extracted masks
    func_seg_mask = seg_mask | brainmask

    if func_seg_mask.ndim != 3:
        raise Exception("The brain mask should be a 3D array")

    dilated_brainmask = ndi.binary_dilation(func_seg_mask, ball(radius))

    return dilated_brainmask, func_seg_mask
