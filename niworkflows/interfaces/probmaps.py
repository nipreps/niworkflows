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
"""Utilities."""
import numpy as np
import nibabel as nb

from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits,
    isdefined,
    File,
    InputMultiPath,
    TraitedSpec,
    BaseInterfaceInputSpec,
    SimpleInterface,
)


LOG = logging.getLogger("nipype.interface")


class _TPM2ROIInputSpec(BaseInterfaceInputSpec):
    in_tpm = File(
        exists=True, mandatory=True, desc="Tissue probability map file in T1 space"
    )
    in_mask = File(
        exists=True, mandatory=True, desc="Binary mask of skull-stripped T1w image"
    )
    mask_erode_mm = traits.Float(
        xor=["mask_erode_prop"], desc="erode input mask (kernel width in mm)"
    )
    erode_mm = traits.Float(
        xor=["erode_prop"], desc="erode output mask (kernel width in mm)"
    )
    mask_erode_prop = traits.Float(
        xor=["mask_erode_mm"], desc="erode input mask (target volume ratio)"
    )
    erode_prop = traits.Float(
        xor=["erode_mm"], desc="erode output mask (target volume ratio)"
    )
    prob_thresh = traits.Float(
        0.95, usedefault=True, desc="threshold for the tissue probability maps"
    )


class _TPM2ROIOutputSpec(TraitedSpec):
    roi_file = File(exists=True, desc="output ROI file")
    eroded_mask = File(exists=True, desc="resulting eroded mask")


class TPM2ROI(SimpleInterface):
    """
    Convert tissue probability maps (TPMs) into ROIs.

    This interface follows the following logic:

    #. Erode ``in_mask`` by ``mask_erode_mm`` and apply to ``in_tpm``
    #. Threshold masked TPM at ``prob_thresh``
    #. Erode resulting mask by ``erode_mm``

    """

    input_spec = _TPM2ROIInputSpec
    output_spec = _TPM2ROIOutputSpec

    def _run_interface(self, runtime):
        mask_erode_mm = self.inputs.mask_erode_mm
        if not isdefined(mask_erode_mm):
            mask_erode_mm = None
        erode_mm = self.inputs.erode_mm
        if not isdefined(erode_mm):
            erode_mm = None
        mask_erode_prop = self.inputs.mask_erode_prop
        if not isdefined(mask_erode_prop):
            mask_erode_prop = None
        erode_prop = self.inputs.erode_prop
        if not isdefined(erode_prop):
            erode_prop = None
        roi_file, eroded_mask = _tpm2roi(
            self.inputs.in_tpm,
            self.inputs.in_mask,
            mask_erode_mm,
            erode_mm,
            mask_erode_prop,
            erode_prop,
            self.inputs.prob_thresh,
            newpath=runtime.cwd,
        )
        self._results["roi_file"] = roi_file
        self._results["eroded_mask"] = eroded_mask
        return runtime


class _AddTPMsInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(
        File(exists=True), mandatory=True, desc="input list of ROIs"
    )
    indices = traits.List(traits.Int, desc="select specific maps")


class _AddTPMsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="union of binarized input files")


class AddTPMs(SimpleInterface):
    """Calculate the union of several :abbr:`TPMs (tissue-probability maps)`."""

    input_spec = _AddTPMsInputSpec
    output_spec = _AddTPMsOutputSpec

    def _run_interface(self, runtime):
        in_files = self.inputs.in_files

        indices = list(range(len(in_files)))
        if isdefined(self.inputs.indices):
            indices = self.inputs.indices

        if len(self.inputs.in_files) < 2:
            self._results["out_file"] = in_files[0]
            return runtime

        first_fname = in_files[indices[0]]
        if len(indices) == 1:
            self._results["out_file"] = first_fname
            return runtime

        im = nb.concat_images([in_files[i] for i in indices])
        data = im.get_fdata().sum(axis=3)
        data = np.clip(data, a_min=0.0, a_max=1.0)

        out_file = fname_presuffix(first_fname, suffix="_tpmsum", newpath=runtime.cwd)
        newnii = im.__class__(data, im.affine, im.header)
        newnii.set_data_dtype(np.float32)

        # Set visualization thresholds
        newnii.header["cal_max"] = 1.0
        newnii.header["cal_min"] = 0.0
        newnii.to_filename(out_file)
        self._results["out_file"] = out_file

        return runtime


def _tpm2roi(
    in_tpm,
    in_mask,
    mask_erosion_mm=None,
    erosion_mm=None,
    mask_erosion_prop=None,
    erosion_prop=None,
    pthres=0.95,
    newpath=None,
):
    """
    Generate a mask from a tissue probability map
    """
    import scipy.ndimage as nd

    tpm_img = nb.load(in_tpm)
    roi_mask = (tpm_img.get_fdata() >= pthres).astype(np.uint8)

    eroded_mask_file = None
    erode_in = (mask_erosion_mm is not None and mask_erosion_mm > 0) or (
        mask_erosion_prop is not None and mask_erosion_prop < 1
    )
    if erode_in:
        eroded_mask_file = fname_presuffix(in_mask, suffix="_eroded", newpath=newpath)
        mask_img = nb.load(in_mask)
        mask_data = np.asanyarray(mask_img.dataobj).astype(np.uint8)
        if mask_erosion_mm:
            iter_n = max(int(mask_erosion_mm / max(mask_img.header.get_zooms())), 1)
            mask_data = nd.binary_erosion(mask_data, iterations=iter_n)
        else:
            orig_vol = np.sum(mask_data > 0)
            while np.sum(mask_data > 0) / orig_vol > mask_erosion_prop:
                mask_data = nd.binary_erosion(mask_data, iterations=1)

        # Store mask
        eroded = nb.Nifti1Image(mask_data, mask_img.affine, mask_img.header)
        eroded.set_data_dtype(np.uint8)
        eroded.to_filename(eroded_mask_file)

        # Mask TPM data (no effect if not eroded)
        roi_mask[~mask_data] = 0

    # shrinking
    erode_out = (erosion_mm is not None and erosion_mm > 0) or (
        erosion_prop is not None and erosion_prop < 1
    )
    if erode_out:
        if erosion_mm:
            iter_n = max(int(erosion_mm / max(tpm_img.header.get_zooms())), 1)
            iter_n = int(erosion_mm / max(tpm_img.header.get_zooms()))
            roi_mask = nd.binary_erosion(roi_mask, iterations=iter_n)
        else:
            orig_vol = np.sum(roi_mask > 0)
            while np.sum(roi_mask > 0) / orig_vol > erosion_prop:
                roi_mask = nd.binary_erosion(roi_mask, iterations=1)

    # Create image to resample
    roi_fname = fname_presuffix(in_tpm, suffix="_roi", newpath=newpath)
    roi_img = nb.Nifti1Image(roi_mask, tpm_img.affine, tpm_img.header)
    roi_img.set_data_dtype(np.uint8)
    roi_img.to_filename(roi_fname)
    return roi_fname, eroded_mask_file or in_mask
