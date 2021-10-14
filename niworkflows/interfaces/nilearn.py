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
"""Utilities based on nilearn."""
import os
import nibabel as nb
import numpy as np

from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits,
    isdefined,
    TraitedSpec,
    BaseInterfaceInputSpec,
    File,
    InputMultiPath,
    SimpleInterface,
)
from nipype.interfaces.mixins import reporting
from .reportlets import base as nrb

try:
    from nilearn import __version__ as NILEARN_VERSION
except ImportError:
    NILEARN_VERSION = "unknown"

LOGGER = logging.getLogger("nipype.interface")
__all__ = ["NILEARN_VERSION", "MaskEPI", "Merge", "ComputeEPIMask"]


class _MaskEPIInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(
        File(exists=True), mandatory=True, desc="input EPI or list of files"
    )
    lower_cutoff = traits.Float(0.2, usedefault=True)
    upper_cutoff = traits.Float(0.85, usedefault=True)
    connected = traits.Bool(True, usedefault=True)
    enhance_t2 = traits.Bool(
        False, usedefault=True, desc="enhance T2 contrast on image"
    )
    opening = traits.Int(2, usedefault=True)
    closing = traits.Bool(True, usedefault=True)
    fill_holes = traits.Bool(True, usedefault=True)
    exclude_zeros = traits.Bool(False, usedefault=True)
    ensure_finite = traits.Bool(True, usedefault=True)
    target_affine = traits.Either(
        None, traits.File(exists=True), default=None, usedefault=True
    )
    target_shape = traits.Either(
        None, traits.File(exists=True), default=None, usedefault=True
    )
    no_sanitize = traits.Bool(False, usedefault=True)


class _MaskEPIOutputSpec(TraitedSpec):
    out_mask = File(exists=True, desc="output mask")


class MaskEPI(SimpleInterface):
    """Run Nilearn's compute_epi_mask."""

    input_spec = _MaskEPIInputSpec
    output_spec = _MaskEPIOutputSpec

    def _run_interface(self, runtime):
        from skimage import morphology as sim
        from scipy.ndimage.morphology import binary_fill_holes
        from nilearn.masking import compute_epi_mask

        in_files = self.inputs.in_files

        if self.inputs.enhance_t2:
            in_files = [_enhance_t2_contrast(f, newpath=runtime.cwd) for f in in_files]

        masknii = compute_epi_mask(
            in_files,
            lower_cutoff=self.inputs.lower_cutoff,
            upper_cutoff=self.inputs.upper_cutoff,
            connected=self.inputs.connected,
            opening=self.inputs.opening,
            exclude_zeros=self.inputs.exclude_zeros,
            ensure_finite=self.inputs.ensure_finite,
            target_affine=self.inputs.target_affine,
            target_shape=self.inputs.target_shape,
        )

        if self.inputs.closing:
            closed = sim.binary_closing(
                np.asanyarray(masknii.dataobj).astype(np.uint8), sim.ball(1)
            ).astype(np.uint8)
            masknii = masknii.__class__(closed, masknii.affine, masknii.header)

        if self.inputs.fill_holes:
            filled = binary_fill_holes(
                np.asanyarray(masknii.dataobj).astype(np.uint8), sim.ball(6)
            ).astype(np.uint8)
            masknii = masknii.__class__(filled, masknii.affine, masknii.header)

        if self.inputs.no_sanitize:
            in_file = self.inputs.in_files
            if isinstance(in_file, list):
                in_file = in_file[0]
            nii = nb.load(in_file)
            qform, code = nii.get_qform(coded=True)
            masknii.set_qform(qform, int(code))
            sform, code = nii.get_sform(coded=True)
            masknii.set_sform(sform, int(code))

        self._results["out_mask"] = fname_presuffix(
            self.inputs.in_files[0], suffix="_mask", newpath=runtime.cwd
        )
        masknii.to_filename(self._results["out_mask"])
        return runtime


class _MergeInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(
        File(exists=True), mandatory=True, desc="input list of files to merge"
    )
    dtype = traits.Enum(
        "f4",
        "f8",
        "u1",
        "u2",
        "u4",
        "i2",
        "i4",
        usedefault=True,
        desc="numpy dtype of output image",
    )
    header_source = File(
        exists=True, desc="a Nifti file from which the header should be copied"
    )
    compress = traits.Bool(
        True, usedefault=True, desc="Use gzip compression on .nii output"
    )


class _MergeOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output merged file")


class Merge(SimpleInterface):
    """Run Nilearn's concat_imgs."""

    input_spec = _MergeInputSpec
    output_spec = _MergeOutputSpec

    def _run_interface(self, runtime):
        from nilearn.image import concat_imgs

        ext = ".nii.gz" if self.inputs.compress else ".nii"
        self._results["out_file"] = fname_presuffix(
            self.inputs.in_files[0],
            suffix="_merged" + ext,
            newpath=runtime.cwd,
            use_ext=False,
        )
        new_nii = concat_imgs(self.inputs.in_files, dtype=self.inputs.dtype)

        if isdefined(self.inputs.header_source):
            src_hdr = nb.load(self.inputs.header_source).header
            new_nii.header.set_xyzt_units(t=src_hdr.get_xyzt_units()[-1])
            new_nii.header.set_zooms(
                list(new_nii.header.get_zooms()[:3]) + [src_hdr.get_zooms()[3]]
            )

        new_nii.to_filename(self._results["out_file"])

        return runtime


class _ComputeEPIMaskInputSpec(nrb._SVGReportCapableInputSpec, BaseInterfaceInputSpec):
    in_file = File(exists=True, desc="3D or 4D EPI file")
    dilation = traits.Int(desc="binary dilation on the nilearn output")


class _ComputeEPIMaskOutputSpec(reporting.ReportCapableOutputSpec):
    mask_file = File(exists=True, desc="Binary brain mask")


class ComputeEPIMask(nrb.SegmentationRC):
    input_spec = _ComputeEPIMaskInputSpec
    output_spec = _ComputeEPIMaskOutputSpec

    def _run_interface(self, runtime):
        from scipy.ndimage.morphology import binary_dilation
        from nilearn.masking import compute_epi_mask

        orig_file_nii = nb.load(self.inputs.in_file)
        in_file_data = orig_file_nii.get_fdata()

        # pad the data to avoid the mask estimation running into edge effects
        in_file_data_padded = np.pad(
            in_file_data, (1, 1), "constant", constant_values=(0, 0)
        )

        padded_nii = nb.Nifti1Image(
            in_file_data_padded, orig_file_nii.affine, orig_file_nii.header
        )

        mask_nii = compute_epi_mask(padded_nii, exclude_zeros=True)

        mask_data = np.asanyarray(mask_nii.dataobj).astype(np.uint8)
        if isdefined(self.inputs.dilation):
            mask_data = binary_dilation(mask_data).astype(np.uint8)

        # reverse image padding
        mask_data = mask_data[1:-1, 1:-1, 1:-1]

        # exclude zero and NaN voxels
        mask_data[in_file_data == 0] = 0
        mask_data[np.isnan(in_file_data)] = 0

        better_mask = nb.Nifti1Image(
            mask_data, orig_file_nii.affine, orig_file_nii.header
        )
        better_mask.set_data_dtype(np.uint8)
        better_mask.to_filename("mask_file.nii.gz")

        self._mask_file = os.path.join(runtime.cwd, "mask_file.nii.gz")

        runtime.returncode = 0
        return super(ComputeEPIMask, self)._run_interface(runtime)

    def _list_outputs(self):
        outputs = super(ComputeEPIMask, self)._list_outputs()
        outputs["mask_file"] = self._mask_file
        return outputs

    def _post_run_hook(self, runtime):
        """Prepare report generation post-hook."""
        self._anat_file = self.inputs.in_file
        self._mask_file = self.aggregate_outputs(runtime=runtime).mask_file
        self._seg_files = [self._mask_file]
        self._masked = True

        LOGGER.info(
            'Generating report for nilearn.compute_epi_mask. file "%s", and mask file "%s"',
            self._anat_file,
            self._mask_file,
        )

        return super(ComputeEPIMask, self)._post_run_hook(runtime)


def _enhance_t2_contrast(in_file, newpath=None, offset=0.5):
    """
    Enhance the T2* contrast of an EPI dataset.

    Performs a logarithmic transformation of intensity that
    effectively splits brain and background and makes the
    overall distribution more Gaussian.
    """
    out_file = fname_presuffix(in_file, suffix="_t1enh", newpath=newpath)
    nii = nb.load(in_file)
    data = nii.get_fdata()
    maxd = data.max()
    newdata = np.log(offset + data / maxd)
    newdata -= newdata.min()
    newdata *= maxd / newdata.max()
    nii = nii.__class__(newdata, nii.affine, nii.header)
    nii.to_filename(out_file)
    return out_file
