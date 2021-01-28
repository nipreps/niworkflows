# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Nibabel-based interfaces."""
import numpy as np
import nibabel as nb
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

IFLOGGER = logging.getLogger("nipype.interface")


class _ApplyMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="an image")
    in_mask = File(exists=True, mandatory=True, desc="a mask")
    threshold = traits.Float(
        0.5, usedefault=True, desc="a threshold to the mask, if it is nonbinary"
    )


class _ApplyMaskOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="masked file")


class ApplyMask(SimpleInterface):
    """Mask the input given a mask."""

    input_spec = _ApplyMaskInputSpec
    output_spec = _ApplyMaskOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        msknii = nb.load(self.inputs.in_mask)
        msk = msknii.get_fdata() > self.inputs.threshold

        self._results["out_file"] = fname_presuffix(
            self.inputs.in_file, suffix="_masked", newpath=runtime.cwd
        )

        if img.dataobj.shape[:3] != msk.shape:
            raise ValueError("Image and mask sizes do not match.")

        if not np.allclose(img.affine, msknii.affine):
            raise ValueError("Image and mask affines are not similar enough.")

        if img.dataobj.ndim == msk.ndim + 1:
            msk = msk[..., np.newaxis]

        masked = img.__class__(img.dataobj * msk, None, img.header)
        masked.to_filename(self._results["out_file"])
        return runtime


class _BinarizeInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="input image")
    thresh_low = traits.Float(mandatory=True, desc="non-inclusive lower threshold")


class _BinarizeOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="masked file")
    out_mask = File(exists=True, desc="output mask")


class Binarize(SimpleInterface):
    """Binarizes the input image applying the given thresholds."""

    input_spec = _BinarizeInputSpec
    output_spec = _BinarizeOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)

        self._results["out_file"] = fname_presuffix(
            self.inputs.in_file, suffix="_masked", newpath=runtime.cwd
        )
        self._results["out_mask"] = fname_presuffix(
            self.inputs.in_file, suffix="_mask", newpath=runtime.cwd
        )

        data = img.get_fdata()
        mask = data > self.inputs.thresh_low
        data[~mask] = 0.0
        masked = img.__class__(data, img.affine, img.header)
        masked.to_filename(self._results["out_file"])

        img.header.set_data_dtype("uint8")
        maskimg = img.__class__(mask.astype("uint8"), img.affine, img.header)
        maskimg.to_filename(self._results["out_mask"])

        return runtime


class _SplitSeriesInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="input 4d image")


class _SplitSeriesOutputSpec(TraitedSpec):
    out_files = OutputMultiObject(File(exists=True), desc="output list of 3d images")


class SplitSeries(SimpleInterface):
    """Split a 4D dataset along the last dimension into a series of 3D volumes."""

    input_spec = _SplitSeriesInputSpec
    output_spec = _SplitSeriesOutputSpec

    def _run_interface(self, runtime):
        in_file = self.inputs.in_file
        img = nb.load(in_file)
        extra_dims = tuple(dim for dim in img.shape[3:] if dim > 1) or (1,)
        if len(extra_dims) != 1:
            raise ValueError(f"Invalid shape {'x'.join(str(s) for s in img.shape)}")
        img = img.__class__(
            img.dataobj.reshape(img.shape[:3] + extra_dims), img.affine, img.header
        )

        self._results["out_files"] = []
        for i, img_3d in enumerate(nb.four_to_three(img)):
            out_file = fname_presuffix(
                in_file, suffix=f"_idx-{i:03}", newpath=runtime.cwd
            )
            img_3d.to_filename(out_file)
            self._results["out_files"].append(out_file)

        return runtime


class _MergeSeriesInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiObject(
        File(exists=True, mandatory=True, desc="input list of 3d images")
    )
    allow_4D = traits.Bool(
        True, usedefault=True, desc="whether 4D images are allowed to be concatenated"
    )


class _MergeSeriesOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output 4d image")


class MergeSeries(SimpleInterface):
    """Merge a series of 3D volumes along the last dimension into a single 4D image."""

    input_spec = _MergeSeriesInputSpec
    output_spec = _MergeSeriesOutputSpec

    def _run_interface(self, runtime):
        nii_list = []
        for f in self.inputs.in_files:
            filenii = nb.squeeze_image(nb.load(f))
            ndim = filenii.dataobj.ndim
            if ndim == 3:
                nii_list.append(filenii)
                continue
            elif self.inputs.allow_4D and ndim == 4:
                nii_list += nb.four_to_three(filenii)
                continue
            else:
                raise ValueError(
                    "Input image has an incorrect number of dimensions" f" ({ndim})."
                )

        img_4d = nb.concat_images(nii_list)
        out_file = fname_presuffix(
            self.inputs.in_files[0], suffix="_merged", newpath=runtime.cwd
        )
        img_4d.to_filename(out_file)

        self._results["out_file"] = out_file
        return runtime


class _RegridToZoomsInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True, mandatory=True, desc="a file whose resolution is to change"
    )
    zooms = traits.Tuple(
        traits.Float,
        traits.Float,
        traits.Float,
        mandatory=True,
        desc="the new resolution",
    )
    order = traits.Int(3, usedefault=True, desc="order of interpolator")
    clip = traits.Bool(
        True,
        usedefault=True,
        desc="clip the data array within the original image's range",
    )
    smooth = traits.Either(
        traits.Bool(),
        traits.Float(),
        default=False,
        usedefault=True,
        desc="apply gaussian smoothing before resampling",
    )


class _RegridToZoomsOutputSpec(TraitedSpec):
    out_file = File(exists=True, dec="the regridded file")


class RegridToZooms(SimpleInterface):
    """Change the resolution of an image (regrid)."""

    input_spec = _RegridToZoomsInputSpec
    output_spec = _RegridToZoomsOutputSpec

    def _run_interface(self, runtime):
        from ..utils.images import resample_by_spacing

        self._results["out_file"] = fname_presuffix(
            self.inputs.in_file, suffix="_regrid", newpath=runtime.cwd
        )
        resample_by_spacing(
            self.inputs.in_file,
            self.inputs.zooms,
            order=self.inputs.order,
            clip=self.inputs.clip,
            smooth=self.inputs.smooth,
        ).to_filename(self._results["out_file"])
        return runtime


class _DemeanImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="image to be demeaned")
    in_mask = File(
        exists=True, mandatory=True, desc="mask where median will be calculated"
    )
    only_mask = traits.Bool(False, usedefault=True, desc="demean only within mask")


class _DemeanImageOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="demeaned image")


class DemeanImage(SimpleInterface):
    input_spec = _DemeanImageInputSpec
    output_spec = _DemeanImageOutputSpec

    def _run_interface(self, runtime):
        from ..utils.images import demean

        self._results["out_file"] = demean(
            self.inputs.in_file,
            self.inputs.in_mask,
            only_mask=self.inputs.only_mask,
            newpath=runtime.cwd,
        )
        return runtime


class _FilledImageLikeInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="image to be demeaned")
    fill_value = traits.Float(1.0, usedefault=True, desc="value to fill")
    dtype = traits.Enum(
        "float32", "uint8", usedefault=True, desc="force output data type"
    )


class _FilledImageLikeOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="demeaned image")


class FilledImageLike(SimpleInterface):
    input_spec = _FilledImageLikeInputSpec
    output_spec = _FilledImageLikeOutputSpec

    def _run_interface(self, runtime):
        from ..utils.images import nii_ones_like

        self._results["out_file"] = nii_ones_like(
            self.inputs.in_file,
            self.inputs.fill_value,
            self.inputs.dtype,
            newpath=runtime.cwd,
        )
        return runtime
