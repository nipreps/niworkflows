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


class _BinaryDilationInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="binary file to dilate")
    radius = traits.Float(3, usedefault=True, desc="structure element (ball) radius")
    iterations = traits.Range(low=0, value=1, usedefault=True, desc="repeat dilation")


class _BinaryDilationOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="the input file, after binary dilation")


class BinaryDilation(SimpleInterface):
    """Morphological binary dilation using Scipy."""

    input_spec = _BinaryDilationInputSpec
    output_spec = _BinaryDilationOutputSpec

    def _run_interface(self, runtime):
        self._results["out_file"] = _dilate(
            self.inputs.in_file,
            radius=self.inputs.radius,
            iterations=self.inputs.iterations,
            newpath=runtime.cwd,
        )
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


class _MergeROIsInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiObject(File(exists=True), desc="ROI files to be merged")


class _MergeROIsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="NIfTI containing all ROIs")


class MergeROIs(SimpleInterface):
    """Combine multiple region of interest files (3D or 4D) into a single file"""

    input_spec = _MergeROIsInputSpec
    output_spec = _MergeROIsOutputSpec

    def _run_interface(self, runtime):
        self._results["out_file"] = _merge_rois(self.inputs.in_files, newpath=runtime.cwd)
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


class _GenerateSamplingReferenceInputSpec(BaseInterfaceInputSpec):
    fixed_image = File(
        exists=True, mandatory=True, desc="the reference file, defines the FoV"
    )
    moving_image = File(exists=True, mandatory=True, desc="the pixel size reference")
    xform_code = traits.Enum(None, 2, 4, usedefault=True, desc="force xform code")
    fov_mask = traits.Either(
        None,
        File(exists=True),
        usedefault=True,
        desc="mask to clip field of view (in fixed_image space)",
    )
    keep_native = traits.Bool(
        True,
        usedefault=True,
        desc="calculate a grid with native resolution covering "
        "the volume extent given by fixed_image, fast forward "
        "fixed_image otherwise.",
    )


class _GenerateSamplingReferenceOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="one file with all inputs flattened")


class GenerateSamplingReference(SimpleInterface):
    """
    Generates a reference grid for resampling one image keeping original resolution,
    but moving data to a different space (e.g. MNI).

    If the `fov_mask` optional input is provided, then the abbr:`FoV (field-of-view)`
    is cropped to a bounding box containing the brain mask plus an offest of two
    voxels along all dimensions. The `fov_mask` should be to the brain mask calculated
    from the T1w, and should not contain the brain stem. The mask is resampled into
    target space, and then the bounding box is calculated. Finally, the FoV is adjusted
    to that bounding box.


    """

    input_spec = _GenerateSamplingReferenceInputSpec
    output_spec = _GenerateSamplingReferenceOutputSpec

    def _run_interface(self, runtime):
        if not self.inputs.keep_native:
            self._results["out_file"] = self.inputs.fixed_image
            return runtime

        from .. import __version__

        self._results["out_file"] = _gen_reference(
            self.inputs.fixed_image,
            self.inputs.moving_image,
            fov_mask=self.inputs.fov_mask,
            force_xform_code=self.inputs.xform_code,
            message="%s (niworkflows v%s)" % (self.__class__.__name__, __version__),
            newpath=runtime.cwd,
        )
        return runtime


class _IntensityClipInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True, mandatory=True, desc="3D file which intensity will be clipped"
    )
    p_min = traits.Float(35.0, usedefault=True, desc="percentile for the lower bound")
    p_max = traits.Float(99.98, usedefault=True, desc="percentile for the upper bound")
    nonnegative = traits.Bool(
        True, usedefault=True, desc="whether input intensities must be positive"
    )
    dtype = traits.Enum(
        "int16", "float32", "uint8", usedefault=True, desc="output datatype"
    )
    invert = traits.Bool(False, usedefault=True, desc="finalize by inverting contrast")


class _IntensityClipOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="file after clipping")


class IntensityClip(SimpleInterface):
    """Clip the intensity range as prescribed by the percentiles."""

    input_spec = _IntensityClipInputSpec
    output_spec = _IntensityClipOutputSpec

    def _run_interface(self, runtime):
        self._results["out_file"] = _advanced_clip(
            self.inputs.in_file,
            p_min=self.inputs.p_min,
            p_max=self.inputs.p_max,
            nonnegative=self.inputs.nonnegative,
            dtype=self.inputs.dtype,
            invert=self.inputs.invert,
            newpath=runtime.cwd,
        )
        return runtime


class _MapLabelsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc="Segmented NIfTI")
    mappings = traits.Dict(
        xor=["mappings_file"],
        desc="Dictionary of label / replacement label pairs",
    )
    mappings_file = File(
        exists=True, xor=["mappings"], help="JSON composed of label / replacement label pairs."
    )


class _MapLabelsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Labeled file")


class MapLabels(SimpleInterface):
    """Remap discrete labels"""

    input_spec = _MapLabelsInputSpec
    output_spec = _MapLabelsOutputSpec

    def _run_interface(self, runtime):
        mapping = self.inputs.mappings or _load_int_json(self.inputs.mappings_file)
        self._results["out_file"] = _remap_labels(
            self.inputs.in_file,
            mapping,
            newpath=runtime.cwd,
        )
        return runtime


def _gen_reference(
    fixed_image,
    moving_image,
    fov_mask=None,
    out_file=None,
    message=None,
    force_xform_code=None,
    newpath=None,
):
    """Generate a sampling reference, and makes sure xform matrices/codes are correct."""
    import nilearn.image as nli

    if out_file is None:
        out_file = fname_presuffix(
            fixed_image, suffix="_reference", newpath=newpath
        )

    # Moving images may not be RAS/LPS (more generally, transverse-longitudinal-axial)
    reoriented_moving_img = nb.as_closest_canonical(nb.load(moving_image))
    new_zooms = reoriented_moving_img.header.get_zooms()[:3]

    # Avoid small differences in reported resolution to cause changes to
    # FOV. See https://github.com/nipreps/fmriprep/issues/512
    # A positive diagonal affine is RAS, hence the need to reorient above.
    new_affine = np.diag(np.round(new_zooms, 3))

    resampled = nli.resample_img(
        fixed_image, target_affine=new_affine, interpolation="nearest"
    )

    if fov_mask is not None:
        # If we have a mask, resample again dropping (empty) samples
        # out of the FoV.
        fixednii = nb.load(fixed_image)
        masknii = nb.load(fov_mask)

        if np.all(masknii.shape[:3] != fixednii.shape[:3]):
            raise RuntimeError("Fixed image and mask do not have the same dimensions.")

        if not np.allclose(masknii.affine, fixednii.affine, atol=1e-5):
            raise RuntimeError("Fixed image and mask have different affines")

        # Get mask into reference space
        masknii = nli.resample_img(
            masknii, target_affine=new_affine, interpolation="nearest"
        )
        res_shape = np.array(masknii.shape[:3])

        # Calculate a bounding box for the input mask
        # with an offset of 2 voxels per face
        bbox = np.argwhere(np.asanyarray(masknii.dataobj) > 0)
        new_origin = np.clip(bbox.min(0) - 2, a_min=0, a_max=None)
        new_end = np.clip(bbox.max(0) + 2, a_min=0, a_max=res_shape - 1)

        # Find new origin, and set into new affine
        new_affine_4 = resampled.affine.copy()
        new_affine_4[:3, 3] = new_affine_4[:3, :3].dot(new_origin) + new_affine_4[:3, 3]

        # Calculate new shapes
        new_shape = new_end - new_origin + 1
        resampled = nli.resample_img(
            fixed_image,
            target_affine=new_affine_4,
            target_shape=new_shape.tolist(),
            interpolation="nearest",
        )

    xform = resampled.affine  # nibabel will pick the best affine
    _, qform_code = resampled.header.get_qform(coded=True)
    _, sform_code = resampled.header.get_sform(coded=True)

    xform_code = sform_code if sform_code > 0 else qform_code
    if xform_code == 1:
        xform_code = 2

    if force_xform_code is not None:
        xform_code = force_xform_code

    # Keep 0, 2, 3, 4 unchanged
    resampled.header.set_qform(xform, int(xform_code))
    resampled.header.set_sform(xform, int(xform_code))
    resampled.header["descrip"] = "reference image generated by %s." % (
        message or "(unknown software)"
    )
    resampled.to_filename(out_file)
    return out_file


def _advanced_clip(
    in_file, p_min=35, p_max=99.98, nonnegative=True, dtype="int16", invert=False, newpath=None,
):
    """
    Remove outliers at both ends of the intensity distribution and fit into a given dtype.

    This interface tries to emulate ANTs workflows' massaging that truncate images into
    the 0-255 range, and applies percentiles for clipping images.
    For image registration, normalizing the intensity into a compact range (e.g., uint8)
    is generally advised.

    To more robustly determine the clipping thresholds, data are removed of spikes
    with a median filter.
    Once the thresholds are calculated, the denoised data are thrown away and the thresholds
    are applied on the original image.

    """
    from pathlib import Path
    import nibabel as nb
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import ball

    out_file = (Path(newpath or "") / "clipped.nii.gz").absolute()

    # Load data
    img = nb.squeeze_image(nb.load(in_file))
    if len(img.shape) != 3:
        raise RuntimeError(f"<{in_file}> is not a 3D file.")
    data = img.get_fdata(dtype="float32")

    # Calculate stats on denoised version, to preempt outliers from biasing
    denoised = ndimage.median_filter(data, footprint=ball(3))

    a_min = np.percentile(
        denoised[denoised > 0] if nonnegative else denoised,
        p_min
    )
    a_max = np.percentile(
        denoised[denoised > 0] if nonnegative else denoised,
        p_max
    )

    # Clip and cast
    data = np.clip(data, a_min=a_min, a_max=a_max)
    data -= data.min()
    data /= data.max()

    if invert:
        data = 1.0 - data

    if dtype in ("uint8", "int16"):
        data = np.round(255 * data).astype(dtype)

    hdr = img.header.copy()
    hdr.set_data_dtype(dtype)
    img.__class__(data, img.affine, hdr).to_filename(out_file)

    return str(out_file)


def _dilate(in_file, radius=3, iterations=1, newpath=None):
    """Dilate (binary) input mask."""
    from pathlib import Path
    import numpy as np
    import nibabel as nb
    from scipy import ndimage
    from skimage.morphology import ball
    from nipype.utils.filemanip import fname_presuffix

    mask = nb.load(in_file)
    newdata = ndimage.binary_dilation(
        np.asanyarray(mask.dataobj) > 0,
        structure=ball(radius),
        iterations=iterations,
    )

    hdr = mask.header.copy()
    hdr.set_data_dtype("uint8")
    out_file = fname_presuffix(in_file, suffix="_dil", newpath=newpath or Path.cwd())
    mask.__class__(newdata.astype("uint8"), mask.affine, hdr).to_filename(out_file)
    return out_file


def _merge_rois(in_files, newpath=None):
    """
    Aggregate individual 4D ROI files together into a single subcortical NIfTI.
    All ROI images are sanity checked with regards to:
    1) Shape
    2) Affine
    3) Overlap

    If any of these checks fail, an ``AssertionError`` will be raised.
    """
    from pathlib import Path
    import nibabel as nb
    import numpy as np

    img = nb.load(in_files[0])
    data = np.array(img.dataobj)
    affine = img.affine
    header = img.header

    nonzero = np.any(data, axis=3)
    for roi in in_files[1:]:
        img = nb.load(roi)
        assert img.shape == data.shape, "Mismatch in image shape"
        assert np.allclose(img.affine, affine), "Mismatch in affine"
        roi_data = np.asanyarray(img.dataobj)
        roi_nonzero = np.any(roi_data, axis=3)
        assert not np.any(roi_nonzero & nonzero), "Overlapping ROIs"
        nonzero |= roi_nonzero
        data += roi_data
        del roi_data

    if newpath is None:
        newpath = Path()
    out_file = str((Path(newpath) / "combined.nii.gz").absolute())
    img.__class__(data, affine, header).to_filename(out_file)
    return out_file


def _remap_labels(in_file, mapping, newpath=None):
    from pathlib import Path
    import nibabel as nb
    import numpy as np

    dtype = np.int16
    img = nb.load(in_file)
    data = np.asarray(img.dataobj, dtype=dtype)
    vec = data.ravel()

    def _relabel(label):
        return mapping.get(label, label)

    labels = np.unique(vec)  # include all labels present
    # copy values and substitute mappings
    subs = np.asarray(list(map(_relabel, labels)), dtype=dtype)
    subbed = np.zeros(labels.max() + 1, dtype=dtype)
    subbed[labels] = subs
    out = subbed[vec].reshape(data.shape)

    if newpath is None:
        newpath = Path()
    out_file = str((Path(newpath) / "relabeled.nii.gz").absolute())
    img.__class__(out, img.affine, header=img.header).to_filename(out_file)
    return out_file


def _load_int_json(json_file):
    import json

    def _keys_as_ints(d):
        return {int(k): v for k, v in d.items()}

    with open(json_file) as fp:
        data = json.load(fp, object_hook=_keys_as_ints)
    return data
