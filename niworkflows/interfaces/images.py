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
"""Image tools interfaces."""
import os
from functools import partial
import numpy as np
import nibabel as nb

from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits,
    TraitedSpec,
    BaseInterfaceInputSpec,
    SimpleInterface,
    File,
    InputMultiObject,
    OutputMultiObject,
    isdefined,
)


LOGGER = logging.getLogger("nipype.interface")


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


class _IntraModalMergeInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiObject(File(exists=True), mandatory=True, desc="input files")
    in_mask = File(exists=True, desc="input mask for grand mean scaling")
    hmc = traits.Bool(True, usedefault=True)
    zero_based_avg = traits.Bool(True, usedefault=True)
    to_ras = traits.Bool(True, usedefault=True)
    grand_mean_scaling = traits.Bool(False, usedefault=True)


class _IntraModalMergeOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="merged image")
    out_avg = File(exists=True, desc="average image")
    out_mats = OutputMultiObject(File(exists=True), desc="output matrices")
    out_movpar = OutputMultiObject(File(exists=True), desc="output movement parameters")


class IntraModalMerge(SimpleInterface):
    """
    Calculate an average of the inputs.

    If the input is 3D, returns the original image.
    Otherwise, splits the images and merges them after
    head-motion correction with FSL ``mcflirt``.
    """

    input_spec = _IntraModalMergeInputSpec
    output_spec = _IntraModalMergeOutputSpec

    def _run_interface(self, runtime):
        in_files = self.inputs.in_files
        if not isinstance(in_files, list):
            in_files = [self.inputs.in_files]

        if self.inputs.to_ras:
            in_files = [reorient(inf, newpath=runtime.cwd) for inf in in_files]

        run_hmc = self.inputs.hmc and len(in_files) > 1

        nii_list = []
        # Remove one-sized extra dimensions
        for i, f in enumerate(in_files):
            filenii = nb.load(f)
            filenii = nb.squeeze_image(filenii)
            if len(filenii.shape) == 5:
                raise RuntimeError("Input image (%s) is 5D." % f)
            if filenii.dataobj.ndim == 4:
                nii_list += nb.four_to_three(filenii)
            else:
                nii_list.append(filenii)

        if len(nii_list) > 1:
            filenii = nb.concat_images(nii_list)
        else:
            filenii = nii_list[0]

        merged_fname = fname_presuffix(
            self.inputs.in_files[0], suffix="_merged", newpath=runtime.cwd
        )
        filenii.to_filename(merged_fname)
        self._results["out_file"] = merged_fname
        self._results["out_avg"] = merged_fname

        if filenii.dataobj.ndim < 4:
            # TODO: generate identity out_mats and zero-filled out_movpar
            return runtime

        if run_hmc:
            from nipype.interfaces.fsl import MCFLIRT

            mcflirt = MCFLIRT(
                cost="normcorr",
                save_mats=True,
                save_plots=True,
                ref_vol=0,
                in_file=merged_fname,
            )
            mcres = mcflirt.run()
            filenii = nb.load(mcres.outputs.out_file)
            self._results["out_file"] = mcres.outputs.out_file
            self._results["out_mats"] = mcres.outputs.mat_file
            self._results["out_movpar"] = mcres.outputs.par_file

        hmcdata = filenii.get_fdata(dtype="float32")
        if self.inputs.grand_mean_scaling:
            if not isdefined(self.inputs.in_mask):
                mean = np.median(hmcdata, axis=-1)
                thres = np.percentile(mean, 25)
                mask = mean > thres
            else:
                mask = nb.load(self.inputs.in_mask).get_fdata(dtype="float32") > 0.5

            nimgs = hmcdata.shape[-1]
            means = np.median(
                hmcdata[mask[..., np.newaxis]].reshape((-1, nimgs)).T, axis=-1
            )
            max_mean = means.max()
            for i in range(nimgs):
                hmcdata[..., i] *= max_mean / means[i]

        hmcdata = hmcdata.mean(axis=3)
        if self.inputs.zero_based_avg:
            hmcdata -= hmcdata.min()

        self._results["out_avg"] = fname_presuffix(
            self.inputs.in_files[0], suffix="_avg", newpath=runtime.cwd
        )
        nb.Nifti1Image(hmcdata, filenii.affine, filenii.header).to_filename(
            self._results["out_avg"]
        )

        return runtime


class _RobustAverageInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc="Either a 3D reference or 4D file to average through the last axis"
    )
    t_mask = traits.List(traits.Bool, desc="List of selected timepoints to be averaged")
    mc_method = traits.Enum(
        "AFNI",
        "FSL",
        None,
        usedefault=True,
        desc="Which software to use to perform motion correction",
    )
    nonnegative = traits.Bool(
        True, usedefault=True, desc="whether the output should be clipped below zero"
    )


class _RobustAverageOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="the averaged image")
    out_volumes = File(exists=True, desc="the volumes selected that have been averaged")
    out_drift = traits.List(
        traits.Float, desc="the ratio to the grand mean or global signal drift"
    )
    out_hmc = OutputMultiObject(File(exists=True), desc="head-motion correction matrices")


class RobustAverage(SimpleInterface):
    """Robustly estimate an average of the input."""

    input_spec = _RobustAverageInputSpec
    output_spec = _RobustAverageOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)

        # If reference is 3D, return it directly
        if img.dataobj.ndim == 3:
            self._results["out_file"] = self.inputs.in_file
            self._results["out_volumes"] = self.inputs.in_file
            self._results["out_drift"] = [1.0]
            return runtime

        fname = partial(fname_presuffix, self.inputs.in_file, newpath=runtime.cwd)

        # Slicing may induce inconsistencies with shape-dependent values in extensions.
        # For now, remove all. If this turns out to be a mistake, we can select extensions
        # that don't break pipeline stages.
        img.header.extensions.clear()
        img = nb.squeeze_image(img)

        # If reference was 4D, but single-volume - write out squeezed and return.
        if img.dataobj.ndim == 3:
            self._results["out_file"] = fname(suffix="_squeezed")
            img.to_filename(self._results["out_file"])
            self._results["out_volumes"] = self.inputs.in_file
            self._results["out_drift"] = [1.0]
            return runtime

        img_len = img.shape[3]
        t_mask = (
            self.inputs.t_mask if isdefined(self.inputs.t_mask) else [True] * img_len
        )

        if len(t_mask) != img_len:
            raise ValueError(
                f"Image length ({img_len} timepoints) unmatched by mask ({len(t_mask)})"
            )

        n_volumes = sum(t_mask)
        if n_volumes < 1:
            raise ValueError("At least one volume should be selected for slicing")

        self._results["out_file"] = fname(suffix="_average")
        self._results["out_volumes"] = fname(suffix="_sliced")

        sliced = nb.concat_images(
            i for i, t in zip(nb.four_to_three(img), t_mask) if t
        )

        data = sliced.get_fdata(dtype="float32")
        # Data can come with outliers showing very high numbers - preemptively prune
        data = np.clip(
            data,
            a_min=0.0 if self.inputs.nonnegative else np.percentile(data, 0.2),
            a_max=np.percentile(data, 99.8),
        )

        gs_drift = np.mean(data, axis=(0, 1, 2))
        gs_drift /= gs_drift.max()
        self._results["out_drift"] = [float(i) for i in gs_drift]

        data /= gs_drift
        data = np.clip(
            data,
            a_min=0.0 if self.inputs.nonnegative else data.min(),
            a_max=data.max(),
        )
        sliced.__class__(data, sliced.affine, sliced.header).to_filename(
            self._results["out_volumes"]
        )

        if n_volumes == 1:
            nb.squeeze_image(sliced).to_filename(self._results["out_file"])
            self._results["out_drift"] = [1.0]
            return runtime

        if self.inputs.mc_method == "AFNI":
            from nipype.interfaces.afni import Volreg

            res = Volreg(
                in_file=self._results["out_volumes"],
                args="-Fourier -twopass",
                zpad=4,
                outputtype="NIFTI_GZ",
            ).run()
            # self._results["out_hmc"] = res.outputs.oned_matrix_save

        elif self.inputs.mc_method == "FSL":
            from nipype.interfaces.fsl import MCFLIRT

            res = MCFLIRT(
                in_file=self._results["out_volumes"],
                ref_vol=0,
                interpolation="sinc",
            ).run()
            self._results["out_hmc"] = res.outputs.mat_file

        if self.inputs.mc_method:
            data = nb.load(res.outputs.out_file).get_fdata(dtype="float32")

        data = np.clip(
            data,
            a_min=0.0 if self.inputs.nonnegative else data.min(),
            a_max=data.max(),
        )

        sliced.__class__(
            np.median(data, axis=3), sliced.affine, sliced.header
        ).to_filename(self._results["out_file"])
        return runtime


CONFORMATION_TEMPLATE = """\t\t<h3 class="elem-title">Anatomical Conformation</h3>
\t\t<ul class="elem-desc">
\t\t\t<li>Input T1w images: {n_t1w}</li>
\t\t\t<li>Output orientation: RAS</li>
\t\t\t<li>Output dimensions: {dims}</li>
\t\t\t<li>Output voxel size: {zooms}</li>
\t\t\t<li>Discarded images: {n_discards}</li>
{discard_list}
\t\t</ul>
"""

DISCARD_TEMPLATE = """\t\t\t\t<li><abbr title="{path}">{basename}</abbr></li>"""


class _TemplateDimensionsInputSpec(BaseInterfaceInputSpec):
    t1w_list = InputMultiObject(
        File(exists=True), mandatory=True, desc="input T1w images"
    )
    max_scale = traits.Float(
        3.0, usedefault=True, desc="Maximum scaling factor in images to accept"
    )


class _TemplateDimensionsOutputSpec(TraitedSpec):
    t1w_valid_list = OutputMultiObject(exists=True, desc="valid T1w images")
    target_zooms = traits.Tuple(
        traits.Float, traits.Float, traits.Float, desc="Target zoom information"
    )
    target_shape = traits.Tuple(
        traits.Int, traits.Int, traits.Int, desc="Target shape information"
    )
    out_report = File(exists=True, desc="conformation report")


class TemplateDimensions(SimpleInterface):
    """
    Finds template target dimensions for a series of T1w images, filtering low-resolution images,
    if necessary.

    Along each axis, the minimum voxel size (zoom) and the maximum number of voxels (shape) are
    found across images.

    The ``max_scale`` parameter sets a bound on the degree of up-sampling performed.
    By default, an image with a voxel size greater than 3x the smallest voxel size
    (calculated separately for each dimension) will be discarded.

    To select images that require no scaling (i.e. all have smallest voxel sizes),
    set ``max_scale=1``.
    """

    input_spec = _TemplateDimensionsInputSpec
    output_spec = _TemplateDimensionsOutputSpec

    def _generate_segment(self, discards, dims, zooms):
        items = [
            DISCARD_TEMPLATE.format(path=path, basename=os.path.basename(path))
            for path in discards
        ]
        discard_list = (
            "\n".join(["\t\t\t<ul>"] + items + ["\t\t\t</ul>"]) if items else ""
        )
        zoom_fmt = "{:.02g}mm x {:.02g}mm x {:.02g}mm".format(*zooms)
        return CONFORMATION_TEMPLATE.format(
            n_t1w=len(self.inputs.t1w_list),
            dims="x".join(map(str, dims)),
            zooms=zoom_fmt,
            n_discards=len(discards),
            discard_list=discard_list,
        )

    def _run_interface(self, runtime):
        # Load images, orient as RAS, collect shape and zoom data
        in_names = np.array(self.inputs.t1w_list)
        orig_imgs = np.vectorize(nb.load)(in_names)
        reoriented = np.vectorize(nb.as_closest_canonical)(orig_imgs)
        all_zooms = np.array([img.header.get_zooms()[:3] for img in reoriented])
        all_shapes = np.array([img.shape[:3] for img in reoriented])

        # Identify images that would require excessive up-sampling
        valid = np.ones(all_zooms.shape[0], dtype=bool)
        while valid.any():
            target_zooms = all_zooms[valid].min(axis=0)
            scales = all_zooms[valid] / target_zooms
            if np.all(scales < self.inputs.max_scale):
                break
            valid[valid] ^= np.any(scales == scales.max(), axis=1)

        # Ignore dropped images
        valid_fnames = np.atleast_1d(in_names[valid]).tolist()
        self._results["t1w_valid_list"] = valid_fnames

        # Set target shape information
        target_zooms = all_zooms[valid].min(axis=0)
        target_shape = all_shapes[valid].max(axis=0)

        self._results["target_zooms"] = tuple(target_zooms.tolist())
        self._results["target_shape"] = tuple(target_shape.tolist())

        # Create report
        dropped_images = in_names[~valid]
        segment = self._generate_segment(dropped_images, target_shape, target_zooms)
        out_report = os.path.join(runtime.cwd, "report.html")
        with open(out_report, "w") as fobj:
            fobj.write(segment)

        self._results["out_report"] = out_report

        return runtime


class _ConformInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="Input image")
    target_zooms = traits.Tuple(
        traits.Float, traits.Float, traits.Float, desc="Target zoom information"
    )
    target_shape = traits.Tuple(
        traits.Int, traits.Int, traits.Int, desc="Target shape information"
    )


class _ConformOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Conformed image")
    transform = File(exists=True, desc="Conformation transform (voxel-to-voxel)")


class Conform(SimpleInterface):
    """
    Conform a series of T1w images to enable merging.

    Performs two basic functions:

    #. Orient to RAS (left-right, posterior-anterior, inferior-superior)
    #. Resample to target zooms (voxel sizes) and shape (number of voxels)

    Note that the output transforms are voxel-to-voxel; the RAS-to-RAS
    transform is the identity transform.

    """

    input_spec = _ConformInputSpec
    output_spec = _ConformOutputSpec

    def _run_interface(self, runtime):
        # Load image, orient as RAS
        fname = self.inputs.in_file
        orig_img = nb.load(fname)
        reoriented = nb.as_closest_canonical(orig_img)

        # Set target shape information
        target_zooms = np.array(self.inputs.target_zooms)
        target_shape = np.array(self.inputs.target_shape)
        target_span = target_shape * target_zooms

        zooms = np.array(reoriented.header.get_zooms()[:3])
        shape = np.array(reoriented.shape[:3])

        # Reconstruct transform from orig to reoriented image
        ornt_xfm = nb.orientations.inv_ornt_aff(
            nb.io_orientation(orig_img.affine), orig_img.shape
        )
        # Identity unless proven otherwise
        target_affine = reoriented.affine.copy()
        conform_xfm = np.eye(4)

        xyz_unit = reoriented.header.get_xyzt_units()[0]
        if xyz_unit == "unknown":
            # Common assumption; if we're wrong, unlikely to be the only thing that breaks
            xyz_unit = "mm"

        # Set a 0.05mm threshold to performing rescaling
        atol_gross = {"meter": 5e-5, "mm": 0.05, "micron": 50}[xyz_unit]
        # if 0.01 > difference > 0.001mm, freesurfer won't be able to merge the images
        atol_fine = {"meter": 1e-6, "mm": 0.001, "micron": 1}[xyz_unit]

        # Update zooms => Modify affine
        # Rescale => Resample to resized voxels
        # Resize => Resample to new image dimensions
        update_zooms = not np.allclose(zooms, target_zooms, atol=atol_fine, rtol=0)
        rescale = not np.allclose(zooms, target_zooms, atol=atol_gross, rtol=0)
        resize = not np.all(shape == target_shape)
        resample = rescale or resize
        if resample or update_zooms:
            # Use an affine with the corrected zooms, whether or not we resample
            if update_zooms:
                scale_factor = target_zooms / zooms
                target_affine[:3, :3] = reoriented.affine[:3, :3] @ np.diag(
                    scale_factor
                )

            if resize:
                # The shift is applied after scaling.
                # Use a proportional shift to maintain relative position in dataset
                size_factor = target_span / (zooms * shape)
                # Use integer shifts to avoid unnecessary interpolation
                offset = (
                    reoriented.affine[:3, 3] * size_factor - reoriented.affine[:3, 3]
                )
                target_affine[:3, 3] = reoriented.affine[:3, 3] + offset.astype(int)

            conform_xfm = np.linalg.inv(reoriented.affine) @ target_affine

            # Create new image
            data = reoriented.dataobj
            if resample:
                import nilearn.image as nli

                data = nli.resample_img(reoriented, target_affine, target_shape).dataobj
            reoriented = reoriented.__class__(data, target_affine, reoriented.header)

        # Image may be reoriented, rescaled, and/or resized
        if reoriented is not orig_img:
            out_name = fname_presuffix(fname, suffix="_ras", newpath=runtime.cwd)
            reoriented.to_filename(out_name)
        else:
            out_name = fname

        transform = ornt_xfm.dot(conform_xfm)
        if not np.allclose(orig_img.affine.dot(transform), target_affine):
            raise ValueError("Original and target affines are not similar")

        mat_name = fname_presuffix(
            fname, suffix=".mat", newpath=runtime.cwd, use_ext=False
        )
        np.savetxt(mat_name, transform, fmt="%.08f")

        self._results["out_file"] = out_name
        self._results["transform"] = mat_name

        return runtime


def reorient(in_file, newpath=None):
    """Reorient Nifti files to RAS."""
    out_file = fname_presuffix(in_file, suffix="_ras", newpath=newpath)
    nb.as_closest_canonical(nb.load(in_file)).to_filename(out_file)
    return out_file


def normalize_xform(img):
    """
    Set identical, valid qform and sform matrices in an image.

    Selects the best available affine (sform > qform > shape-based), and
    coerces it to be qform-compatible (no shears).

    The resulting image represents this same affine as both qform and sform,
    and is marked as NIFTI_XFORM_ALIGNED_ANAT, indicating that it is valid,
    not aligned to template, and not necessarily preserving the original
    coordinates.

    If header would be unchanged, returns input image.
    """
    # Let nibabel convert from affine to quaternions, and recover xform
    tmp_header = img.header.copy()
    tmp_header.set_qform(img.affine)
    xform = tmp_header.get_qform()
    xform_code = 2

    # Check desired codes
    qform, qform_code = img.get_qform(coded=True)
    sform, sform_code = img.get_sform(coded=True)
    if all(
        (
            qform is not None and np.allclose(qform, xform),
            sform is not None and np.allclose(sform, xform),
            int(qform_code) == xform_code,
            int(sform_code) == xform_code,
        )
    ):
        return img

    new_img = img.__class__(img.dataobj, xform, img.header)
    # Unconditionally set sform/qform
    new_img.set_sform(xform, xform_code)
    new_img.set_qform(xform, xform_code)
    return new_img


class _SignalExtractionInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="4-D fMRI nii file")
    label_files = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc="a 3D label image, with 0 denoting "
        "background, or a list of 3D probability "
        "maps (one per label) or the equivalent 4D "
        "file.",
    )
    prob_thres = traits.Range(
        low=0.0,
        high=1.0,
        value=0.5,
        usedefault=True,
        desc="If label_files are probability masks, threshold "
        "at specified probability.",
    )
    class_labels = traits.List(
        mandatory=True,
        desc="Human-readable labels for each segment "
        "in the label file, in order. The length of "
        "class_labels must be equal to the number of "
        "segments (background excluded). This list "
        "corresponds to the class labels in label_file "
        "in ascending order",
    )
    out_file = File(
        "signals.tsv",
        usedefault=True,
        exists=False,
        desc="The name of the file to output to. " "signals.tsv by default",
    )


class _SignalExtractionOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc="tsv file containing the computed "
        "signals, with as many columns as there are labels and as "
        "many rows as there are timepoints in in_file, plus a "
        "header row with values from class_labels",
    )


class SignalExtraction(SimpleInterface):
    """
    Extract mean signals from a time series within a set of ROIs.

    This interface is intended to be a memory-efficient alternative to
    nipype.interfaces.nilearn.SignalExtraction.
    Not all features of nilearn.SignalExtraction are implemented at
    this time.

    """

    input_spec = _SignalExtractionInputSpec
    output_spec = _SignalExtractionOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        mask_imgs = [nb.load(fname) for fname in self.inputs.label_files]
        if len(mask_imgs) == 1 and len(mask_imgs[0].shape) == 4:
            mask_imgs = nb.four_to_three(mask_imgs[0])
        # This check assumes all input masks have same dimensions
        if img.shape[:3] != mask_imgs[0].shape[:3]:
            raise NotImplementedError(
                "Input image and mask should be of "
                "same dimensions before running SignalExtraction"
            )
        # Load the mask.
        # If mask is a list, each mask is treated as its own ROI/parcel
        # If mask is a 3D, each integer is treated as its own ROI/parcel
        if len(mask_imgs) > 1:
            masks = [
                np.asanyarray(mask_img.dataobj) >= self.inputs.prob_thres
                for mask_img in mask_imgs
            ]
        else:
            labelsmap = np.asanyarray(mask_imgs[0].dataobj)
            labels = np.unique(labelsmap)
            labels = labels[labels != 0]
            masks = [labelsmap == label for label in labels]

        if len(masks) != len(self.inputs.class_labels):
            raise ValueError("Number of masks must match number of labels")

        series = np.zeros((img.shape[3], len(masks)))

        data = img.get_fdata()
        for j, mask in enumerate(masks):
            series[:, j] = data[mask, :].mean(axis=0)

        output = np.vstack((self.inputs.class_labels, series.astype(str)))
        self._results["out_file"] = os.path.join(runtime.cwd, self.inputs.out_file)
        np.savetxt(self._results["out_file"], output, fmt=b"%s", delimiter="\t")

        return runtime
