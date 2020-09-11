# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Utilities."""
import os
import re
import json
import shutil
import numpy as np
import nibabel as nb
import nilearn.image as nli
from textwrap import indent
from collections import OrderedDict

import scipy.ndimage as nd
from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.utils.misc import normalize_mc_params
from nipype.interfaces.io import add_traits
from nipype.interfaces.base import (
    traits,
    isdefined,
    File,
    InputMultiPath,
    TraitedSpec,
    BaseInterfaceInputSpec,
    SimpleInterface,
    DynamicTraitedSpec,
)
from .. import __version__


LOG = logging.getLogger("nipype.interface")


class _CopyXFormInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    hdr_file = File(exists=True, mandatory=True, desc="the file we get the header from")


class CopyXForm(SimpleInterface):
    """
    Copy the x-form matrices from `hdr_file` to `out_file`.
    """

    input_spec = _CopyXFormInputSpec
    output_spec = DynamicTraitedSpec

    def __init__(self, fields=None, **inputs):
        self._fields = fields or ["in_file"]
        if isinstance(self._fields, str):
            self._fields = [self._fields]

        super(CopyXForm, self).__init__(**inputs)

        add_traits(self.inputs, self._fields)
        for f in set(self._fields).intersection(list(inputs.keys())):
            setattr(self.inputs, f, inputs[f])

    def _outputs(self):
        base = super(CopyXForm, self)._outputs()
        if self._fields:
            fields = self._fields.copy()
            if "in_file" in fields:
                idx = fields.index("in_file")
                fields.pop(idx)
                fields.insert(idx, "out_file")

            base = add_traits(base, fields)
        return base

    def _run_interface(self, runtime):
        for f in self._fields:
            in_files = getattr(self.inputs, f)
            self._results[f] = []
            if isinstance(in_files, str):
                in_files = [in_files]
            for in_file in in_files:
                out_name = fname_presuffix(
                    in_file, suffix="_xform", newpath=runtime.cwd
                )
                # Copy and replace header
                shutil.copy(in_file, out_name)
                _copyxform(
                    self.inputs.hdr_file,
                    out_name,
                    message="CopyXForm (niworkflows v%s)" % __version__,
                )
                self._results[f].append(out_name)

            # Flatten out one-element lists
            if len(self._results[f]) == 1:
                self._results[f] = self._results[f][0]

        default = self._results.pop("in_file", None)
        if default:
            self._results["out_file"] = default
        return runtime


class _CopyHeaderInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="the file we get the data from")
    hdr_file = File(exists=True, mandatory=True, desc="the file we get the header from")


class _CopyHeaderOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="written file path")


class CopyHeader(SimpleInterface):
    """
    Copy a header from the `hdr_file` to `out_file` with data drawn from
    `in_file`.
    """

    input_spec = _CopyHeaderInputSpec
    output_spec = _CopyHeaderOutputSpec

    def _run_interface(self, runtime):
        in_img = nb.load(self.inputs.hdr_file)
        out_img = nb.load(self.inputs.in_file)
        new_img = out_img.__class__(out_img.dataobj, in_img.affine, in_img.header)
        new_img.set_data_dtype(out_img.get_data_dtype())

        out_name = fname_presuffix(self.inputs.in_file, suffix="_fixhdr", newpath=".")
        new_img.to_filename(out_name)
        self._results["out_file"] = out_name
        return runtime


class _NormalizeMotionParamsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="the input parameters file")
    format = traits.Enum(
        "FSL", "AFNI", "FSFAST", "NIPY", usedefault=True, desc="output format"
    )


class _NormalizeMotionParamsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="written file path")


class NormalizeMotionParams(SimpleInterface):
    """
    Convert input motion parameters into the designated convention.

    """

    input_spec = _NormalizeMotionParamsInputSpec
    output_spec = _NormalizeMotionParamsOutputSpec

    def _run_interface(self, runtime):
        mpars = np.loadtxt(self.inputs.in_file)  # mpars is N_t x 6
        mpars = np.apply_along_axis(
            func1d=normalize_mc_params, axis=1, arr=mpars, source=self.inputs.format
        )
        self._results["out_file"] = os.path.join(runtime.cwd, "motion_params.txt")
        np.savetxt(self._results["out_file"], mpars)
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
        self._results["out_file"] = _gen_reference(
            self.inputs.fixed_image,
            self.inputs.moving_image,
            fov_mask=self.inputs.fov_mask,
            force_xform_code=self.inputs.xform_code,
            message="%s (niworkflows v%s)" % (self.__class__.__name__, __version__),
        )
        return runtime


def _copyxform(ref_image, out_image, message=None):
    # Read in reference and output
    # Use mmap=False because we will be overwriting the output image
    resampled = nb.load(out_image, mmap=False)
    orig = nb.load(ref_image)

    if not np.allclose(orig.affine, resampled.affine):
        LOG.debug(
            "Affines of input and reference images do not match, "
            "FMRIPREP will set the reference image headers. "
            "Please, check that the x-form matrices of the input dataset"
            "are correct and manually verify the alignment of results."
        )

    # Copy xform infos
    qform, qform_code = orig.header.get_qform(coded=True)
    sform, sform_code = orig.header.get_sform(coded=True)
    header = resampled.header.copy()
    header.set_qform(qform, int(qform_code))
    header.set_sform(sform, int(sform_code))
    header["descrip"] = "xform matrices modified by %s." % (message or "(unknown)")

    newimg = resampled.__class__(resampled.dataobj, orig.affine, header)
    newimg.to_filename(out_image)


def _gen_reference(
    fixed_image,
    moving_image,
    fov_mask=None,
    out_file=None,
    message=None,
    force_xform_code=None,
):
    """
    Generates a sampling reference, and makes sure xform matrices/codes are
    correct
    """

    if out_file is None:
        out_file = fname_presuffix(
            fixed_image, suffix="_reference", newpath=os.getcwd()
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
            fixed_image, target_affine=new_affine, interpolation="nearest"
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


class _SanitizeImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="input image")
    n_volumes_to_discard = traits.Int(
        0, usedefault=True, desc="discard n first volumes"
    )
    max_32bit = traits.Bool(
        False,
        usedefault=True,
        desc="cast data to float32 if higher " "precision is encountered",
    )


class _SanitizeImageOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="validated image")
    out_report = File(exists=True, desc="HTML segment containing warning")


class SanitizeImage(SimpleInterface):
    """
    Check the correctness of x-form headers (matrix and code) and fixes
    problematic combinations of values. Removes any extension form the header
    if present.
    This interface implements the `following logic
    <https://github.com/nipreps/fmriprep/issues/873#issuecomment-349394544>`_:
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    | valid quaternions | `qform_code > 0` | `sform_code > 0` | `qform == sform` \
| actions                                        |
    +===================+==================+==================+==================\
+================================================+
    | True              | True             | True             | True             \
| None                                           |
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    | True              | True             | False            | *                \
| sform, scode <- qform, qcode                   |
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    | *                 | True             | *                | False            \
| sform, scode <- qform, qcode                   |
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    | *                 | False            | True             | *                \
| qform, qcode <- sform, scode                   |
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    | *                 | False            | False            | *                \
| sform, qform <- best affine; scode, qcode <- 1 |
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    | False             | *                | False            | *                \
| sform, qform <- best affine; scode, qcode <- 1 |
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    """

    input_spec = _SanitizeImageInputSpec
    output_spec = _SanitizeImageOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        out_report = os.path.join(runtime.cwd, "report.html")

        # Retrieve xform codes
        sform_code = int(img.header._structarr["sform_code"])
        qform_code = int(img.header._structarr["qform_code"])

        # Check qform is valid
        valid_qform = False
        try:
            img.get_qform()
            valid_qform = True
        except ValueError:
            pass

        # Matching affines
        matching_affines = valid_qform and np.allclose(img.get_qform(), img.get_sform())

        save_file = False
        warning_txt = ""

        # Both match, qform valid (implicit with match), codes okay -> do nothing, empty report
        if matching_affines and qform_code > 0 and sform_code > 0:
            self._results["out_file"] = self.inputs.in_file
            open(out_report, "w").close()

        # Row 2:
        elif valid_qform and qform_code > 0:
            img.set_sform(img.get_qform(), qform_code)
            save_file = True
            warning_txt = "Note on orientation: sform matrix set"
            description = """\
<p class="elem-desc">The sform has been copied from qform.</p>
"""
        # Rows 3-4:
        # Note: if qform is not valid, matching_affines is False
        elif sform_code > 0 and (not matching_affines or qform_code == 0):
            img.set_qform(img.get_sform(), sform_code)
            save_file = True
            warning_txt = "Note on orientation: qform matrix overwritten"
            description = """\
<p class="elem-desc">The qform has been copied from sform.</p>
"""
            if not valid_qform and qform_code > 0:
                warning_txt = "WARNING - Invalid qform information"
                description = """\
<p class="elem-desc">
    The qform matrix found in the file header is invalid.
    The qform has been copied from sform.
    Checking the original qform information from the data produced
    by the scanner is advised.
</p>
"""
        # Rows 5-6:
        else:
            affine = img.affine
            img.set_sform(affine, nb.nifti1.xform_codes["scanner"])
            img.set_qform(affine, nb.nifti1.xform_codes["scanner"])
            save_file = True
            warning_txt = "WARNING - Missing orientation information"
            description = """\
<p class="elem-desc">
    Orientation information could not be retrieved from the image header.
    The qform and sform matrices have been set to a default, LAS-oriented affine.
    Analyses of this dataset MAY BE INVALID.
</p>
"""

        if (
            self.inputs.max_32bit and np.dtype(img.get_data_dtype()).itemsize > 4
        ) or self.inputs.n_volumes_to_discard:
            # force float32 only if 64 bit dtype is detected
            if self.inputs.max_32bit and np.dtype(img.get_data_dtype()).itemsize > 4:
                in_data = img.get_fdata(dtype=np.float32)
            else:
                in_data = img.dataobj

            img = nb.Nifti1Image(
                in_data[:, :, :, self.inputs.n_volumes_to_discard:],
                img.affine,
                img.header,
            )
            save_file = True

        if len(img.header.extensions) != 0:
            img.header.extensions.clear()
            save_file = True

        # Store new file
        if save_file:
            out_fname = fname_presuffix(
                self.inputs.in_file, suffix="_valid", newpath=runtime.cwd
            )
            self._results["out_file"] = out_fname
            img.to_filename(out_fname)

        if warning_txt:
            snippet = '<h3 class="elem-title">%s</h3>\n%s\n' % (
                warning_txt,
                description,
            )
            with open(out_report, "w") as fobj:
                fobj.write(indent(snippet, "\t" * 3))

        self._results["out_report"] = out_report
        return runtime


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
    """Convert tissue probability maps (TPMs) into ROIs

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
    """Calculate the union of several :abbr:`TPMs (tissue-probability map)`"""

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


class _AddTSVHeaderInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="input file")
    columns = traits.List(traits.Str, mandatory=True, desc="header for columns")


class _AddTSVHeaderOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output average file")


class AddTSVHeader(SimpleInterface):
    r"""Add a header row to a TSV file

    .. testsetup::

    >>> cwd = os.getcwd()
    >>> os.chdir(tmpdir)

    .. doctest::

    An example TSV:

    >>> np.savetxt('data.tsv', np.arange(30).reshape((6, 5)), delimiter='\t')

    Add headers:

    >>> addheader = AddTSVHeader()
    >>> addheader.inputs.in_file = 'data.tsv'
    >>> addheader.inputs.columns = ['a', 'b', 'c', 'd', 'e']
    >>> res = addheader.run()
    >>> df = pd.read_csv(res.outputs.out_file, delim_whitespace=True,
    ...                  index_col=None)
    >>> df.columns.ravel().tolist()
    ['a', 'b', 'c', 'd', 'e']

    >>> np.all(df.values == np.arange(30).reshape((6, 5)))
    True

    .. testcleanup::

    >>> os.chdir(cwd)

    """
    input_spec = _AddTSVHeaderInputSpec
    output_spec = _AddTSVHeaderOutputSpec

    def _run_interface(self, runtime):
        out_file = fname_presuffix(
            self.inputs.in_file,
            suffix="_motion.tsv",
            newpath=runtime.cwd,
            use_ext=False,
        )
        data = np.loadtxt(self.inputs.in_file)
        np.savetxt(
            out_file,
            data,
            delimiter="\t",
            header="\t".join(self.inputs.columns),
            comments="",
        )

        self._results["out_file"] = out_file
        return runtime


class _JoinTSVColumnsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="input file")
    join_file = File(exists=True, mandatory=True, desc="file to be adjoined")
    side = traits.Enum("right", "left", usedefault=True, desc="where to join")
    columns = traits.List(traits.Str, desc="header for columns")


class _JoinTSVColumnsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output TSV file")


class JoinTSVColumns(SimpleInterface):
    r"""Add a header row to a TSV file

    .. testsetup::

    >>> cwd = os.getcwd()
    >>> os.chdir(tmpdir)

    .. doctest::

    An example TSV:

    >>> data = np.arange(30).reshape((6, 5))
    >>> np.savetxt('data.tsv', data[:, :3], delimiter='\t')
    >>> np.savetxt('add.tsv', data[:, 3:], delimiter='\t')

    Join without naming headers:

    >>> join = JoinTSVColumns()
    >>> join.inputs.in_file = 'data.tsv'
    >>> join.inputs.join_file = 'add.tsv'
    >>> res = join.run()
    >>> df = pd.read_csv(res.outputs.out_file, delim_whitespace=True,
    ...                  index_col=None, dtype=float, header=None)
    >>> df.columns.ravel().tolist() == list(range(5))
    True

    >>> np.all(df.values.astype(int) == data)
    True


    Adding column names:

    >>> join = JoinTSVColumns()
    >>> join.inputs.in_file = 'data.tsv'
    >>> join.inputs.join_file = 'add.tsv'
    >>> join.inputs.columns = ['a', 'b', 'c', 'd', 'e']
    >>> res = join.run()
    >>> res.outputs.out_file  # doctest: +ELLIPSIS
    '...data_joined.tsv'
    >>> df = pd.read_csv(res.outputs.out_file, delim_whitespace=True,
    ...                  index_col=None)
    >>> df.columns.ravel().tolist()
    ['a', 'b', 'c', 'd', 'e']

    >>> np.all(df.values == np.arange(30).reshape((6, 5)))
    True

    >>> join = JoinTSVColumns()
    >>> join.inputs.in_file = 'data.tsv'
    >>> join.inputs.join_file = 'add.tsv'
    >>> join.inputs.side = 'left'
    >>> join.inputs.columns = ['a', 'b', 'c', 'd', 'e']
    >>> res = join.run()
    >>> df = pd.read_csv(res.outputs.out_file, delim_whitespace=True,
    ...                  index_col=None)
    >>> df.columns.ravel().tolist()
    ['a', 'b', 'c', 'd', 'e']

    >>> np.all(df.values == np.hstack((data[:, 3:], data[:, :3])))
    True

    .. testcleanup::

    >>> os.chdir(cwd)

    """
    input_spec = _JoinTSVColumnsInputSpec
    output_spec = _JoinTSVColumnsOutputSpec

    def _run_interface(self, runtime):
        out_file = fname_presuffix(
            self.inputs.in_file,
            suffix="_joined.tsv",
            newpath=runtime.cwd,
            use_ext=False,
        )

        header = ""
        if isdefined(self.inputs.columns) and self.inputs.columns:
            header = "\t".join(self.inputs.columns)

        with open(self.inputs.in_file) as ifh:
            data = ifh.read().splitlines(keepends=False)

        with open(self.inputs.join_file) as ifh:
            join = ifh.read().splitlines(keepends=False)

        if len(data) != len(join):
            raise ValueError("Number of columns in datasets do not match")

        merged = []
        for d, j in zip(data, join):
            line = "%s\t%s" % ((j, d) if self.inputs.side == "left" else (d, j))
            merged.append(line)

        if header:
            merged.insert(0, header)

        with open(out_file, "w") as ofh:
            ofh.write("\n".join(merged))

        self._results["out_file"] = out_file
        return runtime


class _DictMergeInputSpec(BaseInterfaceInputSpec):
    in_dicts = traits.List(
        traits.Either(traits.Dict, traits.Instance(OrderedDict)),
        desc="Dictionaries to be merged. In the event of a collision, values "
        "from dictionaries later in the list receive precedence.",
    )


class _DictMergeOutputSpec(TraitedSpec):
    out_dict = traits.Dict(desc="Merged dictionary")


class DictMerge(SimpleInterface):
    """Merge (ordered) dictionaries."""

    input_spec = _DictMergeInputSpec
    output_spec = _DictMergeOutputSpec

    def _run_interface(self, runtime):
        out_dict = {}
        for in_dict in self.inputs.in_dicts:
            out_dict.update(in_dict)
        self._results["out_dict"] = out_dict
        return runtime


class _TSV2JSONInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="Input TSV file")
    index_column = traits.Str(
        mandatory=True,
        desc="Name of the column in the TSV to be used "
        "as the top-level key in the JSON. All "
        "remaining columns will be assigned as "
        "nested keys.",
    )
    output = traits.Either(
        None,
        File,
        desc="Path where the output file is to be saved. "
        "If this is `None`, then a JSON-compatible "
        "dictionary is returned instead.",
    )
    additional_metadata = traits.Either(
        None,
        traits.Dict,
        traits.Instance(OrderedDict),
        usedefault=True,
        desc="Any additional metadata that "
        "should be applied to all "
        "entries in the JSON.",
    )
    drop_columns = traits.Either(
        None,
        traits.List(),
        usedefault=True,
        desc="List of columns in the TSV to be " "dropped from the JSON.",
    )
    enforce_case = traits.Bool(
        True,
        usedefault=True,
        desc="Enforce snake case for top-level keys " "and camel case for nested keys",
    )


class _TSV2JSONOutputSpec(TraitedSpec):
    output = traits.Either(
        traits.Dict,
        File(exists=True),
        traits.Instance(OrderedDict),
        desc="Output dictionary or JSON file",
    )


class TSV2JSON(SimpleInterface):
    """Convert metadata from TSV format to JSON format.
    """

    input_spec = _TSV2JSONInputSpec
    output_spec = _TSV2JSONOutputSpec

    def _run_interface(self, runtime):
        if not isdefined(self.inputs.output):
            output = fname_presuffix(
                self.inputs.in_file, suffix=".json", newpath=runtime.cwd, use_ext=False
            )
        else:
            output = self.inputs.output

        self._results["output"] = _tsv2json(
            in_tsv=self.inputs.in_file,
            out_json=output,
            index_column=self.inputs.index_column,
            additional_metadata=self.inputs.additional_metadata,
            drop_columns=self.inputs.drop_columns,
            enforce_case=self.inputs.enforce_case,
        )
        return runtime


def _tsv2json(
    in_tsv,
    out_json,
    index_column,
    additional_metadata=None,
    drop_columns=None,
    enforce_case=True,
):
    """
    Convert metadata from TSV format to JSON format.

    Parameters
    ----------
    in_tsv: str
        Path to the metadata in TSV format.
    out_json: str
        Path where the metadata should be saved in JSON format after
        conversion. If this is None, then a dictionary is returned instead.
    index_column: str
        Name of the column in the TSV to be used as an index (top-level key in
        the JSON).
    additional_metadata: dict
        Any additional metadata that should be applied to all entries in the
        JSON.
    drop_columns: list
        List of columns from the input TSV to be dropped from the JSON.
    enforce_case: bool
        Indicates whether BIDS case conventions should be followed. Currently,
        this means that index fields (column names in the associated data TSV)
        use snake case and other fields use camel case.

    Returns
    -------
    str
        Path to the metadata saved in JSON format.
    """
    import pandas as pd

    # Adapted from https://dev.to/rrampage/snake-case-to-camel-case-and- ...
    # back-using-regular-expressions-and-python-m9j
    re_to_camel = r"(.*?)_([a-zA-Z0-9])"
    re_to_snake = r"(^.+?|.*?)((?<![_A-Z])[A-Z]|(?<![_0-9])[0-9]+)"

    def snake(match):
        return "{}_{}".format(match.group(1).lower(), match.group(2).lower())

    def camel(match):
        return "{}{}".format(match.group(1), match.group(2).upper())

    # from fmriprep
    def less_breakable(a_string):
        """ hardens the string to different envs (i.e. case insensitive, no
        whitespace, '#' """
        return "".join(a_string.split()).strip("#")

    drop_columns = drop_columns or []
    additional_metadata = additional_metadata or {}
    tsv_data = pd.read_csv(in_tsv, "\t")
    for k, v in additional_metadata.items():
        tsv_data[k] = [v] * len(tsv_data.index)
    for col in drop_columns:
        tsv_data.drop(labels=col, axis="columns", inplace=True)
    tsv_data.set_index(index_column, drop=True, inplace=True)
    if enforce_case:
        tsv_data.index = [
            re.sub(re_to_snake, snake, less_breakable(i), 0).lower()
            for i in tsv_data.index
        ]
        tsv_data.columns = [
            re.sub(re_to_camel, camel, less_breakable(i).title(), 0).replace("Csf", "CSF")
            for i in tsv_data.columns
        ]
    json_data = tsv_data.to_json(orient="index")
    json_data = json.JSONDecoder(object_pairs_hook=OrderedDict).decode(json_data)
    for i in json_data:
        json_data[i].update(additional_metadata)

    if out_json is None:
        return json_data

    with open(out_json, "w") as f:
        json.dump(json_data, f, indent=4)
    return out_json


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
