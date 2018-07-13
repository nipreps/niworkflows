# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import shutil
import numpy as np
import nibabel as nb
import nilearn.image as nli
from textwrap import indent

from .. import __version__
from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.utils.misc import normalize_mc_params
from nipype.interfaces.base import (
    File, BaseInterfaceInputSpec, TraitedSpec, traits, SimpleInterface
)


LOG = logging.getLogger('nipype.interface')


class CopyXFormInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='the file we get the data from')
    hdr_file = File(exists=True, mandatory=True, desc='the file we get the header from')


class CopyXFormOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='written file path')


class CopyXForm(SimpleInterface):
    """
    Copy the x-form matrices from `hdr_file` to `out_file`.
    """
    input_spec = CopyXFormInputSpec
    output_spec = CopyXFormOutputSpec

    def _run_interface(self, runtime):
        out_name = fname_presuffix(self.inputs.in_file,
                                   suffix='_xform',
                                   newpath=runtime.cwd)
        # Copy and replace header
        shutil.copy(self.inputs.in_file, out_name)
        _copyxform(self.inputs.hdr_file, out_name,
                   message='CopyXForm (niworkflows v%s)' % __version__)
        self._results['out_file'] = out_name
        return runtime


class CopyHeaderInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='the file we get the data from')
    hdr_file = File(exists=True, mandatory=True, desc='the file we get the header from')


class CopyHeaderOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='written file path')


class CopyHeader(SimpleInterface):
    """
    Copy a header from the `hdr_file` to `out_file` with data drawn from
    `in_file`.
    """
    input_spec = CopyHeaderInputSpec
    output_spec = CopyHeaderOutputSpec

    def _run_interface(self, runtime):
        in_img = nb.load(self.inputs.hdr_file)
        out_img = nb.load(self.inputs.in_file)
        new_img = out_img.__class__(out_img.get_data(), in_img.affine, in_img.header)
        new_img.set_data_dtype(out_img.get_data_dtype())

        out_name = fname_presuffix(self.inputs.in_file,
                                   suffix='_fixhdr', newpath='.')
        new_img.to_filename(out_name)
        self._results['out_file'] = out_name
        return runtime


class NormalizeMotionParamsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='the input parameters file')
    format = traits.Enum('FSL', 'AFNI', 'FSFAST', 'NIPY', usedefault=True,
                         desc='output format')


class NormalizeMotionParamsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='written file path')


class NormalizeMotionParams(SimpleInterface):
    """
    Convert input motion parameters into the designated convention.

    """
    input_spec = NormalizeMotionParamsInputSpec
    output_spec = NormalizeMotionParamsOutputSpec

    def _run_interface(self, runtime):
        mpars = np.loadtxt(self.inputs.in_file)  # mpars is N_t x 6
        mpars = np.apply_along_axis(
            func1d=normalize_mc_params,
            axis=1, arr=mpars,
            source=self.inputs.format)
        self._results['out_file'] = os.path.join(runtime.cwd, "motion_params.txt")
        np.savetxt(self._results['out_file'], mpars)
        return runtime


class GenerateSamplingReferenceInputSpec(BaseInterfaceInputSpec):
    fixed_image = File(exists=True, mandatory=True,
                       desc='the reference file, defines the FoV')
    moving_image = File(exists=True, mandatory=True, desc='the pixel size reference')
    xform_code = traits.Enum(None, 2, 4, usedefault=True,
                             desc='force xform code')
    fov_mask = traits.Either(None, File(exists=True), usedefault=True,
                             desc='mask to clip field of view (in fixed_image space)')


class GenerateSamplingReferenceOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='one file with all inputs flattened')


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

    input_spec = GenerateSamplingReferenceInputSpec
    output_spec = GenerateSamplingReferenceOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = _gen_reference(
            self.inputs.fixed_image,
            self.inputs.moving_image,
            fov_mask=self.inputs.fov_mask,
            force_xform_code=self.inputs.xform_code,
            message='%s (niworkflows v%s)' % (self.__class__.__name__, __version__))
        return runtime


def _copyxform(ref_image, out_image, message=None):
    # Read in reference and output
    # Use mmap=False because we will be overwriting the output image
    resampled = nb.load(out_image, mmap=False)
    orig = nb.load(ref_image)

    if not np.allclose(orig.affine, resampled.affine):
        LOG.debug(
            'Affines of input and reference images do not match, '
            'FMRIPREP will set the reference image headers. '
            'Please, check that the x-form matrices of the input dataset'
            'are correct and manually verify the alignment of results.')

    # Copy xform infos
    qform, qform_code = orig.header.get_qform(coded=True)
    sform, sform_code = orig.header.get_sform(coded=True)
    header = resampled.header.copy()
    header.set_qform(qform, int(qform_code))
    header.set_sform(sform, int(sform_code))
    header['descrip'] = 'xform matrices modified by %s.' % (message or '(unknown)')

    newimg = resampled.__class__(resampled.get_data(), orig.affine, header)
    newimg.to_filename(out_image)


def _gen_reference(fixed_image, moving_image, fov_mask=None, out_file=None,
                   message=None, force_xform_code=None):
    """
    Generates a sampling reference, and makes sure xform matrices/codes are
    correct
    """

    if out_file is None:
        out_file = fname_presuffix(fixed_image,
                                   suffix='_reference',
                                   newpath=os.getcwd())

    new_zooms = nli.load_img(moving_image).header.get_zooms()[:3]
    # Avoid small differences in reported resolution to cause changes to
    # FOV. See https://github.com/poldracklab/fmriprep/issues/512
    new_affine = np.diag(np.round(new_zooms, 3))

    resampled = nli.resample_img(fixed_image,
                                 target_affine=new_affine,
                                 interpolation='nearest')

    if fov_mask is not None:
        # If we have a mask, resample again dropping (empty) samples
        # out of the FoV.
        fixednii = nb.load(fixed_image)
        masknii = nb.load(fov_mask)

        if np.all(masknii.shape[:3] != fixednii.shape[:3]):
            raise RuntimeError(
                'Fixed image and mask do not have the same dimensions.')

        if not np.allclose(masknii.affine, fixednii.affine, atol=1e-5):
            raise RuntimeError(
                'Fixed image and mask have different affines')

        # Get mask into reference space
        masknii = nli.resample_img(fixed_image,
                                   target_affine=new_affine,
                                   interpolation='nearest')
        res_shape = np.array(masknii.shape[:3])

        # Calculate a bounding box for the input mask
        # with an offset of 2 voxels per face
        bbox = np.argwhere(masknii.get_data() > 0)
        new_origin = np.clip(bbox.min(0) - 2, a_min=0, a_max=None)
        new_end = np.clip(bbox.max(0) + 2, a_min=0,
                          a_max=res_shape - 1)

        # Find new origin, and set into new affine
        new_affine_4 = resampled.affine.copy()
        new_affine_4[:3, 3] = new_affine_4[:3, :3].dot(
            new_origin) + new_affine_4[:3, 3]

        # Calculate new shapes
        new_shape = new_end - new_origin + 1
        resampled = nli.resample_img(fixed_image,
                                     target_affine=new_affine_4,
                                     target_shape=new_shape.tolist(),
                                     interpolation='nearest')

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
    resampled.header['descrip'] = 'reference image generated by %s.' % (
        message or '(unknown software)')
    resampled.to_filename(out_file)
    return out_file


class SanitizeImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input image')
    n_volumes_to_discard = traits.Int(0, usedefault=True, desc='discard n first volumes')
    max_32bit = traits.Bool(False, usedefault=True, desc='cast data to float32 if higher '
                                                         'precision is encountered')


class SanitizeImageOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='validated image')
    out_report = File(exists=True, desc='HTML segment containing warning')


class SanitizeImage(SimpleInterface):
    """
    Check the correctness of x-form headers (matrix and code) and fixes
    problematic combinations of values. Removes any extension form the header
    if present.
    This interface implements the `following logic
    <https://github.com/poldracklab/fmriprep/issues/873#issuecomment-349394544>`_:
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
    input_spec = SanitizeImageInputSpec
    output_spec = SanitizeImageOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        out_report = os.path.join(runtime.cwd, 'report.html')

        # Retrieve xform codes
        sform_code = int(img.header._structarr['sform_code'])
        qform_code = int(img.header._structarr['qform_code'])

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
        warning_txt = ''

        # Both match, qform valid (implicit with match), codes okay -> do nothing, empty report
        if matching_affines and qform_code > 0 and sform_code > 0:
            self._results['out_file'] = self.inputs.in_file
            open(out_report, 'w').close()

        # Row 2:
        elif valid_qform and qform_code > 0:
            img.set_sform(img.get_qform(), qform_code)
            save_file = True
            warning_txt = 'Note on orientation: sform matrix set'
            description = """\
<p class="elem-desc">The sform has been copied from qform.</p>
"""
        # Rows 3-4:
        # Note: if qform is not valid, matching_affines is False
        elif sform_code > 0 and (not matching_affines or qform_code == 0):
            img.set_qform(img.get_sform(), sform_code)
            save_file = True
            warning_txt = 'Note on orientation: qform matrix overwritten'
            description = """\
<p class="elem-desc">The qform has been copied from sform.</p>
"""
            if not valid_qform and qform_code > 0:
                warning_txt = 'WARNING - Invalid qform information'
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
            img.set_sform(affine, nb.nifti1.xform_codes['scanner'])
            img.set_qform(affine, nb.nifti1.xform_codes['scanner'])
            save_file = True
            warning_txt = 'WARNING - Missing orientation information'
            description = """\
<p class="elem-desc">
    Orientation information could not be retrieved from the image header.
    The qform and sform matrices have been set to a default, LAS-oriented affine.
    Analyses of this dataset MAY BE INVALID.
</p>
"""

        if (self.inputs.max_32bit and
                np.dtype(img.get_data_dtype()).itemsize > 4) or self.inputs.n_volumes_to_discard:
            # force float32 only if 64 bit dtype is detected
            if (self.inputs.max_32bit and np.dtype(img.get_data_dtype()).itemsize > 4):
                in_data = img.get_fdata(dtype=np.float32)
            else:
                in_data = img.dataobj

            img = nb.Nifti1Image(in_data[:, :, :, self.inputs.n_volumes_to_discard:],
                                 img.affine,
                                 img.header)
            save_file = True

        if len(img.header.extensions) != 0:
            img.header.extensions.clear()
            save_file = True

        # Store new file
        if save_file:
            out_fname = fname_presuffix(self.inputs.in_file, suffix='_valid',
                                        newpath=runtime.cwd)
            self._results['out_file'] = out_fname
            img.to_filename(out_fname)

        if warning_txt:
            snippet = '<h3 class="elem-title">%s</h3>\n%s\n' % (
                warning_txt, description)
            with open(out_report, 'w') as fobj:
                fobj.write(indent(snippet, '\t' * 3))

        self._results['out_report'] = out_report
        return runtime
