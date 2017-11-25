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

from .. import __version__
from ..nipype import logging
from ..nipype.utils.filemanip import fname_presuffix
from ..nipype.utils.misc import normalize_mc_params
from ..nipype.interfaces.base import (
    File, BaseInterfaceInputSpec, TraitedSpec, traits, SimpleInterface
)


LOG = logging.getLogger('interface')


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
        self._results['out_file'] = os.path.abspath("motion_params.txt")
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
    resampled = nb.load(out_image)
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
