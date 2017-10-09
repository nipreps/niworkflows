#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities

"""
from __future__ import absolute_import, unicode_literals

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
    fixed_image = File(exists=True, mandatory=True, desc='the reference file')
    moving_image = File(exists=True, mandatory=True, desc='the pixel size reference')
    xform_code = traits.Enum(None, 2, 4, usedefault=True,
                             desc='force xform code')


class GenerateSamplingReferenceOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='one file with all inputs flattened')


class GenerateSamplingReference(SimpleInterface):
    """
    Generates a reference grid for resampling one image keeping original resolution,
    but moving data to a different space (e.g. MNI)
    """

    input_spec = GenerateSamplingReferenceInputSpec
    output_spec = GenerateSamplingReferenceOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = _gen_reference(
            self.inputs.fixed_image,
            self.inputs.moving_image,
            force_xform_code=self.inputs.xform_code,
            message='%s (niworkflows v%s)' % (self.__class__.__name__, __version__))
        return runtime


def _copyxform(ref_image, out_image, message=None):
    # Read in reference and output
    resampled = nb.load(out_image)
    orig = nb.load(ref_image)

    if not np.allclose(orig.affine, resampled.affine):
        LOG.warning('Affines of input and reference images '
                    'do not match, CopyXForm will probably '
                    'make the input image useless.')

    # Copy xform infos
    qform, qform_code = orig.header.get_qform(coded=True)
    sform, sform_code = orig.header.get_sform(coded=True)
    header = resampled.header.copy()
    header.set_qform(qform, int(qform_code))
    header.set_sform(sform, int(sform_code))
    header['descrip'] = 'xform matrices modified by %s.' % (message or '(unknown)')

    newimg = resampled.__class__(resampled.get_data(), orig.affine, header)
    newimg.to_filename(out_image)


def _gen_reference(fixed_image, moving_image, out_file=None, message=None,
                   force_xform_code=None):
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
    new_zooms_round = np.round(new_zooms, 3)

    resampled = nli.resample_img(fixed_image,
                                 target_affine=np.diag(new_zooms_round),
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
