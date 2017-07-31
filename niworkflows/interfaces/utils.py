#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities

"""
from __future__ import absolute_import, unicode_literals

import os
import numpy as np
import nibabel as nb
from niworkflows.nipype.utils.filemanip import fname_presuffix
from niworkflows.nipype.utils.misc import normalize_mc_params
from niworkflows.nipype.interfaces.base import (
    File, BaseInterfaceInputSpec, TraitedSpec, traits
)

from .base import SimpleInterface


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
