#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities

"""

import nibabel as nb
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import File, BaseInterfaceInputSpec, TraitedSpec
from .base import SimpleInterface

class CopyHeaderInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='the file we get the data from')
    hdr_file = File(exists=True, mandatory=True, desc='the file we get the header from')

class CopyHeaderOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='written file path')

class CopyHeader(SimpleInterface):
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
