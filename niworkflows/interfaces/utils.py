#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities

"""

import os.path as op
import nibabel as nb
from nipype.interfaces.base import File, OutputMultiPath, BaseInterfaceInputSpec, TraitedSpec

from .base import SimpleInterface

class CopyHeaderInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='the file we get the data from')
    hdr_file = File(exists=True, mandatory=True, desc='the file we get the header from')

class CopyHeaderOutputSpec(TraitedSpec):
    out_file = OutputMultiPath(File(exists=True, desc='written file path'))

class CopyHeader(SimpleInterface):
    input_spec = CopyHeaderInputSpec
    output_spec = CopyHeaderOutputSpec

    def _run_interface(self, runtime):

        hdr = nb.load(self.inputs.hdr_file).get_header().copy()
        aff = nb.load(self.inputs.hdr_file).get_affine()
        data = nb.load(self.inputs.in_file).get_data()

        fname, ext = op.splitext(op.basename(self.inputs.in_file))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext

        out_name = op.abspath('{}_fixhdr{}'.format(fname, ext))
        nb.Nifti1Image(data, aff, hdr).to_filename(out_name)
        self._results['out_file'] = out_name
        return runtime
