# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Nibabel-based interfaces."""
import numpy as np
import nibabel as nb
from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File,
    SimpleInterface, OutputMultiObject, InputMultiObject
)

IFLOGGER = logging.getLogger('nipype.interface')


class _ApplyMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='an image')
    in_mask = File(exists=True, mandatory=True, desc='a mask')
    threshold = traits.Float(0.5, usedefault=True,
                             desc='a threshold to the mask, if it is nonbinary')


class _ApplyMaskOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='masked file')


class ApplyMask(SimpleInterface):
    """Mask the input given a mask."""

    input_spec = _ApplyMaskInputSpec
    output_spec = _ApplyMaskOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        msknii = nb.load(self.inputs.in_mask)
        msk = msknii.get_fdata() > self.inputs.threshold

        self._results['out_file'] = fname_presuffix(
            self.inputs.in_file, suffix='_masked', newpath=runtime.cwd)

        if img.dataobj.shape[:3] != msk.shape:
            raise ValueError("Image and mask sizes do not match.")

        if not np.allclose(img.affine, msknii.affine):
            raise ValueError("Image and mask affines are not similar enough.")

        if img.dataobj.ndim == msk.ndim + 1:
            msk = msk[..., np.newaxis]

        masked = img.__class__(img.dataobj * msk, None, img.header)
        masked.to_filename(self._results['out_file'])
        return runtime


class _BinarizeInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input image')
    thresh_low = traits.Float(mandatory=True,
                              desc='non-inclusive lower threshold')


class _BinarizeOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='masked file')
    out_mask = File(exists=True, desc='output mask')


class Binarize(SimpleInterface):
    """Binarizes the input image applying the given thresholds."""

    input_spec = _BinarizeInputSpec
    output_spec = _BinarizeOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)

        self._results['out_file'] = fname_presuffix(
            self.inputs.in_file, suffix='_masked', newpath=runtime.cwd)
        self._results['out_mask'] = fname_presuffix(
            self.inputs.in_file, suffix='_mask', newpath=runtime.cwd)

        data = img.get_fdata()
        mask = data > self.inputs.thresh_low
        data[~mask] = 0.0
        masked = img.__class__(data, img.affine, img.header)
        masked.to_filename(self._results['out_file'])

        img.header.set_data_dtype('uint8')
        maskimg = img.__class__(mask.astype('uint8'), img.affine,
                                img.header)
        maskimg.to_filename(self._results['out_mask'])

        return runtime


class _FourToThreeInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input 4d image')


class _FourToThreeOutputSpec(TraitedSpec):
    out_files = OutputMultiObject(File(exists=True),
                                     desc='output list of 3d images')


class FourToThree(SimpleInterface):
    """Split a 4D dataset along the last dimension
    into a series of 3D volumes."""

    input_spec = _FourToThreeInputSpec
    output_spec = _FourToThreeOutputSpec

    def _run_interface(self, runtime):
        filenii = nb.load(self.inputs.in_file)
        if len(filenii.shape) != 4:
            raise RuntimeError('Input image (%s) is not 4D.' % filenii)

        files_3d = nb.four_to_three(filenii)
        self._results['out_files'] = []
        for i, file_3d in enumerate(files_3d):
            out_file = fname_presuffix(in_file, suffix=f"_idx-{i:03}")
            file_3d.to_filename(out_file)
            out_files.append(out_file)

        self._results['out_files'] = out_files
        return runtime


class _MergeSeriesInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiObject(File(exists=True, mandatory=True,
                                     desc='input list of 3d images'))


class _MergeSeriesOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output 4d image')


class MergeSeries(SimpleInterface):
    """Merge a series of 3D volumes along the last dimension into a single
    4D image."""

    input_spec = _MergeSeriesInputSpec
    output_spec = _MergeSeriesOutputSpec

    def _run_interface(self, runtime):
        nii_list = []
        for f in self.inputs.in_files:
            filenii = nb.load(f)
            filenii = nb.squeeze_image(filenii)
            if len(filenii.shape) != 3:
                raise RuntimeError('Input image (%s) is not 3D.' % f)
            else:
                nii_list.append(filenii)
        img_4d = nb.funcs.concat_images(nii_list)
        out_file = fname_presuffix(self.inputs.in_files[0], suffix="_merged")
        img_4d.to_filename(out_file)

        self._results['out_file'] = out_file
        return runtime
