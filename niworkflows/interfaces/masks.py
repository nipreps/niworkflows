# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
ReportCapableInterfaces for masks tools

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import nibabel as nb
from nilearn.masking import compute_epi_mask
import scipy.ndimage as nd

from .. import NIWORKFLOWS_LOG
from ..nipype.interfaces import fsl, ants
from ..nipype.interfaces.base import (
    File, BaseInterfaceInputSpec, traits, isdefined)
from ..nipype.algorithms import confounds
from . import report_base as nrc


class BETInputSpecRPT(nrc.ReportCapableInputSpec,
                      fsl.preprocess.BETInputSpec):
    pass


class BETOutputSpecRPT(nrc.ReportCapableOutputSpec,
                       fsl.preprocess.BETOutputSpec):
    pass


class BETRPT(nrc.SegmentationRC, fsl.BET):
    input_spec = BETInputSpecRPT
    output_spec = BETOutputSpecRPT

    def _run_interface(self, runtime):
        if self.inputs.generate_report:
            self.inputs.mask = True

        return super(BETRPT, self)._run_interface(runtime)

    def _post_run_hook(self, runtime):
        ''' generates a report showing slices from each axis of an arbitrary
        volume of in_file, with the resulting binary brain mask overlaid '''

        self._anat_file = self.inputs.in_file
        self._mask_file = self.aggregate_outputs(runtime=runtime).mask_file
        self._seg_files = [self._mask_file]
        self._masked = self.inputs.mask
        self._report_title = "BET: brain mask over anatomical input"

        NIWORKFLOWS_LOG.info('Generating report for BET. file "%s", and mask file "%s"',
                             self._anat_file, self._mask_file)


class BrainExtractionInputSpecRPT(nrc.ReportCapableInputSpec,
                                  ants.segmentation.BrainExtractionInputSpec):
    pass


class BrainExtractionOutputSpecRPT(nrc.ReportCapableOutputSpec,
                                   ants.segmentation.BrainExtractionOutputSpec):
    pass


class BrainExtractionRPT(nrc.SegmentationRC, ants.segmentation.BrainExtraction):
    input_spec = BrainExtractionInputSpecRPT
    output_spec = BrainExtractionOutputSpecRPT

    def _post_run_hook(self, runtime):
        ''' generates a report showing slices from each axis '''

        brain_extraction_mask = self.aggregate_outputs(runtime=runtime).BrainExtractionMask

        if isdefined(self.inputs.keep_temporary_files) and self.inputs.keep_temporary_files == 1:
            self._anat_file = self.aggregate_outputs(runtime=runtime).N4Corrected0
        else:
            self._anat_file = self.inputs.anatomical_image
        self._mask_file = brain_extraction_mask
        self._seg_files = [brain_extraction_mask]
        self._masked = False
        self._report_title = 'ANTS BrainExtraction: brain mask over anatomical input'

        NIWORKFLOWS_LOG.info('Generating report for ANTS BrainExtraction. file "%s", mask "%s"',
                             self._anat_file, self._mask_file)


# TODO: move this interface to nipype.interfaces.nilearn
class ComputeEPIMaskInputSpec(nrc.ReportCapableInputSpec,
                              BaseInterfaceInputSpec):
    in_file = File(exists=True, desc="3D or 4D EPI file")
    dilation = traits.Int(desc="binary dilation on the nilearn output")


class ComputeEPIMaskOutputSpec(nrc.ReportCapableOutputSpec):
    mask_file = File(exists=True, desc="Binary brain mask")


class ComputeEPIMask(nrc.SegmentationRC):
    input_spec = ComputeEPIMaskInputSpec
    output_spec = ComputeEPIMaskOutputSpec

    def _run_interface(self, runtime):
        orig_file_nii = nb.load(self.inputs.in_file)
        in_file_data = orig_file_nii.get_data()

        # pad the data to avoid the mask estimation running into edge effects
        in_file_data_padded = np.pad(in_file_data, (1, 1), 'constant',
                                     constant_values=(0, 0))

        padded_nii = nb.Nifti1Image(in_file_data_padded, orig_file_nii.affine,
                                    orig_file_nii.header)

        mask_nii = compute_epi_mask(padded_nii, exclude_zeros=True)

        mask_data = mask_nii.get_data()
        if isdefined(self.inputs.dilation):
            mask_data = nd.morphology.binary_dilation(mask_data).astype(np.uint8)

        # reverse image padding
        mask_data = mask_data[1:-1, 1:-1, 1:-1]

        # exclude zero and NaN voxels
        mask_data[in_file_data == 0] = 0
        mask_data[np.isnan(in_file_data)] = 0

        better_mask = nb.Nifti1Image(mask_data, orig_file_nii.affine,
                                     orig_file_nii.header)
        better_mask.set_data_dtype(np.uint8)
        better_mask.to_filename("mask_file.nii.gz")

        self._mask_file = os.path.abspath("mask_file.nii.gz")

        runtime.returncode = 0
        return super(ComputeEPIMask, self)._run_interface(runtime)

    def _list_outputs(self):
        outputs = super(ComputeEPIMask, self)._list_outputs()
        outputs['mask_file'] = self._mask_file
        return outputs

    def _post_run_hook(self, runtime):
        ''' generates a report showing slices from each axis of an arbitrary
        volume of in_file, with the resulting binary brain mask overlaid '''

        self._anat_file = self.inputs.in_file
        self._mask_file = self.aggregate_outputs(runtime=runtime).mask_file
        self._seg_files = [self._mask_file]
        self._masked = True
        self._report_title = "nilearn.compute_epi_mask: brain mask over EPI input"

        NIWORKFLOWS_LOG.info(
            'Generating report for nilearn.compute_epi_mask. file "%s", and mask file "%s"',
            self._anat_file, self._mask_file)


class ACompCorInputSpecRPT(nrc.ReportCapableInputSpec,
                           confounds.CompCorInputSpec):
    pass


class ACompCorOutputSpecRPT(nrc.ReportCapableOutputSpec,
                            confounds.CompCorOutputSpec):
    pass


class ACompCorRPT(nrc.SegmentationRC, confounds.ACompCor):
    input_spec = ACompCorInputSpecRPT
    output_spec = ACompCorOutputSpecRPT

    def _post_run_hook(self, runtime):
        ''' generates a report showing slices from each axis '''

        assert len(self.inputs.mask_files) == 1, \
            "ACompCorRPT only supports a single input mask. " \
            "A list %s was found." % self.inputs.mask_files
        self._anat_file = self.inputs.realigned_file
        self._mask_file = self.inputs.mask_files[0]
        self._seg_files = self.inputs.mask_files
        self._masked = False
        self._report_title = 'aCompCor ROI'

        NIWORKFLOWS_LOG.info('Generating report for aCompCor. file "%s", mask "%s"',
                             self.inputs.realigned_file, self._mask_file)


class TCompCorInputSpecRPT(nrc.ReportCapableInputSpec,
                           confounds.TCompCorInputSpec):
    pass


class TCompCorOutputSpecRPT(nrc.ReportCapableOutputSpec,
                            confounds.TCompCorOutputSpec):
    pass


class TCompCorRPT(nrc.SegmentationRC, confounds.TCompCor):
    input_spec = TCompCorInputSpecRPT
    output_spec = TCompCorOutputSpecRPT

    def _post_run_hook(self, runtime):
        ''' generates a report showing slices from each axis '''

        high_variance_masks = self.aggregate_outputs(runtime=runtime).high_variance_masks

        assert not isinstance(high_variance_masks, list),\
            "TCompCorRPT only supports a single output high variance mask. " \
            "A list %s was found." % str(high_variance_masks)
        self._anat_file = self.inputs.realigned_file
        self._mask_file = high_variance_masks
        self._seg_files = [high_variance_masks]
        self._masked = False
        self._report_title = 'tCompCor - high variance voxels'

        NIWORKFLOWS_LOG.info('Generating report for tCompCor. file "%s", mask "%s"',
                             self.inputs.realigned_file,
                             self.aggregate_outputs(runtime=runtime).high_variance_masks)


class SimpleShowMaskInputSpec(nrc.ReportCapableInputSpec):
    background_file = File(exists=True, mandatory=True, desc='file before')
    mask_file = File(exists=True, mandatory=True, desc='file before')
    generate_report = traits.Bool(True, usedefault=True)


class SimpleShowMaskRPT(nrc.SegmentationRC):
    input_spec = SimpleShowMaskInputSpec
    output_spec = nrc.ReportCapableOutputSpec

    def _post_run_hook(self, runtime):
        self._anat_file = self.inputs.background_file
        self._mask_file = self.inputs.mask_file
        self._seg_files = [self.inputs.mask_file]
        self._masked = True
        self._report_title = "Brain mask"

        return runtime
