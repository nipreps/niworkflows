# -*- coding: utf-8 -*-
"""
ReportCapableInterfaces for masks tools

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
from nipype.interfaces import fsl, ants
from nipype.interfaces.base import File, BaseInterfaceInputSpec, traits, isdefined
from niworkflows.common import report as nrc
from niworkflows import NIWORKFLOWS_LOG
from nilearn.masking import compute_epi_mask
import scipy.ndimage as nd
import numpy as np
from nilearn.image import new_img_like

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
        self._mask_file = self.aggregate_outputs().mask_file
        self._seg_files = [self._mask_file]
        self._masked = self.inputs.mask
        self._report_title = "BET: brain mask over anatomical input"

        NIWORKFLOWS_LOG.info('Generating report for BET. file "%s", and mask file "%s"',
                             self._anat_file, self._mask_file)

# TODO: move this interface to nipype.interfaces.nilearn

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

        brain_extraction_mask = self.aggregate_outputs().BrainExtractionMask

        self._anat_file = self.inputs.anatomical_image
        self._mask_file = brain_extraction_mask
        self._seg_files = [brain_extraction_mask]
        self._masked = False
        self._report_title = 'ANTS BrainExtraction: brain mask over anatomical input'

        NIWORKFLOWS_LOG.info('Generating report for ANTS BrainExtraction. file "%s", mask "%s"',
                             self._anat_file, self._mask_file)


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
        mask_nii = compute_epi_mask(self.inputs.in_file)

        mask_data = mask_nii.get_data()
        if isdefined(self.inputs.dilation):
            mask_data = nd.morphology.binary_dilation(mask_data).astype(np.uint8)
        better_mask = new_img_like(mask_nii, mask_data)
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
        self._mask_file = self.aggregate_outputs().mask_file
        self._seg_files = [self._mask_file]
        self._masked = True
        self._report_title = "nilearn.compute_epi_mask: brain mask over EPI input"

        NIWORKFLOWS_LOG.info('Generating report for nilearn.compute_epi_mask. file "%s", and mask file "%s"',
                             self._anat_file, self._mask_file)
