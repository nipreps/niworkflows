# -*- coding: utf-8 -*-
"""
ReportCapableInterfaces for masks tools

"""
from __future__ import absolute_import, division, print_function, unicode_literals

from nipype.interfaces import fsl, ants
from niworkflows.common import report as nrc
from niworkflows import NIWORKFLOWS_LOG

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
