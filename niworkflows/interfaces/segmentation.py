# -*- coding: utf-8 -*-
"""
ReportCapableInterfaces for segmentation tools


"""
from __future__ import absolute_import, division, print_function


from nipype.interfaces import fsl
from nipype.interfaces.base import traits, isdefined
from niworkflows.common import report as nrc
from niworkflows.viz.utils import plot_segs
from niworkflows import NIWORKFLOWS_LOG

class FASTInputSpecRPT(nrc.ReportCapableInputSpec,
                       fsl.preprocess.FASTInputSpec):
    pass

class FASTOutputSpecRPT(nrc.ReportCapableOutputSpec,
                        fsl.preprocess.FASTOutputSpec):
    pass

class FASTRPT(nrc.SegmentationRC,
              fsl.FAST):
    input_spec = FASTInputSpecRPT
    output_spec = FASTOutputSpecRPT

    def _post_run_hook(self, runtime):
        ''' generates a report showing nine slices, three per axis, of an
        arbitrary volume of `in_files`, with the resulting segmentation
        overlaid '''
        self._anat_file = self.inputs.in_files[0],
        self._mask_file = self.aggregate_outputs().tissue_class_map
        self._seg_files = self.aggregate_outputs().tissue_class_files
        self._masked = False
        self._report_title = "FAST: segmentation over anatomical"

        NIWORKFLOWS_LOG.info('Generating report for FAST (in_files {}, segmentation {}, individual tissue classes {}).'.
                             format(self.inputs.in_files,
                                    self.aggregate_outputs().tissue_class_map,
                                    self.aggregate_outputs().tissue_class_files))


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
        ''' generates a report showing nine slices, three per axis,
        of an arbitrary
        volume of in_file, with the resulting binary brain mask overlaid '''

        self._anat_file = self.inputs.in_file
        self._mask_file = self.aggregate_outputs().mask_file
        self._seg_files = [self._mask_file]
        self._masked = self.inputs.mask
        self._report_title = "BET: brain mask over anatomical input"

        NIWORKFLOWS_LOG.info('Generating report for BET. file "%s", and mask file "%s"',
                             self._anat_file, self._mask_file)
