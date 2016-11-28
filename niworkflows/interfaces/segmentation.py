# -*- coding: utf-8 -*-
"""
ReportCapableInterfaces for segmentation tools


"""
from __future__ import absolute_import, division, print_function


from nipype.interfaces import fsl
from nipype.interfaces.base import traits, isdefined
from niworkflows.common import report as nrc
from niworkflows.viz.utils import plot_mask
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

    def _generate_report(self):
        ''' generates a report showing nine slices, three per axis, of an
        arbitrary volume of `in_files`, with the resulting segmentation
        overlaid '''

        NIWORKFLOWS_LOG.info('Generating report for FAST (in_files {}, segmentation {}).'.
                             format(self.inputs.in_files,
                                    self.aggregate_outputs().tissue_class_map))

        plot_mask(
            self.inputs.in_files,
            self.aggregate_outputs().tissue_class_map,
            out_file=self._out_report,
            title="FAST: segmentation over anatomical"
        )

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

    def _generate_report(self):
        ''' generates a report showing nine slices, three per axis,
        of an arbitrary
        volume of in_file, with the resulting binary brain mask overlaid '''

        mask_file = self.aggregate_outputs().out_file
        if self.inputs.mask:
            mask_file = self.aggregate_outputs().mask_file

        NIWORKFLOWS_LOG.info('Generating report for file "%s", and mask file "%s"',
                             self.inputs.in_file, mask_file)
        plot_mask(
            self.inputs.in_file, mask_file,
            out_file=self._out_report, masked=self.inputs.mask,
            title="BET: brain mask over anatomical input"
        )
