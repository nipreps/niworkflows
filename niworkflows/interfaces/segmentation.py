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
