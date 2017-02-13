# -*- coding: utf-8 -*-
"""
ReportCapableInterfaces for segmentation tools


"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os

from nipype.interfaces import fsl, freesurfer
from niworkflows.common import report as nrc
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

    def _run_interface(self, runtime):
        if self.inputs.generate_report:
            self.inputs.segments = True

        return super(FASTRPT, self)._run_interface(runtime)

    def _post_run_hook(self, runtime):
        ''' generates a report showing nine slices, three per axis, of an
        arbitrary volume of `in_files`, with the resulting segmentation
        overlaid '''
        self._anat_file = self.inputs.in_files[0],
        self._mask_file = self.aggregate_outputs().tissue_class_map
        self._seg_files = self.aggregate_outputs().tissue_class_files
        self._masked = False
        self._report_title = "FAST: segmentation over anatomical"

        NIWORKFLOWS_LOG.info('Generating report for FAST (in_files %s, '
                             'segmentation %s, individual tissue classes %s).',
                             self.inputs.in_files,
                             self.aggregate_outputs().tissue_class_map,
                             self.aggregate_outputs().tissue_class_files)


class ReconAllInputSpecRPT(nrc.ReportCapableInputSpec,
                           freesurfer.preprocess.ReconAllInputSpec):
    pass

class ReconAllOutputSpecRPT(nrc.ReportCapableOutputSpec,
                            freesurfer.preprocess.ReconAllOutputSpec):
    pass

class ReconAllRPT(nrc.SurfaceSegmentationRC, freesurfer.preprocess.ReconAll):
    input_spec = ReconAllInputSpecRPT
    output_spec = ReconAllOutputSpecRPT

    def _post_run_hook(self, runtime):
        ''' generates a report showing nine slices, three per axis, of an
        arbitrary volume of `in_files`, with the resulting segmentation
        overlaid '''
        outputs = self.aggregate_outputs()
        self._anat_file = os.path.join(outputs.subjects_dir,
                                       outputs.subject_id,
                                       'mri', 'brain.mgz')
        self._contour = os.path.join(outputs.subjects_dir,
                                     outputs.subject_id,
                                     'mri', 'ribbon.mgz')
        self._masked = False
        self._report_title = "ReconAll: segmentation over anatomical"

        NIWORKFLOWS_LOG.info('Generating report for ReconAll (subject %s)',
                             outputs.subject_id)
