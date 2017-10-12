# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
ReportCapableInterfaces for segmentation tools


"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os

from ..nipype.interfaces.base import File
from ..nipype.interfaces import fsl, freesurfer
from ..common import report as nrc
from .. import NIWORKFLOWS_LOG


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
        self._mask_file = self.aggregate_outputs(runtime=runtime).tissue_class_map
        # We are skipping the CSF class because with combination with others
        # it only shows the skullstriping mask
        self._seg_files = self.aggregate_outputs(runtime=runtime).tissue_class_files[1:]
        self._masked = False
        self._report_title = "FAST: segmentation over anatomical"

        NIWORKFLOWS_LOG.info('Generating report for FAST (in_files %s, '
                             'segmentation %s, individual tissue classes %s).',
                             self.inputs.in_files,
                             self.aggregate_outputs(runtime=runtime).tissue_class_map,
                             self.aggregate_outputs(runtime=runtime).tissue_class_files)


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
        outputs = self.aggregate_outputs(runtime=runtime)
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


class MELODICInputSpecRPT(nrc.ReportCapableInputSpec,
                          fsl.model.MELODICInputSpec):
    out_report = File(
        'melodic_reportlet.svg', usedefault=True, desc='Filename for the visual'
                                                       ' report generated '
                                                       'by Nipype.')
    report_mask = File(desc='Mask used to draw the outline on the reportlet. '
                            'If not set the mask will be derived from the data.')


class MELODICOutputSpecRPT(nrc.ReportCapableOutputSpec,
                           fsl.model.MELODICOutputSpec):
    pass


class MELODICRPT(nrc.ReportCapableInterface, fsl.MELODIC):
    input_spec = MELODICInputSpecRPT
    output_spec = MELODICOutputSpecRPT

    def _generate_report(self):
        from niworkflows.viz.utils import plot_melodic_components
        plot_melodic_components(melodic_dir=self._melodic_dir,
                                in_file=self.inputs.in_files[0],
                                tr=self.inputs.tr_sec,
                                out_file=self.inputs.out_report,
                                compress=self.inputs.compress_report,
                                report_mask=self.inputs.report_mask)

    def _post_run_hook(self, runtime):
        ''' generates a report showing nine slices, three per axis, of an
        arbitrary volume of `in_files`, with the resulting segmentation
        overlaid '''
        outputs = self.aggregate_outputs(runtime=runtime)
        self._melodic_dir = outputs.out_dir

        NIWORKFLOWS_LOG.info('Generating report for MELODIC')


class ICA_AROMAInputSpecRPT(nrc.ReportCapableInputSpec,
                            fsl.aroma.ICA_AROMAInputSpec):
    out_report = File(
        'ica_aroma_reportlet.svg', usedefault=True, desc='Filename for the visual'
                                                         ' report generated '
                                                         'by Nipype.')
    report_mask = File(desc='Mask used to draw the outline on the reportlet. '
                            'If not set the mask will be derived from the data.')


class ICA_AROMAOutputSpecRPT(nrc.ReportCapableOutputSpec,
                             fsl.aroma.ICA_AROMAOutputSpec):
    pass


class ICA_AROMARPT(nrc.ReportCapableInterface, fsl.ICA_AROMA):
    input_spec = ICA_AROMAInputSpecRPT
    output_spec = ICA_AROMAOutputSpecRPT

    def _generate_report(self):
        from niworkflows.viz.utils import plot_melodic_components
        plot_melodic_components(melodic_dir=self.inputs.melodic_dir,
                                in_file=self.inputs.in_file,
                                out_file=self.inputs.out_report,
                                compress=self.inputs.compress_report,
                                report_mask=self.inputs.report_mask,
                                noise_components_file=self._noise_components_file
                                )

    def _post_run_hook(self, runtime):
        outputs = self.aggregate_outputs(runtime=runtime)
        self._noise_components_file = os.path.join(outputs.out_dir,
                                                   "classified_motion_ICs.txt")

        NIWORKFLOWS_LOG.info('Generating report for ICA AROMA')
