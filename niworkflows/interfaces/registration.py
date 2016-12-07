# -*- coding: utf-8 -*-
"""
ReportCapableInterfaces for registration tools

"""
from __future__ import absolute_import, division, print_function, unicode_literals

from nipype.interfaces import fsl, ants
from niworkflows.anat import mni
import niworkflows.common.report as nrc
from niworkflows import NIWORKFLOWS_LOG
from nipype.interfaces.base import isdefined
from nilearn.image import index_img

class RobustMNINormalizationInputSpecRPT(
    nrc.RegistrationRCInputSpec, mni.RobustMNINormalizationInputSpec):
    pass

class RobustMNINormalizationOutputSpecRPT(
    nrc.ReportCapableOutputSpec, mni.RegistrationOutputSpec):
    pass

class RobustMNINormalizationRPT(
    nrc.RegistrationRC, mni.RobustMNINormalization):
    input_spec = RobustMNINormalizationInputSpecRPT
    output_spec = RobustMNINormalizationOutputSpecRPT

    def _post_run_hook(self, runtime):
        # We need to dig into the internal ants.Registration interface
        self._fixed_image = self.norm.inputs.fixed_image[0]  # and get first item
        if isdefined(self.norm.inputs.fixed_image_mask):
            self._fixed_image_mask = self.norm.inputs.fixed_image_mask
        self._moving_image = self.aggregate_outputs().warped_image
        NIWORKFLOWS_LOG.info('Report - setting fixed (%s) and moving (%s) images',
                             self._fixed_image, self._moving_image)


class ANTSRegistrationInputSpecRPT(nrc.RegistrationRCInputSpec,
                                   ants.registration.RegistrationInputSpec):
    pass

class ANTSRegistrationOutputSpecRPT(nrc.ReportCapableOutputSpec,
                                    ants.registration.RegistrationOutputSpec):
    pass

class ANTSRegistrationRPT(nrc.RegistrationRC, ants.Registration):
    input_spec = ANTSRegistrationInputSpecRPT
    output_spec = ANTSRegistrationOutputSpecRPT

    def _post_run_hook(self, runtime):
        self._fixed_image = self.inputs.fixed_image[0]
        self._moving_image = self.aggregate_outputs().warped_image
        NIWORKFLOWS_LOG.info('Report - setting fixed (%s) and moving (%s) images',
                             self._fixed_image, self._moving_image)


class ANTSApplyTransformsInputSpecRPT(nrc.RegistrationRCInputSpec,
                                      ants.resampling.ApplyTransformsInputSpec):
    pass

class ANTSApplyTransformsOutputSpecRPT(nrc.ReportCapableOutputSpec,
                                       ants.resampling.ApplyTransformsOutputSpec):
    pass

class ANTSApplyTransformsRPT(nrc.RegistrationRC, ants.ApplyTransforms):
    input_spec = ANTSApplyTransformsInputSpecRPT
    output_spec = ANTSApplyTransformsOutputSpecRPT

    def _post_run_hook(self, runtime):
        self._fixed_image = self.inputs.reference_image
        self._moving_image = self.aggregate_outputs().output_image
        NIWORKFLOWS_LOG.info('Report - setting fixed (%s) and moving (%s) images',
                             self._fixed_image, self._moving_image)


class ApplyTOPUPInputSpecRPT(nrc.RegistrationRCInputSpec,
                             fsl.epi.ApplyTOPUPInputSpec):
    pass


class ApplyTOPUPOutputSpecRPT(nrc.ReportCapableOutputSpec,
                              fsl.epi.ApplyTOPUPOutputSpec):
    pass


class ApplyTOPUPRPT(nrc.RegistrationRC, fsl.ApplyTOPUP):
    input_spec = ApplyTOPUPInputSpecRPT
    output_spec = ApplyTOPUPOutputSpecRPT

    def _post_run_hook(self, runtime):
        self._fixed_image_label = "before"
        self._moving_image_label = "after"
        self._fixed_image = index_img(self.inputs.in_files[0], 0)
        self._moving_image = index_img(self.aggregate_outputs().out_corrected, 0)
        NIWORKFLOWS_LOG.info('Report - setting before (%s) and after (%s) images',
                             self._fixed_image, self._moving_image)


class FLIRTInputSpecRPT(nrc.RegistrationRCInputSpec,
                        fsl.preprocess.FLIRTInputSpec):
    pass

class FLIRTOutputSpecRPT(nrc.ReportCapableOutputSpec,
                         fsl.preprocess.FLIRTOutputSpec):
    pass

class FLIRTRPT(nrc.RegistrationRC, fsl.FLIRT):
    input_spec = FLIRTInputSpecRPT
    output_spec = FLIRTOutputSpecRPT

    def _post_run_hook(self, runtime):
        self._fixed_image = self.inputs.reference
        self._moving_image = self.aggregate_outputs().out_file
        self._contour = self.inputs.wm_seg if isdefined(self.inputs.wm_seg) else None
        NIWORKFLOWS_LOG.info(
            'Report - setting fixed (%s) and moving (%s) images',
            self._fixed_image, self._moving_image)


# COMMENTED OUT UNTIL WE REALLY IMPLEMENT IT
# class ApplyXFMInputSpecRPT(nrc.RegistrationRCInputSpec,
#                            fsl.preprocess.ApplyXFMInputSpec):
#     pass
#
# class ApplyXFMRPT(FLIRTRPT):
#     ''' ApplyXFM is a wrapper around FLIRT. ApplyXFMRPT is a wrapper around FLIRTRPT.'''
#     input_spec = ApplyXFMInputSpecRPT
