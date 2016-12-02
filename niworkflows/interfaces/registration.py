# -*- coding: utf-8 -*-
"""
ReportCapableInterfaces for registration tools

"""
from __future__ import absolute_import, division, print_function, unicode_literals

from nipype.interfaces import fsl, ants
from niworkflows.anat import mni
import niworkflows.common.report as nrc
from niworkflows import NIWORKFLOWS_LOG

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
        self._fixed_image = self.inputs.fixed_image
        self._moving_image = self.aggregate_outputs().warped_image
        NIWORKFLOWS_LOG.info('Report - setting fixed (%s) and moving (%s) images',
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
        NIWORKFLOWS_LOG.info('Report - setting fixed (%s) and moving (%s) images',
                             self._fixed_image, self._moving_image)



# COMMENTED OUT UNTIL WE REALLY IMPLEMENT IT
# class ApplyXFMInputSpecRPT(nrc.RegistrationRCInputSpec,
#                            fsl.preprocess.ApplyXFMInputSpec):
#     pass
#
# class ApplyXFMRPT(FLIRTRPT):
#     ''' ApplyXFM is a wrapper around FLIRT. ApplyXFMRPT is a wrapper around FLIRTRPT.'''
#     input_spec = ApplyXFMInputSpecRPT
