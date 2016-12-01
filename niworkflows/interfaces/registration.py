# -*- coding: utf-8 -*-
"""
ReportCapableInterfaces for registration tools

"""
from __future__ import absolute_import, division, print_function, unicode_literals

from nipype.interfaces import ants, fsl
from niworkflows.common import report as nrc


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


class ApplyXFMInputSpecRPT(nrc.RegistrationRCInputSpec,
                           fsl.preprocess.ApplyXFMInputSpec):
    pass

class ApplyXFMRPT(FLIRTRPT):
    ''' ApplyXFM is a wrapper around FLIRT. ApplyXFMRPT is a wrapper around FLIRTRPT.'''
    input_spec = ApplyXFMInputSpecRPT

class ANTSRegistrationInputSpecRPT(nrc.RegistrationRCInputSpec,
                                   ants.registration.RegistrationInputSpec):
    pass

class ANTSRegistrationOutputSpecRPT(nrc.ReportCapableOutputSpec,
                                    ants.registration.RegistrationOutputSpec):
    pass

class ANTSRegistrationRPT(nrc.RegistrationRC, ants.Registration):
    input_spec = ANTSRegistrationInputSpecRPT
    output_spec = ANTSRegistrationOutputSpecRPT
