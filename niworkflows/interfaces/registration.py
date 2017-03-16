# -*- coding: utf-8 -*-
"""
ReportCapableInterfaces for registration tools

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os

from nipype.interfaces import fsl, ants, freesurfer
from niworkflows.anat import mni
import niworkflows.common.report as nrc
from niworkflows import NIWORKFLOWS_LOG
from nipype.interfaces.base import isdefined, File
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
    wm_seg = File(argstr='-wmseg %s',
                  desc='reference white matter segmentation mask')


class ApplyTOPUPOutputSpecRPT(nrc.ReportCapableOutputSpec,
                              fsl.epi.ApplyTOPUPOutputSpec):
    pass


class ApplyTOPUPRPT(nrc.RegistrationRC, fsl.ApplyTOPUP):
    input_spec = ApplyTOPUPInputSpecRPT
    output_spec = ApplyTOPUPOutputSpecRPT

    def _post_run_hook(self, runtime):
        self._fixed_image_label = "after"
        self._moving_image_label = "before"
        self._fixed_image = index_img(self.aggregate_outputs().out_corrected, 0)
        self._moving_image = index_img(self.inputs.in_files[0], 0)
        self._contour = self.inputs.wm_seg if isdefined(self.inputs.wm_seg) else None
        NIWORKFLOWS_LOG.info('Report - setting corrected (%s) and warped (%s) images',
                             self._fixed_image, self._moving_image)


class FUGUEInputSpecRPT(nrc.RegistrationRCInputSpec,
                        fsl.preprocess.FUGUEInputSpec):
    wm_seg = File(argstr='-wmseg %s',
                  desc='reference white matter segmentation mask')


class FUGUEOutputSpecRPT(nrc.ReportCapableOutputSpec,
                         fsl.preprocess.FUGUEOutputSpec):
    pass

class FUGUERPT(nrc.RegistrationRC, fsl.FUGUE):
    input_spec = FUGUEInputSpecRPT
    output_spec = FUGUEOutputSpecRPT

    def _post_run_hook(self, runtime):
        self._fixed_image_label = "after"
        self._moving_image_label = "before"
        self._fixed_image = self.aggregate_outputs().unwarped_file
        self._moving_image = self.inputs.in_file
        self._contour = self.inputs.wm_seg if isdefined(self.inputs.wm_seg) else None
        NIWORKFLOWS_LOG.info(
            'Report - setting corrected (%s) and warped (%s) images',
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


class ApplyXFMInputSpecRPT(nrc.RegistrationRCInputSpec,
                           fsl.preprocess.ApplyXFMInputSpec):
    pass

class ApplyXFMRPT(FLIRTRPT, fsl.ApplyXFM):
    input_spec = ApplyXFMInputSpecRPT
    output_spec = FLIRTOutputSpecRPT


class BBRegisterInputSpecRPT(nrc.RegistrationRCInputSpec,
                             freesurfer.preprocess.BBRegisterInputSpec):
    if freesurfer.preprocess.FSVersion >= LooseVersion('6.0.0'):
        init = traits.Enum(
            'coreg', 'rr', 'spm', 'fsl', 'header', 'best',
            argstr='--init-%s', usedefault=True, xor=['init_reg_file'],
            desc='initialize registration with mri_coreg, spm, fsl, or header')
    else:
        init = traits.Enum(
            'fsl', 'spm', 'header',
            argstr='--init-%s', usedefault=True, xor=['init_reg_file'],
            desc='initialize registration with fsl, spm, or header')

class BBRegisterOutputSpecRPT(nrc.ReportCapableOutputSpec,
                              freesurfer.preprocess.BBRegisterOutputSpec):
    pass

class BBRegisterRPT(nrc.RegistrationRC, freesurfer.BBRegister):
    input_spec = BBRegisterInputSpecRPT
    output_spec = BBRegisterOutputSpecRPT

    def _post_run_hook(self, runtime):
        mri_dir = os.path.join(self.inputs.subjects_dir, self.inputs.subject_id,
                               'mri')
        self._fixed_image = os.path.join(mri_dir, 'brainmask.mgz')
        self._moving_image = self.aggregate_outputs().registered_file
        self._contour = os.path.join(mri_dir, 'ribbon.mgz')
        NIWORKFLOWS_LOG.info(
            'Report - setting fixed (%s) and moving (%s) images',
            self._fixed_image, self._moving_image)
