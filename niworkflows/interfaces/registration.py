# -*- coding: utf-8 -*-
"""
ReportCapableInterfaces for registration tools

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
from distutils.version import LooseVersion

import nibabel as nb
import numpy as np
from nilearn import image as nli
from nipype.algorithms.confounds import is_outlier
from nipype.utils.filemanip import fname_presuffix

from nipype.interfaces.base import (traits, isdefined, TraitedSpec,
                                    BaseInterfaceInputSpec, File)
from .base import SimpleInterface

from nipype.interfaces import freesurfer as fs
from nipype.interfaces import fsl, ants, afni
from niworkflows.anat import mni
import niworkflows.common.report as nrc
from niworkflows import NIWORKFLOWS_LOG
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


if LooseVersion("0.0.0") < fs.Info.looseversion() < LooseVersion("6.0.0"):
    class BBRegisterInputSpecRPT(nrc.RegistrationRCInputSpec,
                                 fs.preprocess.BBRegisterInputSpec):
        pass
else:
    class BBRegisterInputSpecRPT(nrc.RegistrationRCInputSpec,
                                 fs.preprocess.BBRegisterInputSpec6):
        pass


class BBRegisterOutputSpecRPT(nrc.ReportCapableOutputSpec,
                              fs.preprocess.BBRegisterOutputSpec):
    pass


class BBRegisterRPT(nrc.RegistrationRC, fs.BBRegister):
    input_spec = BBRegisterInputSpecRPT
    output_spec = BBRegisterOutputSpecRPT

    def _post_run_hook(self, runtime):
        mri_dir = os.path.join(self.inputs.subjects_dir,
                               self.inputs.subject_id, 'mri')
        self._fixed_image = os.path.join(mri_dir, 'brainmask.mgz')
        self._moving_image = self.aggregate_outputs().registered_file
        self._contour = os.path.join(mri_dir, 'ribbon.mgz')
        NIWORKFLOWS_LOG.info(
            'Report - setting fixed (%s) and moving (%s) images',
            self._fixed_image, self._moving_image)


class SimpleBeforeAfterInputSpecRPT(nrc.RegistrationRCInputSpec):
    before = File(exists=True, mandatory=True, desc='file before')
    after = File(exists=True, mandatory=True, desc='file after')
    wm_seg = File(desc='reference white matter segmentation mask')


class SimpleBeforeAfterOutputSpecRPT(nrc.ReportCapableOutputSpec):
    pass

class SimpleBeforeAfterRPT(nrc.RegistrationRC):
    input_spec = SimpleBeforeAfterInputSpecRPT
    output_spec = SimpleBeforeAfterOutputSpecRPT

    def _run_interface(self, runtime):
        """ there is not inner interface to run """
        self._out_report = os.path.abspath(self.inputs.out_report)

        self._fixed_image_label = "after"
        self._moving_image_label = "before"
        self._fixed_image = self.inputs.after
        self._moving_image = self.inputs.before
        self._contour = self.inputs.wm_seg if isdefined(self.inputs.wm_seg) else None
        NIWORKFLOWS_LOG.info(
            'Report - setting before (%s) and after (%s) images',
            self._fixed_image, self._moving_image)

        self._generate_report()
        NIWORKFLOWS_LOG.info('Successfully created report (%s)', self._out_report)

        return runtime


class ResampleBeforeAfterInputSpecRPT(SimpleBeforeAfterInputSpecRPT):
    base = traits.Enum('before', 'after', usedefault=True, mandatory=True)


class ResampleBeforeAfterRPT(SimpleBeforeAfterRPT):
    input_spec = ResampleBeforeAfterInputSpecRPT
    def _run_interface(self, runtime):
        """ there is not inner interface to run """
        self._out_report = os.path.abspath(self.inputs.out_report)

        self._fixed_image_label = "after"
        self._moving_image_label = "before"
        self._fixed_image = self.inputs.after
        self._moving_image = self.inputs.before
        if self.inputs.base == 'before':
            resampled_after = nli.resample_to_img(self._fixed_image, self._moving_image)
            fname = fname_presuffix(self._fixed_image, suffix='_resampled', newpath='.')
            resampled_after.to_filename(fname)
            self._fixed_image = os.path.abspath(fname)
        else:
            resampled_before = nli.resample_to_img(self._moving_image, self._fixed_image)
            fname = fname_presuffix(self._moving_image, suffix='_resampled', newpath='.')
            resampled_before.to_filename(fname)
            self._moving_image = os.path.abspath(fname)
        self._contour = self.inputs.wm_seg if isdefined(self.inputs.wm_seg) else None
        NIWORKFLOWS_LOG.info(
            'Report - setting before (%s) and after (%s) images',
            self._fixed_image, self._moving_image)

        self._generate_report()
        NIWORKFLOWS_LOG.info('Successfully created report (%s)', self._out_report)
        os.unlink(fname)

        return runtime


class EstimateReferenceImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="4D EPI file")
    mc_method = traits.Enum("AFNI", "FSL", dsec="Which software to use to perform motion correction",
                            usedefault=True)


class EstimateReferenceImageOutputSpec(TraitedSpec):
    ref_image = File(exists=True, desc="3D reference image")
    n_volumes_to_discard = traits.Int(desc="Number of detected non-steady "
                                           "state volumes in the beginning of "
                                           "the input file")


class EstimateReferenceImage(SimpleInterface):
    """
    Given an 4D EPI file estimate an optimal reference image that could be later
    used for motion estimation and coregistration purposes. If detected uses
    T1 saturated volumes (non-steady state) otherwise a median of
    of a subset of motion corrected volumes is used.
    """
    input_spec = EstimateReferenceImageInputSpec
    output_spec = EstimateReferenceImageOutputSpec

    def _run_interface(self, runtime):
        in_nii = nb.load(self.inputs.in_file)
        data_slice = in_nii.dataobj[:, :, :, :50]
        global_signal = data_slice.mean(axis=0).mean(
            axis=0).mean(axis=0)

        n_volumes_to_discard = is_outlier(global_signal)

        out_ref_fname = os.path.abspath("ref_image.nii.gz")

        if n_volumes_to_discard == 0:
            if in_nii.shape[-1] > 40:
                slice = data_slice[:, :, :, 20:40]
                slice_fname = os.path.abspath("slice.nii.gz")
                nb.Nifti1Image(slice, in_nii.affine,
                               in_nii.header).to_filename(slice_fname)
            else:
                slice_fname = self.inputs.in_file

            if self.inputs.mc_method == "AFNI":
                res = afni.Volreg(in_file=slice_fname, args='-Fourier -twopass',
                                  zpad=4, outputtype='NIFTI_GZ').run()
            elif self.inputs.mc_method == "FSL":
                res = fsl.MCFLIRT(in_file=slice_fname,
                                  ref_vol=0, interpolation='sinc').run()
            mc_slice_nii = nb.load(res.outputs.out_file)

            median_image_data = np.median(mc_slice_nii.get_data(), axis=3)
            nb.Nifti1Image(median_image_data, in_nii.affine,
                           in_nii.header).to_filename(out_ref_fname)
        else:
            median_image_data = np.median(
                data_slice[:, :, :, :n_volumes_to_discard], axis=3)
            nb.Nifti1Image(median_image_data, in_nii.affine,
                           in_nii.header).to_filename(out_ref_fname)

        self._results["ref_image"] = out_ref_fname
        self._results["n_volumes_to_discard"] = n_volumes_to_discard

        return runtime
