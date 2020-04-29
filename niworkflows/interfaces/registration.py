# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""ReportCapableInterfaces for registration tools."""
import os
from distutils.version import LooseVersion

import nibabel as nb
import numpy as np
from nilearn import image as nli
from nilearn.image import index_img
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits,
    isdefined,
    TraitedSpec,
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
)
from nipype.interfaces.mixins import reporting
from nipype.interfaces import freesurfer as fs
from nipype.interfaces import fsl, ants, afni

from .. import NIWORKFLOWS_LOG
from . import report_base as nrc
from .mni import (
    _RobustMNINormalizationInputSpec,
    _RobustMNINormalizationOutputSpec,
    RobustMNINormalization,
)
from .fixes import (
    FixHeaderApplyTransforms as ApplyTransforms,
    FixHeaderRegistration as Registration,
)


class _RobustMNINormalizationInputSpecRPT(
    nrc._SVGReportCapableInputSpec, _RobustMNINormalizationInputSpec
):
    pass


class _RobustMNINormalizationOutputSpecRPT(
    reporting.ReportCapableOutputSpec, _RobustMNINormalizationOutputSpec
):
    pass


class RobustMNINormalizationRPT(nrc.RegistrationRC, RobustMNINormalization):
    input_spec = _RobustMNINormalizationInputSpecRPT
    output_spec = _RobustMNINormalizationOutputSpecRPT

    def _post_run_hook(self, runtime):
        # We need to dig into the internal ants.Registration interface
        self._fixed_image = self._get_ants_args()["fixed_image"]
        if isinstance(self._fixed_image, (list, tuple)):
            self._fixed_image = self._fixed_image[0]  # get first item if list

        if self._get_ants_args().get("fixed_image_mask") is not None:
            self._fixed_image_mask = self._get_ants_args().get("fixed_image_mask")
        self._moving_image = self.aggregate_outputs(runtime=runtime).warped_image
        NIWORKFLOWS_LOG.info(
            "Report - setting fixed (%s) and moving (%s) images",
            self._fixed_image,
            self._moving_image,
        )

        return super(RobustMNINormalizationRPT, self)._post_run_hook(runtime)


class _ANTSRegistrationInputSpecRPT(
    nrc._SVGReportCapableInputSpec, ants.registration.RegistrationInputSpec
):
    pass


class _ANTSRegistrationOutputSpecRPT(
    reporting.ReportCapableOutputSpec, ants.registration.RegistrationOutputSpec
):
    pass


class ANTSRegistrationRPT(nrc.RegistrationRC, Registration):
    input_spec = _ANTSRegistrationInputSpecRPT
    output_spec = _ANTSRegistrationOutputSpecRPT

    def _post_run_hook(self, runtime):
        self._fixed_image = self.inputs.fixed_image[0]
        self._moving_image = self.aggregate_outputs(runtime=runtime).warped_image
        NIWORKFLOWS_LOG.info(
            "Report - setting fixed (%s) and moving (%s) images",
            self._fixed_image,
            self._moving_image,
        )

        return super(ANTSRegistrationRPT, self)._post_run_hook(runtime)


class _ANTSApplyTransformsInputSpecRPT(
    nrc._SVGReportCapableInputSpec, ants.resampling.ApplyTransformsInputSpec
):
    pass


class _ANTSApplyTransformsOutputSpecRPT(
    reporting.ReportCapableOutputSpec, ants.resampling.ApplyTransformsOutputSpec
):
    pass


class ANTSApplyTransformsRPT(nrc.RegistrationRC, ApplyTransforms):
    input_spec = _ANTSApplyTransformsInputSpecRPT
    output_spec = _ANTSApplyTransformsOutputSpecRPT

    def _post_run_hook(self, runtime):
        self._fixed_image = self.inputs.reference_image
        self._moving_image = self.aggregate_outputs(runtime=runtime).output_image
        NIWORKFLOWS_LOG.info(
            "Report - setting fixed (%s) and moving (%s) images",
            self._fixed_image,
            self._moving_image,
        )

        return super(ANTSApplyTransformsRPT, self)._post_run_hook(runtime)


class _ApplyTOPUPInputSpecRPT(
    nrc._SVGReportCapableInputSpec, fsl.epi.ApplyTOPUPInputSpec
):
    wm_seg = File(argstr="-wmseg %s", desc="reference white matter segmentation mask")


class _ApplyTOPUPOutputSpecRPT(
    reporting.ReportCapableOutputSpec, fsl.epi.ApplyTOPUPOutputSpec
):
    pass


class ApplyTOPUPRPT(nrc.RegistrationRC, fsl.ApplyTOPUP):
    input_spec = _ApplyTOPUPInputSpecRPT
    output_spec = _ApplyTOPUPOutputSpecRPT

    def _post_run_hook(self, runtime):
        self._fixed_image_label = "after"
        self._moving_image_label = "before"
        self._fixed_image = index_img(
            self.aggregate_outputs(runtime=runtime).out_corrected, 0
        )
        self._moving_image = index_img(self.inputs.in_files[0], 0)
        self._contour = self.inputs.wm_seg if isdefined(self.inputs.wm_seg) else None
        NIWORKFLOWS_LOG.info(
            "Report - setting corrected (%s) and warped (%s) images",
            self._fixed_image,
            self._moving_image,
        )

        return super(ApplyTOPUPRPT, self)._post_run_hook(runtime)


class _FUGUEInputSpecRPT(nrc._SVGReportCapableInputSpec, fsl.preprocess.FUGUEInputSpec):
    wm_seg = File(argstr="-wmseg %s", desc="reference white matter segmentation mask")


class _FUGUEOutputSpecRPT(
    reporting.ReportCapableOutputSpec, fsl.preprocess.FUGUEOutputSpec
):
    pass


class FUGUERPT(nrc.RegistrationRC, fsl.FUGUE):
    input_spec = _FUGUEInputSpecRPT
    output_spec = _FUGUEOutputSpecRPT

    def _post_run_hook(self, runtime):
        self._fixed_image_label = "after"
        self._moving_image_label = "before"
        self._fixed_image = self.aggregate_outputs(runtime=runtime).unwarped_file
        self._moving_image = self.inputs.in_file
        self._contour = self.inputs.wm_seg if isdefined(self.inputs.wm_seg) else None
        NIWORKFLOWS_LOG.info(
            "Report - setting corrected (%s) and warped (%s) images",
            self._fixed_image,
            self._moving_image,
        )

        return super(FUGUERPT, self)._post_run_hook(runtime)


class _FLIRTInputSpecRPT(nrc._SVGReportCapableInputSpec, fsl.preprocess.FLIRTInputSpec):
    pass


class _FLIRTOutputSpecRPT(
    reporting.ReportCapableOutputSpec, fsl.preprocess.FLIRTOutputSpec
):
    pass


class FLIRTRPT(nrc.RegistrationRC, fsl.FLIRT):
    input_spec = _FLIRTInputSpecRPT
    output_spec = _FLIRTOutputSpecRPT

    def _post_run_hook(self, runtime):
        self._fixed_image = self.inputs.reference
        self._moving_image = self.aggregate_outputs(runtime=runtime).out_file
        self._contour = self.inputs.wm_seg if isdefined(self.inputs.wm_seg) else None
        NIWORKFLOWS_LOG.info(
            "Report - setting fixed (%s) and moving (%s) images",
            self._fixed_image,
            self._moving_image,
        )

        return super(FLIRTRPT, self)._post_run_hook(runtime)


class _ApplyXFMInputSpecRPT(
    nrc._SVGReportCapableInputSpec, fsl.preprocess.ApplyXFMInputSpec
):
    pass


class ApplyXFMRPT(FLIRTRPT, fsl.ApplyXFM):
    input_spec = _ApplyXFMInputSpecRPT
    output_spec = _FLIRTOutputSpecRPT


if LooseVersion("0.0.0") < fs.Info.looseversion() < LooseVersion("6.0.0"):
    _BBRegisterInputSpec = fs.preprocess.BBRegisterInputSpec
else:
    _BBRegisterInputSpec = fs.preprocess.BBRegisterInputSpec6


class _BBRegisterInputSpecRPT(nrc._SVGReportCapableInputSpec, _BBRegisterInputSpec):
    # Adds default=True, usedefault=True
    out_lta_file = traits.Either(
        traits.Bool,
        File,
        default=True,
        usedefault=True,
        argstr="--lta %s",
        min_ver="5.2.0",
        desc="write the transformation matrix in LTA format",
    )


class _BBRegisterOutputSpecRPT(
    reporting.ReportCapableOutputSpec, fs.preprocess.BBRegisterOutputSpec
):
    pass


class BBRegisterRPT(nrc.RegistrationRC, fs.BBRegister):
    input_spec = _BBRegisterInputSpecRPT
    output_spec = _BBRegisterOutputSpecRPT

    def _post_run_hook(self, runtime):
        outputs = self.aggregate_outputs(runtime=runtime)
        mri_dir = os.path.join(self.inputs.subjects_dir, self.inputs.subject_id, "mri")
        target_file = os.path.join(mri_dir, "brainmask.mgz")

        # Apply transform for simplicity
        mri_vol2vol = fs.ApplyVolTransform(
            source_file=self.inputs.source_file,
            target_file=target_file,
            lta_file=outputs.out_lta_file,
            interp="nearest",
        )
        res = mri_vol2vol.run()

        self._fixed_image = target_file
        self._moving_image = res.outputs.transformed_file
        self._contour = os.path.join(mri_dir, "ribbon.mgz")
        NIWORKFLOWS_LOG.info(
            "Report - setting fixed (%s) and moving (%s) images",
            self._fixed_image,
            self._moving_image,
        )

        return super(BBRegisterRPT, self)._post_run_hook(runtime)


class _MRICoregInputSpecRPT(
    nrc._SVGReportCapableInputSpec, fs.registration.MRICoregInputSpec
):
    pass


class _MRICoregOutputSpecRPT(
    reporting.ReportCapableOutputSpec, fs.registration.MRICoregOutputSpec
):
    pass


class MRICoregRPT(nrc.RegistrationRC, fs.MRICoreg):
    input_spec = _MRICoregInputSpecRPT
    output_spec = _MRICoregOutputSpecRPT

    def _post_run_hook(self, runtime):
        outputs = self.aggregate_outputs(runtime=runtime)
        mri_dir = None
        if isdefined(self.inputs.subject_id):
            mri_dir = os.path.join(
                self.inputs.subjects_dir, self.inputs.subject_id, "mri"
            )

        if isdefined(self.inputs.reference_file):
            target_file = self.inputs.reference_file
        else:
            target_file = os.path.join(mri_dir, "brainmask.mgz")

        # Apply transform for simplicity
        mri_vol2vol = fs.ApplyVolTransform(
            source_file=self.inputs.source_file,
            target_file=target_file,
            lta_file=outputs.out_lta_file,
            interp="nearest",
        )
        res = mri_vol2vol.run()

        self._fixed_image = target_file
        self._moving_image = res.outputs.transformed_file
        if mri_dir is not None:
            self._contour = os.path.join(mri_dir, "ribbon.mgz")
        NIWORKFLOWS_LOG.info(
            "Report - setting fixed (%s) and moving (%s) images",
            self._fixed_image,
            self._moving_image,
        )

        return super(MRICoregRPT, self)._post_run_hook(runtime)


class _SimpleBeforeAfterInputSpecRPT(nrc._SVGReportCapableInputSpec):
    before = File(exists=True, mandatory=True, desc="file before")
    after = File(exists=True, mandatory=True, desc="file after")
    wm_seg = File(desc="reference white matter segmentation mask")
    before_label = traits.Str("before", usedefault=True)
    after_label = traits.Str("after", usedefault=True)


class SimpleBeforeAfterRPT(nrc.RegistrationRC, nrc.ReportingInterface):
    input_spec = _SimpleBeforeAfterInputSpecRPT

    def _post_run_hook(self, runtime):
        """ there is not inner interface to run """
        self._fixed_image_label = self.inputs.after_label
        self._moving_image_label = self.inputs.before_label
        self._fixed_image = self.inputs.after
        self._moving_image = self.inputs.before
        self._contour = self.inputs.wm_seg if isdefined(self.inputs.wm_seg) else None
        NIWORKFLOWS_LOG.info(
            "Report - setting before (%s) and after (%s) images",
            self._fixed_image,
            self._moving_image,
        )

        return super(SimpleBeforeAfterRPT, self)._post_run_hook(runtime)


class _ResampleBeforeAfterInputSpecRPT(_SimpleBeforeAfterInputSpecRPT):
    base = traits.Enum("before", "after", usedefault=True, mandatory=True)


class ResampleBeforeAfterRPT(SimpleBeforeAfterRPT):
    input_spec = _ResampleBeforeAfterInputSpecRPT

    def _post_run_hook(self, runtime):
        self._fixed_image = self.inputs.after
        self._moving_image = self.inputs.before
        if self.inputs.base == "before":
            resampled_after = nli.resample_to_img(self._fixed_image, self._moving_image)
            fname = fname_presuffix(
                self._fixed_image, suffix="_resampled", newpath=runtime.cwd
            )
            resampled_after.to_filename(fname)
            self._fixed_image = fname
        else:
            resampled_before = nli.resample_to_img(
                self._moving_image, self._fixed_image
            )
            fname = fname_presuffix(
                self._moving_image, suffix="_resampled", newpath=runtime.cwd
            )
            resampled_before.to_filename(fname)
            self._moving_image = fname
        self._contour = self.inputs.wm_seg if isdefined(self.inputs.wm_seg) else None
        NIWORKFLOWS_LOG.info(
            "Report - setting before (%s) and after (%s) images",
            self._fixed_image,
            self._moving_image,
        )

        runtime = super(ResampleBeforeAfterRPT, self)._post_run_hook(runtime)
        NIWORKFLOWS_LOG.info("Successfully created report (%s)", self._out_report)
        os.unlink(fname)

        return runtime


class _EstimateReferenceImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="4D EPI file")
    sbref_file = File(exists=True, desc="Single band reference image")
    mc_method = traits.Enum(
        "AFNI",
        "FSL",
        usedefault=True,
        desc="Which software to use to perform motion correction",
    )


class _EstimateReferenceImageOutputSpec(TraitedSpec):
    ref_image = File(exists=True, desc="3D reference image")
    n_volumes_to_discard = traits.Int(
        desc="Number of detected non-steady "
        "state volumes in the beginning of "
        "the input file"
    )


class EstimateReferenceImage(SimpleInterface):
    """
    Given an 4D EPI file estimate an optimal reference image that could be later
    used for motion estimation and coregistration purposes. If detected uses
    T1 saturated volumes (non-steady state) otherwise a median of
    of a subset of motion corrected volumes is used.
    """

    input_spec = _EstimateReferenceImageInputSpec
    output_spec = _EstimateReferenceImageOutputSpec

    def _run_interface(self, runtime):
        ref_name = self.inputs.in_file
        ref_nii = nb.load(ref_name)
        n_volumes_to_discard = _get_vols_to_discard(ref_nii)

        self._results["n_volumes_to_discard"] = n_volumes_to_discard

        out_ref_fname = os.path.join(runtime.cwd, "ref_bold.nii.gz")
        if isdefined(self.inputs.sbref_file):
            out_ref_fname = os.path.join(runtime.cwd, "ref_sbref.nii.gz")
            ref_name = self.inputs.sbref_file
            ref_nii = nb.squeeze_image(nb.load(ref_name))

            # If reference is only 1 volume, return it directly
            if len(ref_nii.shape) == 3:
                ref_nii.header.extensions.clear()
                ref_nii.to_filename(out_ref_fname)
                self._results["ref_image"] = out_ref_fname
                return runtime
            else:
                # Reset this variable as it no longer applies
                # and value for the output is stored in self._results
                n_volumes_to_discard = 0

        # Slicing may induce inconsistencies with shape-dependent values in extensions.
        # For now, remove all. If this turns out to be a mistake, we can select extensions
        # that don't break pipeline stages.
        ref_nii.header.extensions.clear()

        if n_volumes_to_discard == 0:
            if ref_nii.shape[-1] > 40:
                ref_name = os.path.join(runtime.cwd, "slice.nii.gz")
                nb.Nifti1Image(
                    ref_nii.dataobj[:, :, :, 20:40], ref_nii.affine, ref_nii.header
                ).to_filename(ref_name)

            if self.inputs.mc_method == "AFNI":
                res = afni.Volreg(
                    in_file=ref_name,
                    args="-Fourier -twopass",
                    zpad=4,
                    outputtype="NIFTI_GZ",
                ).run()
            elif self.inputs.mc_method == "FSL":
                res = fsl.MCFLIRT(
                    in_file=ref_name, ref_vol=0, interpolation="sinc"
                ).run()
            mc_slice_nii = nb.load(res.outputs.out_file)

            median_image_data = np.median(mc_slice_nii.get_fdata(), axis=3)
        else:
            median_image_data = np.median(
                ref_nii.dataobj[:, :, :, :n_volumes_to_discard], axis=3
            )

        nb.Nifti1Image(median_image_data, ref_nii.affine, ref_nii.header).to_filename(
            out_ref_fname
        )

        self._results["ref_image"] = out_ref_fname

        return runtime


def _get_vols_to_discard(img):
    from nipype.algorithms.confounds import is_outlier

    data_slice = img.dataobj[:, :, :, :50]
    global_signal = data_slice.mean(axis=0).mean(axis=0).mean(axis=0)
    return is_outlier(global_signal)
