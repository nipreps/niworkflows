# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""ReportCapableInterfaces for registration tools."""
import os
from distutils.version import LooseVersion

from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits,
    isdefined,
    File,
)
from nipype.interfaces.mixins import reporting
from nipype.interfaces import freesurfer as fs
from nipype.interfaces import fsl, ants

from ... import NIWORKFLOWS_LOG
from . import base as nrb
from ..norm import (
    _SpatialNormalizationInputSpec,
    _SpatialNormalizationOutputSpec,
    SpatialNormalization,
)
from ..fixes import (
    FixHeaderApplyTransforms as ApplyTransforms,
    FixHeaderRegistration as Registration,
)


class _SpatialNormalizationInputSpecRPT(
    nrb._SVGReportCapableInputSpec, _SpatialNormalizationInputSpec
):
    pass


class _SpatialNormalizationOutputSpecRPT(
    reporting.ReportCapableOutputSpec, _SpatialNormalizationOutputSpec
):
    pass


class SpatialNormalizationRPT(nrb.RegistrationRC, SpatialNormalization):
    input_spec = _SpatialNormalizationInputSpecRPT
    output_spec = _SpatialNormalizationOutputSpecRPT

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

        return super(SpatialNormalizationRPT, self)._post_run_hook(runtime)


class _ANTSRegistrationInputSpecRPT(
    nrb._SVGReportCapableInputSpec, ants.registration.RegistrationInputSpec
):
    pass


class _ANTSRegistrationOutputSpecRPT(
    reporting.ReportCapableOutputSpec, ants.registration.RegistrationOutputSpec
):
    pass


class ANTSRegistrationRPT(nrb.RegistrationRC, Registration):
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
    nrb._SVGReportCapableInputSpec, ants.resampling.ApplyTransformsInputSpec
):
    pass


class _ANTSApplyTransformsOutputSpecRPT(
    reporting.ReportCapableOutputSpec, ants.resampling.ApplyTransformsOutputSpec
):
    pass


class ANTSApplyTransformsRPT(nrb.RegistrationRC, ApplyTransforms):
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
    nrb._SVGReportCapableInputSpec, fsl.epi.ApplyTOPUPInputSpec
):
    wm_seg = File(argstr="-wmseg %s", desc="reference white matter segmentation mask")


class _ApplyTOPUPOutputSpecRPT(
    reporting.ReportCapableOutputSpec, fsl.epi.ApplyTOPUPOutputSpec
):
    pass


class ApplyTOPUPRPT(nrb.RegistrationRC, fsl.ApplyTOPUP):
    input_spec = _ApplyTOPUPInputSpecRPT
    output_spec = _ApplyTOPUPOutputSpecRPT

    def _post_run_hook(self, runtime):
        from nilearn.image import index_img

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


class _FUGUEInputSpecRPT(nrb._SVGReportCapableInputSpec, fsl.preprocess.FUGUEInputSpec):
    wm_seg = File(argstr="-wmseg %s", desc="reference white matter segmentation mask")


class _FUGUEOutputSpecRPT(
    reporting.ReportCapableOutputSpec, fsl.preprocess.FUGUEOutputSpec
):
    pass


class FUGUERPT(nrb.RegistrationRC, fsl.FUGUE):
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


class _FLIRTInputSpecRPT(nrb._SVGReportCapableInputSpec, fsl.preprocess.FLIRTInputSpec):
    pass


class _FLIRTOutputSpecRPT(
    reporting.ReportCapableOutputSpec, fsl.preprocess.FLIRTOutputSpec
):
    pass


class FLIRTRPT(nrb.RegistrationRC, fsl.FLIRT):
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
    nrb._SVGReportCapableInputSpec, fsl.preprocess.ApplyXFMInputSpec
):
    pass


class ApplyXFMRPT(FLIRTRPT, fsl.ApplyXFM):
    input_spec = _ApplyXFMInputSpecRPT
    output_spec = _FLIRTOutputSpecRPT


if LooseVersion("0.0.0") < fs.Info.looseversion() < LooseVersion("6.0.0"):
    _BBRegisterInputSpec = fs.preprocess.BBRegisterInputSpec
else:
    _BBRegisterInputSpec = fs.preprocess.BBRegisterInputSpec6


class _BBRegisterInputSpecRPT(nrb._SVGReportCapableInputSpec, _BBRegisterInputSpec):
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


class BBRegisterRPT(nrb.RegistrationRC, fs.BBRegister):
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
    nrb._SVGReportCapableInputSpec, fs.registration.MRICoregInputSpec
):
    pass


class _MRICoregOutputSpecRPT(
    reporting.ReportCapableOutputSpec, fs.registration.MRICoregOutputSpec
):
    pass


class MRICoregRPT(nrb.RegistrationRC, fs.MRICoreg):
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


class _SimpleBeforeAfterInputSpecRPT(nrb._SVGReportCapableInputSpec):
    before = File(exists=True, mandatory=True, desc="file before")
    after = File(exists=True, mandatory=True, desc="file after")
    wm_seg = File(desc="reference white matter segmentation mask")
    before_label = traits.Str("before", usedefault=True)
    after_label = traits.Str("after", usedefault=True)
    dismiss_affine = traits.Bool(
        False, usedefault=True, desc="rotate image(s) to cardinal axes"
    )


class SimpleBeforeAfterRPT(nrb.RegistrationRC, nrb.ReportingInterface):
    input_spec = _SimpleBeforeAfterInputSpecRPT

    def _post_run_hook(self, runtime):
        """ there is not inner interface to run """
        self._fixed_image_label = self.inputs.after_label
        self._moving_image_label = self.inputs.before_label
        self._fixed_image = self.inputs.after
        self._moving_image = self.inputs.before
        self._contour = self.inputs.wm_seg if isdefined(self.inputs.wm_seg) else None
        self._dismiss_affine = self.inputs.dismiss_affine
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
        from nilearn import image as nli

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
