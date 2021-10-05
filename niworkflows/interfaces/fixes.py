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
import os

import nibabel as nb

from nipype.interfaces.base import traits, InputMultiObject, File
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.ants.resampling import ApplyTransforms, ApplyTransformsInputSpec
from nipype.interfaces.ants.registration import (
    Registration,
    RegistrationInputSpec as _RegistrationInputSpec,
)
from nipype.interfaces.ants.segmentation import (
    N4BiasFieldCorrection as VanillaN4,
    N4BiasFieldCorrectionOutputSpec as VanillaN4OutputSpec,
)

from .. import __version__
from ..utils.images import _copyxform


class _FixTraitApplyTransformsInputSpec(ApplyTransformsInputSpec):
    transforms = InputMultiObject(
        traits.Either(File(exists=True), 'identity'),
        argstr="%s",
        mandatory=True,
        desc="transform files: will be applied in reverse order. For "
        "example, the last specified transform will be applied first.",
    )


class FixHeaderApplyTransforms(ApplyTransforms):
    """
    A replacement for nipype.interfaces.ants.resampling.ApplyTransforms that
    fixes the resampled image header to match the xform of the reference
    image
    """

    input_spec = _FixTraitApplyTransformsInputSpec

    def _run_interface(self, runtime, correct_return_codes=(0,)):
        # Run normally
        runtime = super(FixHeaderApplyTransforms, self)._run_interface(
            runtime, correct_return_codes
        )

        _copyxform(
            self.inputs.reference_image,
            os.path.abspath(self._gen_filename("output_image")),
            message="%s (niworkflows v%s)" % (self.__class__.__name__, __version__),
        )
        return runtime


class _FixHeaderRegistrationInputSpec(_RegistrationInputSpec):
    restrict_deformation = traits.List(
        traits.List(traits.Range(low=0.0, high=1.0)),
        desc=(
            "This option allows the user to restrict the optimization of "
            "the displacement field, translation, rigid or affine transform "
            "on a per-component basis. For example, if one wants to limit "
            "the deformation or rotation of 3-D volume to the  first two "
            "dimensions, this is possible by specifying a weight vector of "
            "'1x1x0' for a deformation field or '1x1x0x1x1x0' for a rigid "
            "transformation.  Low-dimensional restriction only works if "
            "there are no preceding transformations."
        ),
    )


class FixHeaderRegistration(Registration):
    """
    A replacement for nipype.interfaces.ants.registration.Registration that
    fixes the resampled image header to match the xform of the reference
    image
    """

    input_spec = _FixHeaderRegistrationInputSpec

    def _run_interface(self, runtime, correct_return_codes=(0,)):
        # Run normally
        runtime = super(FixHeaderRegistration, self)._run_interface(
            runtime, correct_return_codes
        )

        # Forward transform
        out_file = self._get_outputfilenames(inverse=False)
        if out_file is not None and out_file:
            _copyxform(
                self.inputs.fixed_image[0],
                os.path.abspath(out_file),
                message="%s (niworkflows v%s)" % (self.__class__.__name__, __version__),
            )

        # Inverse transform
        out_file = self._get_outputfilenames(inverse=True)
        if out_file is not None and out_file:
            _copyxform(
                self.inputs.moving_image[0],
                os.path.abspath(out_file),
                message="%s (niworkflows v%s)" % (self.__class__.__name__, __version__),
            )

        return runtime


class _FixN4BiasFieldCorrectionOutputSpec(VanillaN4OutputSpec):
    negative_values = traits.Bool(
        False,
        usedefault=True,
        desc="Indicates whether the input was corrected for "
        "nonpositive values by adding a constant offset.",
    )


class FixN4BiasFieldCorrection(VanillaN4):
    """Checks and fixes for nonpositive values in the input to ``N4BiasFieldCorrection``."""

    output_spec = _FixN4BiasFieldCorrectionOutputSpec

    def __init__(self, *args, **kwargs):
        """Add a private property to keep the path to the right input."""
        self._input_image = None
        self._negative_values = False
        super(FixN4BiasFieldCorrection, self).__init__(*args, **kwargs)

    def _format_arg(self, name, trait_spec, value):
        if name == "input_image":
            return trait_spec.argstr % self._input_image
        return super(FixN4BiasFieldCorrection, self)._format_arg(
            name, trait_spec, value
        )

    def _parse_inputs(self, skip=None):
        self._input_image = self.inputs.input_image
        # Check intensities
        input_nii = nb.load(self.inputs.input_image)
        datamin = input_nii.get_fdata().min()
        if datamin < 0:
            self._input_image = fname_presuffix(
                self.inputs.input_image, suffix="_scaled", newpath=os.getcwd()
            )
            data = input_nii.get_fdata() - datamin
            newnii = input_nii.__class__(data, input_nii.affine, input_nii.header)
            newnii.to_filename(self._input_image)
            self._negative_values = True

        return super(FixN4BiasFieldCorrection, self)._parse_inputs(skip=skip)

    def _list_outputs(self):
        outputs = super(FixN4BiasFieldCorrection, self)._list_outputs()
        outputs["negative_values"] = self._negative_values
        return outputs
