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
"""Handling NIfTI headers."""
import os
import shutil
from textwrap import indent
import numpy as np
import nibabel as nb
import transforms3d

from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits,
    File,
    TraitedSpec,
    BaseInterfaceInputSpec,
    SimpleInterface,
    DynamicTraitedSpec,
)
from nipype.interfaces.io import add_traits
from ..utils.images import _copyxform
from .. import __version__


LOGGER = logging.getLogger("nipype.interface")


class _CopyXFormInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    hdr_file = File(exists=True, mandatory=True, desc="the file we get the header from")


class CopyXForm(SimpleInterface):
    """
    Copy the *x-form* orientation headers from ``hdr_file`` to an arbitrary set of images.

    Target images that will get their x-form headers replaced should be prescribed
    using the ``fields`` argument at interface instantiation.

    """

    input_spec = _CopyXFormInputSpec
    output_spec = DynamicTraitedSpec

    def __init__(self, fields=None, **inputs):
        self._fields = fields or ["in_file"]
        if isinstance(self._fields, str):
            self._fields = [self._fields]

        super(CopyXForm, self).__init__(**inputs)

        add_traits(self.inputs, self._fields)
        for f in set(self._fields).intersection(list(inputs.keys())):
            setattr(self.inputs, f, inputs[f])

    def _outputs(self):
        base = super(CopyXForm, self)._outputs()
        if self._fields:
            fields = self._fields.copy()
            if "in_file" in fields:
                idx = fields.index("in_file")
                fields.pop(idx)
                fields.insert(idx, "out_file")

            base = add_traits(base, fields)
        return base

    def _run_interface(self, runtime):
        for f in self._fields:
            in_files = getattr(self.inputs, f)
            self._results[f] = []
            if isinstance(in_files, str):
                in_files = [in_files]
            for in_file in in_files:
                out_name = fname_presuffix(
                    in_file, suffix="_xform", newpath=runtime.cwd
                )
                # Copy and replace header
                shutil.copy(in_file, out_name)
                _copyxform(
                    self.inputs.hdr_file,
                    out_name,
                    message="CopyXForm (niworkflows v%s)" % __version__,
                )
                self._results[f].append(out_name)

            # Flatten out one-element lists
            if len(self._results[f]) == 1:
                self._results[f] = self._results[f][0]

        default = self._results.pop("in_file", None)
        if default:
            self._results["out_file"] = default
        return runtime


class _CopyHeaderInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="the file we get the data from")
    hdr_file = File(exists=True, mandatory=True, desc="the file we get the header from")


class _CopyHeaderOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="written file path")


class CopyHeader(SimpleInterface):
    """
    Copy a header from the `hdr_file` to `out_file` with data drawn from
    `in_file`.
    """

    input_spec = _CopyHeaderInputSpec
    output_spec = _CopyHeaderOutputSpec

    def _run_interface(self, runtime):
        in_img = nb.load(self.inputs.hdr_file)
        out_img = nb.load(self.inputs.in_file)
        new_img = out_img.__class__(out_img.dataobj, in_img.affine, in_img.header)
        new_img.set_data_dtype(out_img.get_data_dtype())

        out_name = fname_presuffix(self.inputs.in_file, suffix="_fixhdr", newpath=".")
        new_img.to_filename(out_name)
        self._results["out_file"] = out_name
        return runtime


class _ValidateImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="input image")


class _ValidateImageOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="validated image")
    out_report = File(exists=True, desc="HTML segment containing warning")


class ValidateImage(SimpleInterface):
    """
    Check the correctness of x-form headers (matrix and code).

    This interface implements the `following logic
    <https://github.com/nipreps/fmriprep/issues/873#issuecomment-349394544>`__:

    .. list-table:: ``ValidateImage`` truth table
       :widths: 15 15 15 15 40
       :header-rows: 1

       * - valid quaternions
         - ``qform_code`` > 0
         - ``sform_code`` > 0
         - ``qform == sform``
         - actions
       * - ``True``
         - ``True``
         - ``True``
         - ``True``
         - None
       * - ``True``
         - ``True``
         - ``False``
         - \\*
         - sform, scode <- qform, qcode
       * - \\*
         - \\*
         - ``True``
         - ``False``
         - qform, qcode <- sform, scode
       * - \\*
         - ``False``
         - ``True``
         - \\*
         - qform, qcode <- sform, scode
       * - \\*
         - ``False``
         - ``False``
         - \\*
         - sform, qform <- best affine; scode, qcode <- 1
       * - ``False``
         - \\*
         - ``False``
         - \\*
         - sform, qform <- best affine; scode, qcode <- 1

    """

    input_spec = _ValidateImageInputSpec
    output_spec = _ValidateImageOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        out_report = os.path.join(runtime.cwd, "report.html")

        # Retrieve xform codes
        sform_code = int(img.header._structarr["sform_code"])
        qform_code = int(img.header._structarr["qform_code"])

        # Check qform is valid
        valid_qform = False
        try:
            qform = img.get_qform()
            valid_qform = True
        except ValueError:
            pass

        sform = img.get_sform()
        if np.linalg.det(sform) == 0:
            valid_sform = False
        else:
            RZS = sform[:3, :3]
            zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
            valid_sform = np.allclose(zooms, img.header.get_zooms()[:3])

        # Matching affines
        matching_affines = valid_qform and np.allclose(qform, sform)

        # Both match, qform valid (implicit with match), codes okay -> do nothing, empty report
        if matching_affines and qform_code > 0 and sform_code > 0:
            self._results["out_file"] = self.inputs.in_file
            open(out_report, "w").close()
            self._results["out_report"] = out_report
            return runtime

        # A new file will be written
        out_fname = fname_presuffix(
            self.inputs.in_file, suffix="_valid", newpath=runtime.cwd
        )
        self._results["out_file"] = out_fname

        # Row 2:
        if valid_qform and qform_code > 0 and (sform_code == 0 or not valid_sform):
            img.set_sform(qform, qform_code)
            warning_txt = "Note on orientation: sform matrix set"
            description = """\
<p class="elem-desc">The sform has been copied from qform.</p>
"""
        # Rows 3-4:
        # Note: if qform is not valid, matching_affines is False
        elif (valid_sform and sform_code > 0) and (
            not matching_affines or qform_code == 0
        ):
            img.set_qform(sform, sform_code)
            new_qform = img.get_qform()
            if valid_qform:
                # False alarm - the difference is due to precision loss of qform
                if np.allclose(new_qform, qform) and qform_code > 0:
                    self._results["out_file"] = self.inputs.in_file
                    open(out_report, "w").close()
                    self._results["out_report"] = out_report
                    return runtime
                # Replacing an existing, valid qform. Report magnitude of change.
                diff = np.linalg.inv(qform) @ new_qform
                trans, rot, _, _ = transforms3d.affines.decompose44(diff)
                angle = transforms3d.axangles.mat2axangle(rot)[1]
                xyz_unit = img.header.get_xyzt_units()[0]
                if xyz_unit == "unknown":
                    xyz_unit = "mm"

                total_trans = np.sqrt(
                    np.sum(trans * trans)
                )  # Add angle and total_trans to report
                warning_txt = "Note on orientation: qform matrix overwritten"
                description = f"""\
    <p class="elem-desc">
    The qform has been copied from sform.
    The difference in angle is {angle:.02g} radians.
    The difference in translation is {total_trans:.02g}{xyz_unit}.
    </p>
    """
            elif qform_code > 0:
                # qform code indicates the qform is supposed to be valid. Use more stridency.
                warning_txt = "WARNING - Invalid qform information"
                description = """\
<p class="elem-desc">
    The qform matrix found in the file header is invalid.
    The qform has been copied from sform.
    Checking the original qform information from the data produced
    by the scanner is advised.
</p>
"""
            else:  # qform_code == 0
                # qform is not expected to be valids. Simple note.
                warning_txt = "Note on orientation: qform matrix overwritten"
                description = (
                    '<p class="elem-desc">The qform has been copied from sform.</p>'
                )
        # Rows 5-6:
        else:
            affine = img.header.get_base_affine()
            img.set_sform(affine, nb.nifti1.xform_codes["scanner"])
            img.set_qform(affine, nb.nifti1.xform_codes["scanner"])
            warning_txt = "WARNING - Missing orientation information"
            description = """\
<p class="elem-desc">
    FMRIPREP could not retrieve orientation information from the image header.
    The qform and sform matrices have been set to a default, LAS-oriented affine.
    Analyses of this dataset MAY BE INVALID.
</p>
"""
        snippet = '<h3 class="elem-title">%s</h3>\n%s\n' % (warning_txt, description)
        # Store new file and report
        img.to_filename(out_fname)
        with open(out_report, "w") as fobj:
            fobj.write(indent(snippet, "\t" * 3))

        self._results["out_report"] = out_report
        return runtime


class _MatchHeaderInputSpec(BaseInterfaceInputSpec):
    reference = File(
        exists=True, mandatory=True, desc="NIfTI file with reference header"
    )
    in_file = File(
        exists=True, mandatory=True, desc="NIfTI file which header will be checked"
    )


class _MatchHeaderOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="NIfTI file with fixed header")


class MatchHeader(SimpleInterface):
    input_spec = _MatchHeaderInputSpec
    output_spec = _MatchHeaderOutputSpec

    def _run_interface(self, runtime):
        refhdr = nb.load(self.inputs.reference).header.copy()
        imgnii = nb.load(self.inputs.in_file)
        imghdr = imgnii.header.copy()

        imghdr["dim_info"] = refhdr["dim_info"]  # dim_info is lost sometimes

        # Set qform
        qform = refhdr.get_qform()
        qcode = int(refhdr["qform_code"])
        if not np.allclose(qform, imghdr.get_qform()):
            LOGGER.warning("q-forms of reference and mask are substantially different")
        imghdr.set_qform(qform, qcode)

        # Set sform
        sform = refhdr.get_sform()
        scode = int(refhdr["sform_code"])
        if not np.allclose(sform, imghdr.get_sform()):
            LOGGER.warning("s-forms of reference and mask are substantially different")
        imghdr.set_sform(sform, scode)

        out_file = fname_presuffix(
            self.inputs.in_file, suffix="_hdr", newpath=runtime.cwd
        )

        imgnii.__class__(imgnii.dataobj, imghdr.get_best_affine(), imghdr).to_filename(
            out_file
        )
        self._results["out_file"] = out_file
        return runtime


class _SanitizeImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="input image")
    n_volumes_to_discard = traits.Int(
        0, usedefault=True, desc="discard n first volumes"
    )
    max_32bit = traits.Bool(
        False,
        usedefault=True,
        desc="cast data to float32 if higher " "precision is encountered",
    )


class _SanitizeImageOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="validated image")
    out_report = File(exists=True, desc="HTML segment containing warning")


class SanitizeImage(SimpleInterface):
    """
    Check the correctness of x-form headers (matrix and code) and fixes
    problematic combinations of values. Removes any extension form the header
    if present.
    This interface implements the `following logic
    <https://github.com/nipreps/fmriprep/issues/873#issuecomment-349394544>`_:


    .. list-table:: ``SanitizeImage`` truth table
       :widths: 15 15 15 15 40
       :header-rows: 1

       * - valid quaternions
         - ``qform_code`` > 0
         - ``sform_code`` > 0
         - ``qform == sform``
         - actions
       * - ``True``
         - ``True``
         - ``True``
         - ``True``
         - None
       * - ``True``
         - ``True``
         - ``False``
         - \\*
         - sform, scode <- qform, qcode
       * - \\*
         - ``True``
         - \\*
         - ``False``
         - sform, scode <- qform, qcode
       * - \\*
         - ``False``
         - ``True``
         - \\*
         - qform, qcode <- sform, scode
       * - \\*
         - ``False``
         - ``False``
         - \\*
         - sform, qform <- best affine; scode, qcode <- 1
       * - ``False``
         - \\*
         - ``False``
         - \\*
         - sform, qform <- best affine; scode, qcode <- 1

    """

    input_spec = _SanitizeImageInputSpec
    output_spec = _SanitizeImageOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        out_report = os.path.join(runtime.cwd, "report.html")

        # Retrieve xform codes
        sform_code = int(img.header._structarr["sform_code"])
        qform_code = int(img.header._structarr["qform_code"])

        # Check qform is valid
        valid_qform = False
        try:
            img.get_qform()
            valid_qform = True
        except ValueError:
            pass

        # Matching affines
        matching_affines = valid_qform and np.allclose(img.get_qform(), img.get_sform())

        save_file = False
        warning_txt = ""

        # Both match, qform valid (implicit with match), codes okay -> do nothing, empty report
        if matching_affines and qform_code > 0 and sform_code > 0:
            self._results["out_file"] = self.inputs.in_file
            open(out_report, "w").close()

        # Row 2:
        elif valid_qform and qform_code > 0:
            img.set_sform(img.get_qform(), qform_code)
            save_file = True
            warning_txt = "Note on orientation: sform matrix set"
            description = """\
<p class="elem-desc">The sform has been copied from qform.</p>
"""
        # Rows 3-4:
        # Note: if qform is not valid, matching_affines is False
        elif sform_code > 0 and (not matching_affines or qform_code == 0):
            img.set_qform(img.get_sform(), sform_code)
            save_file = True
            warning_txt = "Note on orientation: qform matrix overwritten"
            description = """\
<p class="elem-desc">The qform has been copied from sform.</p>
"""
            if not valid_qform and qform_code > 0:
                warning_txt = "WARNING - Invalid qform information"
                description = """\
<p class="elem-desc">
    The qform matrix found in the file header is invalid.
    The qform has been copied from sform.
    Checking the original qform information from the data produced
    by the scanner is advised.
</p>
"""
        # Rows 5-6:
        else:
            affine = img.affine
            img.set_sform(affine, nb.nifti1.xform_codes["scanner"])
            img.set_qform(affine, nb.nifti1.xform_codes["scanner"])
            save_file = True
            warning_txt = "WARNING - Missing orientation information"
            description = """\
<p class="elem-desc">
    Orientation information could not be retrieved from the image header.
    The qform and sform matrices have been set to a default, LAS-oriented affine.
    Analyses of this dataset MAY BE INVALID.
</p>
"""

        if (
            self.inputs.max_32bit and np.dtype(img.get_data_dtype()).itemsize > 4
        ) or self.inputs.n_volumes_to_discard:
            # force float32 only if 64 bit dtype is detected
            if self.inputs.max_32bit and np.dtype(img.get_data_dtype()).itemsize > 4:
                in_data = img.get_fdata(dtype=np.float32)
            else:
                in_data = img.dataobj

            img = nb.Nifti1Image(
                in_data[:, :, :, self.inputs.n_volumes_to_discard:],
                img.affine,
                img.header,
            )
            save_file = True

        if len(img.header.extensions) != 0:
            img.header.extensions.clear()
            save_file = True

        # Store new file
        if save_file:
            out_fname = fname_presuffix(
                self.inputs.in_file, suffix="_valid", newpath=runtime.cwd
            )
            self._results["out_file"] = out_fname
            img.to_filename(out_fname)

        if warning_txt:
            snippet = '<h3 class="elem-title">%s</h3>\n%s\n' % (
                warning_txt,
                description,
            )
            with open(out_report, "w") as fobj:
                fobj.write(indent(snippet, "\t" * 3))

        self._results["out_report"] = out_report
        return runtime
