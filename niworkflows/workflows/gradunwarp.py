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
"""Workflow for the unwarping ."""

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from ..engine.workflows import LiterateWorkflow as Workflow


def init_gradunwarp_wf(
    name="gradunwarp_wf",
):
    from nipype.interfaces.fsl import ConvertWarp
    from ..interfaces.gradunwarp import GradUnwarp

    wf = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["input_file", "coeff_file", "grad_file"]), name="inputnode"
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "warp_file",
                "corrected_file"
            ]
        ),
        name="outputnode",
    )

    gradient_unwarp = pe.MapNode(
        GradUnwarp(),
        name="grad_unwarp",
        iterfield=["infile"]
    )

    convert_warp = pe.MapNode(
        ConvertWarp(
            abswarp=True,
            out_relwarp=True,
        ),
        name='convert_warp_abs2rel',
        iterfield=["reference", "warp1"]
    )

    def _warp_fsl2itk(in_warp):
        import os
        import nitransforms.io
        from nipype.utils.filemanip import fname_presuffix
        fsl_warp = nitransforms.io.fsl.FSLDisplacementsField.from_filename(in_warp)
        itk_warp_data = fsl_warp.get_fdata().reshape(fsl_warp.shape[:3]+(1,3))
        itk_warp_data[...,(0,1)] *= -1
        itk_warp = fsl_warp.__class__(itk_warp_data, fsl_warp.affine)
        itk_warp.header.set_intent("vector")
        out_fname = fname_presuffix(in_warp, suffix="_itk", newpath=os.getcwd())
        itk_warp.to_filename(out_fname)
        return out_fname

    warp_fsl2itk = pe.MapNode(
        niu.Function(
            function=_warp_fsl2itk,
            output_names=["itk_warp"],
            input_names=["in_warp"]
        ),
        iterfield=['in_warp'],
        name="warp_fsl2itk"
    )

    # fmt:off
    wf.connect([
        (inputnode, gradient_unwarp, [
            ("input_file", "infile"),
            ("coeff_file", "coeffile"),
            ("grad_file", "gradfile"),
        ]),
        (inputnode, convert_warp, [("input_file", "reference")]),
        (gradient_unwarp, convert_warp, [("warp_file", "warp1")]),
        (gradient_unwarp, outputnode, [("corrected_file", "corrected_file")]),
        (convert_warp, warp_fsl2itk, [("out_file", "in_warp")]),
        (warp_fsl2itk, outputnode, [("itk_warp", "warp_file")])
    ])
    # fmt:on

    return wf
