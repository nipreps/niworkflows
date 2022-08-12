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

    gradient_unwarp = pe.MapNode(GradUnwarp(), name="grad_unwarp")
    convert_warp = pe.MapNode(
        ConvertWarp(
            abswarp=True,
            out_relwarp=True,
        ),
        name='convert_warp_abs2rel'
        )

    # fmt:off
    wf.connect([
        (inputnode, gradient_unwarp, [
            ("input_file", "infile"),
            ("coeff_file", "coeffile"),
            ("grad_file", "gradfile"),
        ]),
        (inputnode, convert_warp, [
            ("in_file", "reference"),
        ]),
        (gradient_unwarp, convert_warp, [
            ("warp_file", "warp1"),
        ]),
        (gradient_unwarp, output_node, [
            ("corrected_file", "corrected_file")
        ])
    ])
    # fmt:on

    return wf
