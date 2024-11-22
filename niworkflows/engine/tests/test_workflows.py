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
"""Test the LiterateWorkflow."""

from nipype.interfaces import afni
from nipype.interfaces import utility as niu
from nipype.pipeline.engine import Node

from ..workflows import LiterateWorkflow as Workflow


def _reorient_wf(name='ReorientWorkflow'):
    """A workflow to reorient images to 'RPI' orientation."""
    workflow = Workflow(name=name)
    workflow.__desc__ = 'Inner workflow. '
    inputnode = Node(niu.IdentityInterface(fields=['in_file']), name='inputnode')
    outputnode = Node(niu.IdentityInterface(fields=['out_file']), name='outputnode')
    deoblique = Node(afni.Refit(deoblique=True), name='deoblique')
    reorient = Node(afni.Resample(orientation='RPI', outputtype='NIFTI_GZ'), name='reorient')
    workflow.connect(
        [
            (inputnode, deoblique, [('in_file', 'in_file')]),
            (deoblique, reorient, [('out_file', 'in_file')]),
            (reorient, outputnode, [('out_file', 'out_file')]),
        ]
    )
    return workflow


def test_boilerplate():
    """Check the boilerplate is generated."""
    workflow = Workflow(name='test')
    workflow.__desc__ = 'Outer workflow. '
    workflow.__postdesc__ = 'Outer workflow (postdesc).'

    inputnode = Node(niu.IdentityInterface(fields=['in_file']), name='inputnode')
    inner = _reorient_wf()

    # fmt: off
    workflow.connect([
        (inputnode, inner, [('in_file', 'inputnode.in_file')]),
    ])
    # fmt: on

    assert workflow.visit_desc() == 'Outer workflow. Inner workflow. Outer workflow (postdesc).'
