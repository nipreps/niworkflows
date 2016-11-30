# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2016-07-21 14:11:03
# @Last Modified by:   oesteban
# @Last Modified time: 2016-11-22 16:41:07
from __future__ import absolute_import, division, print_function

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import afni

def reorient_wf(name='ReorientWorkflow'):
    """A workflow to reorient images to 'RPI' orientation"""
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_file']), name='outputnode')

    deoblique = pe.Node(afni.Refit(deoblique=True), name='deoblique')
    reorient = pe.Node(afni.Resample(
        orientation='RPI', outputtype='NIFTI_GZ'), name='reorient')
    workflow.connect([
        (inputnode, deoblique, [('in_file', 'in_file')]),
        (deoblique, reorient, [('out_file', 'in_file')]),
        (reorient, outputnode, [('out_file', 'out_file')])
    ])
    return workflow
