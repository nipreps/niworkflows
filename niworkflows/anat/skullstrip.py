# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2016-07-21 11:28:52
# @Last Modified by:   oesteban
# @Last Modified time: 2016-07-21 11:41:15
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl
from nipype.interfaces import afni

def afni_wf(name='AFNISkullStripWorkflow'):
    """
    Skull-stripping workflow

    Derived from the codebase of the QAP:
    https://github.com/preprocessed-connectomes-project/\
quality-assessment-protocol/blob/master/qap/anatomical_preproc.py#L105


    """

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file', 'out_mask']),
                         name='outputnode')

    sstrip = pe.Node(afni.SkullStrip(outputtype='NIFTI_GZ'), name='skullstrip')
    sstrip_orig_vol = pe.Node(afni.Calc(
        expr='a*step(b)', outputtype='NIFTI_GZ'), name='sstrip_orig_vol')
    binarize = pe.Node(fsl.Threshold(args='-bin', thresh=1.e-3), name='binarize')

    workflow.connect([
        (inputnode, sstrip, [('in_file', 'in_file')]),
        (inputnode, sstrip_orig_vol, [('in_file', 'in_file_a')]),
        (sstrip, sstrip_orig_vol, [('out_file', 'in_file_b')]),
        (sstrip_orig_vol, binarize, [('out_file', 'in_file')]),
        (sstrip_orig_vol, outputnode, [('out_file', 'out_file')]),
        (binarize, outputnode, [('out_file', 'out_mask')])
    ])
    return workflow
