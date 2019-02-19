#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
Workflows for applying image processing tools to TSV files
"""

from nipype.interfaces import utility as niu, fsl
import nipype.pipeline.engine as pe
from ..interfaces.timeseries1D import (
    TSVToImage, ImageToTSV, TSVFrom1D, TSVTo1D
)


def init_apply_to_2d_wf(function_node,
                        t_rep,
                        name='apply_to_2d_wf',
                        mode=None,
                        input_field='in_file',
                        output_field='out_file'):
    """
    This workflow applies any time series processing node designed for NIfTI
    time series processing to a TSV file. This currently only works with one-
    to-one processing functions: those that generate a single main output from
    a single main input.


    .. workflow::
        :graph2use: orig
        :simple_form: yes
        from niworkflows.func import init_temporal_filter_wf
        wf = init_temporal_filter_wf()


    **Parameters**

        function_node
            The nipype image processing node that should be applied to the
            input one- or two-dimensional TSV file. This should be a MapNode
            with an appropriate ``iterfield`` set.
        t_rep
            Repetition time or sampling interval for the data in the TSV file.
        name
            Name of workflow (default: ``apply_to_2d_wf``)
        mode
            Either ``afni3d`` or None (default). Set to ``afni3d`` if the
            transformation to be applied uses an interface to an AFNI ``3d~``
            function. (This changes the architecture of the workflow.)
        input_field
            Input field of ``function_node`` (default ``in_file``)
        output_field
            Output field of ``function_node`` (default ``out_file``)


    **Inputs**

        in_files
            A list of one- or two-dimensional TSV files to which the
            processing step in `function_node` should be applied.


    **Outputs**

        out_files
            TSV files with the requested transformation applied.
    """
    workflow = pe.Workflow(name=name)

    # I/O nodes
    inputnode = pe.Node(
        name='inputnode',
        interface=niu.IdentityInterface(fields=['in_files']))
    outputnode = pe.Node(
        name='outputnode',
        interface=niu.IdentityInterface(fields=['out_files']))

    # Assemble the workflow.
    if mode == 'afni3d':
        to_1d = pe.MapNode(name='to_1d', iterfield=['in_file'],
                           interface=TSVTo1D(transpose=True))
        from_1d = pe.MapNode(name='from_1d', iterfield=['in_file', 'header'],
                             interface=TSVFrom1D(in_format='columns'))
        # This is extremely hackish, but it seems to be the only way to get
        # AFNI 3D operations to work on 1D files in Nipype as of now
        suffix_1d_1 = pe.MapNode(
            name=('suffix_1D_first'), iterfield=['src'],
            interface=niu.Function(
                input_names=['src'],
                output_names=['real_dst'],
                function=_copy_to_1D))
        suffix_1d_2 = pe.MapNode(
            name=('suffix_1D_second'), iterfield=['src'],
            interface=niu.Function(
                input_names=['src'],
                output_names=['real_dst'],
                function=_copy_to_1D))

        workflow.connect([
            (inputnode, to_1d, [('in_files', 'in_file')]),
            (to_1d, suffix_1d_1, [('out_file', 'src')]),
            (suffix_1d_1, function_node, [('real_dst', input_field)]),
            (function_node, suffix_1d_2, [(output_field, 'src')]),
            (suffix_1d_2, from_1d, [('real_dst', 'in_file')]),
            (to_1d, from_1d, [('header', 'header')]),
            (from_1d, outputnode, [('out_file', 'out_files')])
        ])
    else:
        # Map tsv to image and image to tsv
        tsv_to_img = pe.MapNode(
            name='tsv2img', iterfield=['in_file'],
            interface=TSVToImage(t_rep=t_rep))
        img_to_tsv = pe.MapNode(
            interface=ImageToTSV(),
            name='img2tsv', iterfield=['in_file', 'header'])
        # Copy header information
        copy_geometry = pe.MapNode(
            name='copy_geometry', iterfield=['in_file', 'dest_file'],
            interface=fsl.utils.CopyGeom(output_type='NIFTI_GZ'))

        workflow.connect([
            (inputnode, tsv_to_img, [('in_files', 'in_file')]),
            (tsv_to_img, function_node, [('out_file', input_field)]),
            (tsv_to_img, copy_geometry, [('out_file', 'in_file')]),
            (function_node, copy_geometry, [(output_field, 'dest_file')]),
            (copy_geometry, img_to_tsv, [('out_file', 'in_file')]),
            (tsv_to_img, img_to_tsv, [('header', 'header')]),
            (img_to_tsv, outputnode, [('out_file', 'out_files')])
        ])

    return workflow


def _copy_to_1D(src):
    """A hackish wrapper around shutil functions to trick Nipype."""
    from shutil import copyfile
    from nipype.utils.filemanip import split_filename
    path, name, _ = split_filename(src)
    dst = '{}/{}.1D'.format(path, name)
    real_dst = copyfile(src, dst)
    return real_dst
