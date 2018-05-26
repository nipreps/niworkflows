#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Nipype translation of ANTs workflows
------------------------------------

"""
from __future__ import print_function, division, absolute_import, unicode_literals

import os
from multiprocessing import cpu_count
from ..nipype.pipeline import engine as pe
from ..nipype.interfaces import ants, utility as niu
from ..interfaces.ants import ImageMath, ResampleImageBySpacing


def brain_extraction(name='antsBrainExtraction',
                     in_template='OASIS',
                     float=True,
                     debug=False,
                     random_seeding=True,
                     omp_nthreads=None,
                     in_segmentation_model='T1'):
    """
    The official antsBrainExtraction.sh workflow converted into Nipype,
    only for 3D images.

    Inputs
    ------

    `in_file`
        The input anatomical image to be segmented, typically T1-weighted.
        If a list of anatomical images is provided, subsequently specified
        images are used during the segmentation process.
        However, only the first image is used in the registration of priors.
        Our suggestion would be to specify the T1w as the first image.


    `in_template`
        The brain template from which regions will be projected
        Anatomical template created using e.g. LPBA40 data set with
        buildtemplateparallel.sh in ANTs.

    `in_mask`
        Brain probability mask created using e.g. LPBA40 data set which
        have brain masks defined, and warped to anatomical template and
        averaged resulting in a probability image.

    Optional Inputs
    ---------------

    `in_segmentation_model`
        A k-means segmentation is run to find gray or white matter around
        the edge of the initial brain mask warped from the template.
        This produces a segmentation image with K classes, ordered by mean
        intensity in increasing order. With this option, you can control
        K and tell the script which classes represent CSF, gray and white matter.
        Format (K, csfLabel, gmLabel, wmLabel)
        Examples:
        -c 3,1,2,3 for T1 with K=3, CSF=1, GM=2, WM=3 (default)
        -c 3,3,2,1 for T2 with K=3, CSF=3, GM=2, WM=1
        -c 3,1,3,2 for FLAIR with K=3, CSF=1 GM=3, WM=2
        -c 4,4,2,3 uses K=4, CSF=4, GM=2, WM=3

    `registration_mask`
        Mask used for registration to limit the metric computation to
        a specific region.


    """
    wf = pe.Workflow(name)

    if (in_template not in ('OASIS', 'NKI', 'MNI152AsymNlin2009c') and
       not os.path.exists(in_template)):
        raise RuntimeError

    if in_segmentation_model.lower() not in ('t1', 't1w', 't2', 't2w', 'flair'):
        raise NotImplementedError

    if omp_nthreads is None or omp_nthreads < 1:
        omp_nthreads = cpu_count()

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'in_template', 'in_mask']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['bias_corrected', 'out_file', 'out_mask', 'bias_image']), name='outputnode')

    trunc = pe.MapNode(ImageMath(operation='TruncateImageIntensity', op2='0.01 0.999 256'),
                       name='truncate_images', iterfield=['op1'])
    inu_n4 = pe.MapNode(
        ants.N4BiasFieldCorrection(
            dimension=3, save_bias=True, copy_header=True,
            n_iterations=[50] * 4, convergence_threshold=1e-7, shrink_factor=4,
            bspline_fitting_distance=200),
        n_procs=omp_nthreads, name='inu_n4', iterfield=['input_image'])

    res_tmpl = pe.Node(ResampleImageBySpacing(out_spacing=(4, 4, 4),
                       apply_smoothing=True), name='res_tmpl')
    res_tmpl.inputs.input_image = in_template
    res_target = pe.Node(ResampleImageBySpacing(out_spacing=(4, 4, 4),
                         apply_smoothing=True), name='res_target')

    lap_tmpl = pe.Node(ImageMath(operation='Laplacian', op2='1.5 1'),
                       name='lap_tmpl')
    lap_tmpl.inputs.input_image = in_template
    lap_target = pe.Node(ImageMath(operation='Laplacian', op2='1.5 1'),
                         name='lap_target')

    wf.connect([
        (inputnode, trunc, [('in_file', 'op1')]),
        (trunc, inu_n4, [('output_image', 'input_image')]),
        (inu_n4, outputnode, [('output_image', 'bias_corrected')]),
        (inu_n4, res_target, [
            (('output_image', _pop), 'input_image')]),
        (inu_n4, lap_target, [
            (('output_image', _pop), 'op1')]),

    ])
    return wf


def _list(in_files):
    if isinstance(in_files, (bytes, str)):
        return [in_files]
    return in_files


def _pop(in_files):
    if isinstance(in_files, (list, tuple)):
        return in_files[0]
    return in_files
