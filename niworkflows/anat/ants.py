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
from pkg_resources import resource_filename as pkgr_fn
from ..data import TEMPLATE_MAP, get_dataset
from ..nipype.pipeline import engine as pe
from ..nipype.interfaces import ants, utility as niu
from ..interfaces.ants import ImageMath, ResampleImageBySpacing, AI
from ..interfaces.fixes import FixHeaderRegistration as Registration


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

    template_path = None
    if in_template in TEMPLATE_MAP:
        template_path = get_dataset(TEMPLATE_MAP[in_template])
    else:
        template_path = in_template

    # Append template modality
    template_path = os.path.join(template_path,
                                 '1mm_%s.nii.gz' % in_segmentation_model[:2].upper())

    if not os.path.exists(template_path):
        raise ValueError(f'Template path "{template_path}" not found.')

    if in_segmentation_model.lower() not in ('t1', 't1w', 't2', 't2w', 'flair'):
        raise NotImplementedError

    if omp_nthreads is None or omp_nthreads < 1:
        omp_nthreads = cpu_count()

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'in_mask']),
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
    res_tmpl.inputs.input_image = template_path
    res_target = pe.Node(ResampleImageBySpacing(out_spacing=(4, 4, 4),
                         apply_smoothing=True), name='res_target')

    lap_tmpl = pe.Node(ImageMath(operation='Laplacian', op2='1.5 1'),
                       name='lap_tmpl')
    lap_tmpl.inputs.op1 = template_path
    lap_target = pe.Node(ImageMath(operation='Laplacian', op2='1.5 1'),
                         name='lap_target')
    mrg_tmpl = pe.Node(niu.Merge(2), name='mrg_tmpl')
    mrg_tmpl.inputs.in1 = template_path
    mrg_target = pe.Node(niu.Merge(2), name='mrg_target')

    # TODO: add extraction registration mask
    init_aff = pe.Node(AI(
        metric=('Mattes', 32, 'Regular', 0.2),
        transform=('Affine', 0.1),
        search_factor=(20, 0.12),
        # TODO search_grid=(40, (0, 40, 40)),
        principal_axes=False,
        convergence=(10, 1e-6, 10),
        verbose=True), name='init_aff', n_procs=omp_nthreads)

    norm = pe.Node(Registration(
        from_file=pkgr_fn('niworkflows.data', 'antsBrainExtraction_precise.json')),
        name='norm')

    wf.connect([
        (inputnode, trunc, [('in_file', 'op1')]),
        (trunc, inu_n4, [('output_image', 'input_image')]),
        (inu_n4, outputnode, [('output_image', 'bias_corrected')]),
        (inu_n4, res_target, [
            (('output_image', _pop), 'input_image')]),
        (inu_n4, lap_target, [
            (('output_image', _pop), 'op1')]),
        (res_tmpl, init_aff, [('output_image', 'fixed_image')]),
        (res_target, init_aff, [('output_image', 'moving_image')]),
        (inu_n4, mrg_target, [('output_image', 'in1')]),
        (lap_tmpl, mrg_tmpl, [('output_image', 'in2')]),
        (lap_target, mrg_target, [('output_image', 'in2')]),

        (init_aff, norm, [('output_transform', 'initial_moving_transform')]),
        (mrg_tmpl, norm, [('out', 'fixed_image')]),
        (mrg_target, norm, [('out', 'moving_image')]),
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
