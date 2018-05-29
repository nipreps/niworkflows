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
from packaging.version import parse as parseversion, Version
from ..data import TEMPLATE_MAP, get_dataset
from ..nipype.pipeline import engine as pe
from ..nipype.interfaces import utility as niu
from ..nipype.interfaces.ants import N4BiasFieldCorrection, Atropos
from ..interfaces.ants import (
    ImageMath,
    ResampleImageBySpacing,
    AI,
    ThresholdImage,
)
from ..interfaces.fixes import (
    FixHeaderRegistration as Registration,
    FixHeaderApplyTransforms as ApplyTransforms,
)

ATROPOS_MODELS = {
    'T1': (3, 1, 2, 3),
    'T2': (3, 3, 2, 1),
    'FLAIR': (3, 1, 3, 2),
}


def brain_extraction(name='antsBrainExtraction',
                     in_template='OASIS',
                     use_float=True,
                     debug=False,
                     omp_nthreads=None,
                     mem_gb=3.0,
                     in_segmentation_model='T1',
                     atropos_use_random_seed=True,
                     atropos_workflow=False):
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
    tpl_target_path = os.path.join(template_path,
                                   '1mm_%s.nii.gz' % in_segmentation_model[:2].upper())

    if not os.path.exists(tpl_target_path):
        raise ValueError(f'Template path "{tpl_target_path}" not found.')


    if omp_nthreads is None or omp_nthreads < 1:
        omp_nthreads = cpu_count()

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'in_mask']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['bias_corrected', 'out_file', 'out_mask', 'bias_image']), name='outputnode')

    trunc = pe.MapNode(ImageMath(operation='TruncateImageIntensity', op2='0.01 0.999 256'),
                       name='truncate_images', iterfield=['op1'])
    inu_n4 = pe.MapNode(
        N4BiasFieldCorrection(
            dimension=3, save_bias=True, copy_header=True,
            n_iterations=[50] * 4, convergence_threshold=1e-7, shrink_factor=4,
            bspline_fitting_distance=200),
        n_procs=omp_nthreads, name='inu_n4', iterfield=['input_image'])

    res_tmpl = pe.Node(ResampleImageBySpacing(out_spacing=(4, 4, 4),
                       apply_smoothing=True), name='res_tmpl')
    res_tmpl.inputs.input_image = tpl_target_path
    res_target = pe.Node(ResampleImageBySpacing(out_spacing=(4, 4, 4),
                         apply_smoothing=True), name='res_target')

    lap_tmpl = pe.Node(ImageMath(operation='Laplacian', op2='1.5 1'),
                       name='lap_tmpl')
    lap_tmpl.inputs.op1 = tpl_target_path
    lap_target = pe.Node(ImageMath(operation='Laplacian', op2='1.5 1'),
                         name='lap_target')
    mrg_tmpl = pe.Node(niu.Merge(2), name='mrg_tmpl')
    mrg_tmpl.inputs.in1 = tpl_target_path
    mrg_target = pe.Node(niu.Merge(2), name='mrg_target')

    init_aff = pe.Node(AI(
        metric=('Mattes', 32, 'Regular', 0.2),
        transform=('Affine', 0.1),
        search_factor=(20, 0.12),
        principal_axes=False,
        convergence=(10, 1e-6, 10),
        verbose=True),
        name='init_aff',
        n_procs=omp_nthreads)

    if parseversion(Registration().version) > Version('2.2.0'):
        init_aff.inputs.search_grid = (40, (0, 40, 40))

    norm = pe.Node(Registration(
        from_file=pkgr_fn('niworkflows.data', 'antsBrainExtraction_precise.json')),
        name='norm',
        n_procs=omp_nthreads,
        mem_gb=mem_gb)
    norm.inputs.float = use_float

    map_brainmask = pe.Node(
        ApplyTransforms(interpolation='Gaussian', float=True),
        name='map_brainmask',
        mem_gb=1
    )
    map_brainmask.inputs.input_image = os.path.join(
        template_path, '1mm_brainprobmask.nii.gz')

    thr_brainmask = pe.Node(ThresholdImage(
        dimension=3, th_low=0.5, th_high=1.0, inside_value=1,
        outside_value=0), name='thr_brainmask')

    # Morpholgical dilation, radius=2
    dil_brainmask = pe.Node(ImageMath(operation='MD', op2='2'),
                            name='dil_brainmask')
    # Get largest connected component
    get_brainmask = pe.Node(ImageMath(operation='GetLargestComponent'),
                            name='get_brainmask')

    wf.connect([
        (inputnode, trunc, [('in_file', 'op1')]),
        (inputnode, init_aff, [('in_mask', 'fixed_image_mask')]),
        (inputnode, norm, [('in_mask', 'fixed_image_mask')]),
        (inputnode, map_brainmask, [(('in_file', _pop), 'reference_image')]),
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
        (norm, map_brainmask, [('forward_transforms', 'transforms')]),
        (map_brainmask, thr_brainmask, [('output_image', 'input_image')]),
        (thr_brainmask, dil_brainmask, [('output_image', 'op1')]),
        (dil_brainmask, get_brainmask, [('output_image', 'op1')]),
        (get_brainmask, outputnode, [('output_image', 'out_mask')]),
    ])
    return wf


def atropos_workflow(name='atropos_wf',
                     use_float=True,
                     debug=False,
                     use_random_seed=True,
                     omp_nthreads=None,
                     mem_gb=3.0,
                     padding=10,
                     in_segmentation_model='T1',
                     run_atropos_workflow=False):
    """
    Implements superstep 6 of ``antsBrainExtraction.sh``
    """
    wf = pe.Workflow(name)

    if in_segmentation_model.upper() not in ATROPOS_MODELS:
        raise NotImplementedError
    else:
        in_segmentation_model = ATROPOS_MODELS[in_segmentation_model.upper()]

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_files', 'in_mask']),
                        name='inputnode')

    # Run atropos (core node)
    atropos = pe.Node(Atropos(
        dimension=3,
        initialization='KMeans',
        number_of_tissue_classes=in_segmentation_model[0],
        n_iterations=3,
        convergence_threshold=0.0,
        mrf_radius=[1, 1, 1],
        mrf_smoothing_factor=0.1,
        likelihood_model='Gaussian',
        use_random_seed=use_random_seed),
        name='atropos', n_procs=omp_nthreads, mem_gb=mem_gb)

    # massage outputs
    pad_segm = pe.Node(ImageMath(operation='PadImage', op2='%d' % padding),
                       name='pad_segm')
    pad_mask = pe.Node(ImageMath(operation='PadImage', op2='%d' % padding),
                       name='pad_mask')

    # Split segmentation in binary masks
    select_labels = pe.MapNode(niu.Function(function=_select_labels),
                               iterfield=['label'])
    select_labels.label = list(reversed(in_segmentation_model[1:]))

    # Select WM (element 0)
    sel_wm = pe.Node(niu.Select(index=[0]), name='sel_wm',
                     run_without_submitting=True)
    # Select GM (element 1)
    sel_gm = pe.Node(niu.Select(index=[1]), name='sel_gm',
                     run_without_submitting=True)

    # Select largest components (GM, WM)
    # ImageMath ${DIMENSION} ${EXTRACTION_WM} GetLargestComponent ${EXTRACTION_WM}

    # Fill holes and calculate intersection
    # ImageMath ${DIMENSION} ${EXTRACTION_TMP} FillHoles ${EXTRACTION_GM} 2
    # MultiplyImages ${DIMENSION} ${EXTRACTION_GM} ${EXTRACTION_TMP} ${EXTRACTION_GM}

    # MultiplyImages ${DIMENSION} ${EXTRACTION_WM} ${ATROPOS_WM_CLASS_LABEL} ${EXTRACTION_WM}
    # ImageMath ${DIMENSION} ${EXTRACTION_TMP} ME ${EXTRACTION_CSF} 10

    # ImageMath ${DIMENSION} ${EXTRACTION_GM} addtozero ${EXTRACTION_GM} ${EXTRACTION_TMP}
    # MultiplyImages ${DIMENSION} ${EXTRACTION_GM} ${ATROPOS_GM_CLASS_LABEL} ${EXTRACTION_GM}
    # ImageMath ${DIMENSION} ${EXTRACTION_SEGMENTATION} addtozero ${EXTRACTION_WM} ${EXTRACTION_GM}

    wf.connect([
        (inputnode, pad_mask, [('in_mask', 'op1')]),
        (inputnode, atropos, [('in_files', 'intensity_images'),
                              ('in_mask', 'mask_image')]),
        (atropos, pad_mask, [('classified_image', 'op1')]),
        (atropos, select_labels, [('classified_image', 'in_labels')]),
        (select_labels, sel_wm, [('out', 'inlist')]),
        (select_labels, sel_gm, [('out', 'inlist')]),

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


def _select_labels(in_labels, label):
    import numpy as np
    import nibabel as nb
    from niworkflows.nipype.utils.filemanip import fname_presuffix

    nii = nb.load(in_labels)
    data = np.zeros(nii.shape)
    data[nii.get_data() == label] = 1
    newnii = nii.__class__(data, nii.affine, nii.header)
    newnii.set_data_dtype('uint8')
    out_file = fname_presuffix(in_labels, suffix='_class-%02d' % label)
    newnii.to_filename(out_file)
    return out_file
