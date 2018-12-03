#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
TemplateFlow: registration workflows
------------------------------------

"""
from __future__ import print_function, division, absolute_import, unicode_literals

# general purpose
import pkg_resources as pkgr
import logging
from multiprocessing import cpu_count

# nipype
from nipype import logging as nlogging
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces.ants import N4BiasFieldCorrection

# niworkflows
from ..data import get_template
from .ants import init_brain_extraction_wf, AI
from ..interfaces.fixes import (
    FixHeaderRegistration as Registration,
    # FixHeaderApplyTransforms as ApplyTransforms,
)


def init_templateflow_wf(
    in_files,
    mov_template,
    ref_template='MNI152NLin2009cAsym',
    use_float=True,
    omp_nthreads=None,
    mem_gb=3.0,
    modality='T1w',
    normalization_quality='precise',
    name='templateflow_wf',
):
    """
    A Nipype workflow to perform image registration between two templates
    *R* and *M*. *R* is the *reference template*, selected by a templateflow
    identifier such as ``MNI152NLin2009cAsym``, and *M* is the *moving
    template* (e.g., ``MNI152Lin``). This workflows maps data defined on
    template-*M* space onto template-*R* space.


    1. Run the subrogate images through ``antsBrainExtraction``.
    2. Recompute :abbr:`INU (intensity non-uniformity)` correction using
        the mask obtained in 1).
    3. Independently, run spatial normalization of every
       :abbr:`INU (intensity non-uniformity)` corrected image
       (supplied via ``in_files``) to both templates.
    4. Calculate an initialization between both templates, using them directly.
    5. Run multi-channel image registration of the images resulting from
        3). Both sets of images (one registered to *R* and another to *M*)
        are then used as reference and moving images in the registration
        framework.

    **Parameters**

    in_files: list of files
        a list of paths pointing to the images that will be used as surrogates
    mov_template: str
        a templateflow identifier for template-*M*
    ref_template: str
        a templateflow identifier for template-*R* (default: ``MNI152NLin2009cAsym``).


    """

    # Get path to templates
    tpl_ref_root = get_template(ref_template)
    tpl_mov_root = get_template(mov_template)

    tpl_ref = str(
        tpl_ref_root / ('tpl-%s_space-MNI_res-01_%s.nii.gz' % (ref_template, modality)))
    tpl_ref_mask = str(
        tpl_ref_root / ('tpl-%s_space-MNI_res-01_brainmask.nii.gz' % ref_template))
    tpl_mov = str(
        tpl_mov_root / ('tpl-%s_space-MNI_res-01_%s.nii.gz' % (mov_template, modality)))
    tpl_mov_mask = str(
        tpl_mov_root / ('tpl-%s_space-MNI_res-01_brainmask.nii.gz' % mov_template))
    ninputs = len(in_files)

    wf = pe.Workflow(name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_files']),
                        name='inputnode')
    inputnode.iterables = ('in_files', in_files)

    ref_bex = init_brain_extraction_wf(
        in_template=ref_template,
        omp_nthreads=omp_nthreads,
        mem_gb=mem_gb,
        modality=modality[:2],
        name='reference_bex',
    )

    mov_bex = init_brain_extraction_wf(
        in_template=mov_template,
        omp_nthreads=omp_nthreads,
        mem_gb=mem_gb,
        modality=modality[:2],
        name='moving_bex',
    )

    ref_inu = pe.MapNode(
        N4BiasFieldCorrection(
            dimension=3, save_bias=True, copy_header=True,
            n_iterations=[50] * 5, convergence_threshold=1e-7, shrink_factor=4,
            bspline_fitting_distance=200),
        n_procs=omp_nthreads, name='ref_inu',
        iterfield=['input_image', 'mask_image'])

    ref_norm = pe.MapNode(
        Registration(
            from_file=pkgr.resource_filename(
                'niworkflows.data', 't1w-mni_registration_%s_000.json' % normalization_quality)),
        name='ref_norm', n_procs=omp_nthreads,
        iterfield=['moving_image', 'moving_image_masks', 'initial_moving_transform'])
    ref_norm.inputs.fixed_image = tpl_ref
    ref_norm.inputs.fixed_image_masks = tpl_ref_mask
    ref_norm.inputs.environ = {
        'NSLOTS': '%d' % omp_nthreads,
        'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS': '%d' % omp_nthreads,
        'OMP_NUM_THREADS': '%d' % omp_nthreads,
    }

    # Register the INU-corrected image to the other template
    mov_norm = pe.MapNode(
        Registration(
            from_file=pkgr.resource_filename(
                'niworkflows.data', 't1w-mni_registration_%s_000.json' % normalization_quality)),
        name='mov_norm', n_procs=omp_nthreads,
        iterfield=['moving_image', 'moving_image_masks', 'initial_moving_transform'])
    mov_norm.inputs.fixed_image = tpl_mov
    mov_norm.inputs.fixed_image_masks = tpl_mov_mask
    mov_norm.inputs.environ = {
        'NSLOTS': '%d' % omp_nthreads,
        'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS': '%d' % omp_nthreads,
        'OMP_NUM_THREADS': '%d' % omp_nthreads,
    }

    # Initialize between-templates transform with antsAI
    init_aff = pe.Node(AI(
        metric=('Mattes', 32, 'Regular', 0.2),
        transform=('Affine', 0.1),
        search_factor=(20, 0.12),
        principal_axes=False,
        convergence=(10, 1e-6, 10),
        verbose=True),
        name='init_aff',
        n_procs=omp_nthreads)
    init_aff.inputs.fixed_image = tpl_ref
    init_aff.inputs.fixed_image_mask = tpl_ref_mask
    init_aff.inputs.moving_image = tpl_mov
    init_aff.inputs.moving_image_mask = tpl_mov_mask

    ref_buffer = pe.JoinNode(niu.IdentityInterface(
        fields=['fixed_image']),
        joinsource='inputnode', joinfield='fixed_image', name='ref_buffer')

    mov_buffer = pe.JoinNode(niu.IdentityInterface(
        fields=['moving_image']),
        joinsource='inputnode', joinfield='moving_image', name='mov_buffer')

    flow = pe.Node(
        Registration(
            from_file=pkgr.resource_filename(
                'niworkflows.data', 't1w-mni_registration_%s_000.json' % normalization_quality)),
        name='flow_norm', n_procs=omp_nthreads,
    )
    flow.inputs.fixed_image_masks = tpl_ref_mask
    flow.inputs.moving_image_masks = tpl_mov_mask
    flow.inputs.metric = [[v] * ninputs for v in flow.inputs.metric]
    flow.inputs.metric_weight = [[1 / ninputs] * ninputs
                                 for _ in flow.inputs.metric_weight]
    flow.inputs.radius_or_number_of_bins = [
        [v] * ninputs for v in flow.inputs.radius_or_number_of_bins]
    flow.inputs.sampling_percentage = [
        [v] * ninputs for v in flow.inputs.sampling_percentage]
    flow.inputs.sampling_strategy = [
        [v] * ninputs for v in flow.inputs.sampling_strategy]
    flow.inputs.environ = {
        'NSLOTS': '%d' % omp_nthreads,
        'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS': '%d' % omp_nthreads,
        'OMP_NUM_THREADS': '%d' % omp_nthreads,
    }

    wf.connect([
        (inputnode, ref_bex, [('in_files', 'inputnode.in_files')]),
        (inputnode, mov_bex, [('in_files', 'inputnode.in_files')]),
        (inputnode, ref_inu, [('in_files', 'input_image')]),
        (ref_bex, ref_inu, [('outputnode.out_mask', 'mask_image')]),
        (ref_inu, ref_norm, [('output_image', 'moving_image')]),
        (ref_bex, ref_norm, [('outputnode.out_mask', 'moving_image_masks'),
                             ('outputnode.out_fwd_xfm', 'initial_moving_transform')]),
        (ref_inu, mov_norm, [('output_image', 'moving_image')]),
        (mov_bex, mov_norm, [('outputnode.out_mask', 'moving_image_masks'),
                             ('outputnode.out_fwd_xfm', 'initial_moving_transform')]),
        (init_aff, flow, [('output_transform', 'initial_moving_transform')]),
        (ref_norm, ref_buffer, [('warped_image', 'fixed_image')]),
        (mov_norm, mov_buffer, [('warped_image', 'moving_image')]),
        (ref_buffer, flow, [(('fixed_image', _flatten), 'fixed_image')]),
        (mov_buffer, flow, [(('moving_image', _flatten), 'moving_image')]),

    ])
    return wf


def _flatten(inlist):
    return [item for sublist in inlist for item in sublist]


def cli():
    """Run templateflow on commandline"""
    from pathlib import Path
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter

    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument('input_files', nargs='+', action='store', help='the input file',
                        type=Path)
    parser.add_argument('--reference', '-r', action='store', help='the reference template')
    parser.add_argument('--moving', '-m', action='store', help='the moving template')
    parser.add_argument('--cpu-count', '--nproc', action='store', type=int,
                        default=cpu_count(), help='number of processors')
    parser.add_argument('--omp-nthreads', action='store', type=int,
                        default=cpu_count(), help='number of threads')
    parser.add_argument('--testing', action='store_true',
                        default=False, help='run in testing mode')
    parser.add_argument("-v", "--verbose", dest="verbose_count", action="count", default=0,
                        help="increases log verbosity for each occurence, debug level is -vvv")
    parser.add_argument('--legacy', action='store_true', default=False,
                        help='use LegacyMultiProc')
    opts = parser.parse_args()

    plugin_settings = {'plugin': 'Linear'}
    if opts.cpu_count > 1:
        plugin_settings = {
            'plugin': 'LegacyMultiProc' if opts.legacy else 'MultiProc',
            'plugin_args': {
                'raise_insufficient': False,
                'maxtasksperchild': 1,
                'n_procs': opts.cpu_count,
            }
        }

    # Retrieve logging level
    log_level = int(max(25 - 5 * opts.verbose_count, logging.DEBUG))
    # Set logging
    nlogging.getLogger('nipype.workflow').setLevel(log_level)
    nlogging.getLogger('nipype.interface').setLevel(log_level)
    nlogging.getLogger('nipype.utils').setLevel(log_level)

    tf = init_templateflow_wf(
        [[str(f)] for f in opts.input_files],
        opts.moving, ref_template=opts.reference,
        omp_nthreads=opts.omp_nthreads,
        normalization_quality='precise' if not opts.testing else 'testing')
    tf.base_dir = str(Path().resolve())
    tf.run(**plugin_settings)
