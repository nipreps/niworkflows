#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
TemplateFlow: registration workflows
------------------------------------

"""

# general purpose
import pkg_resources as pkgr
import logging
from os import getenv, cpu_count

# nipype
from nipype import logging as nlogging
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import freesurfer as fs
from nipype.interfaces.ants import N4BiasFieldCorrection, ApplyTransformsToPoints
from nipype.interfaces.io import FreeSurferSource

# niworkflows
from ..data import get_template
from .ants import init_brain_extraction_wf
from ..interfaces.ants import AI, AntsJointFusion
from ..interfaces.fixes import (
    FixHeaderRegistration as Registration,
    FixHeaderApplyTransforms as ApplyTransforms,
)
from ..interfaces.bids import DerivativesDataSink
from ..interfaces.surf import (
    GiftiToCSV, CSVToGifti, SurfacesToPointCloud, PoissonRecon,
    UnzipJoinedSurfaces, PLYtoGifti,
)
from .freesurfer import init_gifti_surface_wf


def init_templateflow_wf(
    bids_dir,
    output_dir,
    participant_label,
    mov_template,
    ref_template='MNI152NLin2009cAsym',
    use_float=True,
    omp_nthreads=None,
    mem_gb=3.0,
    modality='T1w',
    normalization_quality='precise',
    name='templateflow_wf',
    fs_subjects_dir=getenv('SUBJECTS_DIR'),
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
    ninputs = len(participant_label)

    ants_env = {
        'NSLOTS': '%d' % omp_nthreads,
        'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS': '%d' % omp_nthreads,
        'OMP_NUM_THREADS': '%d' % omp_nthreads,
    }

    wf = pe.Workflow(name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['participant_label']),
                        name='inputnode')
    inputnode.iterables = ('participant_label', sorted(list(participant_label)))

    pick_file = pe.Node(niu.Function(function=_bids_pick),
                        name='pick_file', run_without_submitting=True)
    pick_file.inputs.bids_root = bids_dir

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

    ref_inu = pe.Node(
        N4BiasFieldCorrection(
            dimension=3, save_bias=True, copy_header=True,
            n_iterations=[50] * 5, convergence_threshold=1e-7, shrink_factor=4,
            bspline_fitting_distance=200, environ=ants_env),
        n_procs=omp_nthreads, name='ref_inu')

    ref_norm = pe.Node(
        Registration(
            from_file=pkgr.resource_filename(
                'niworkflows.data', 't1w-mni_registration_%s_000.json' % normalization_quality)),
        name='ref_norm', n_procs=omp_nthreads)
    ref_norm.inputs.fixed_image = tpl_ref
    ref_norm.inputs.fixed_image_masks = tpl_ref_mask
    ref_norm.inputs.environ = ants_env

    # Register the INU-corrected image to the other template
    mov_norm = pe.Node(
        Registration(
            from_file=pkgr.resource_filename(
                'niworkflows.data', 't1w-mni_registration_%s_000.json' % normalization_quality)),
        name='mov_norm', n_procs=omp_nthreads)
    mov_norm.inputs.fixed_image = tpl_mov
    mov_norm.inputs.fixed_image_masks = tpl_mov_mask
    mov_norm.inputs.environ = ants_env

    # Initialize between-templates transform with antsAI
    init_aff = pe.Node(AI(
        metric=('Mattes', 32, 'Regular', 0.2),
        transform=('Affine', 0.1),
        search_factor=(20, 0.12),
        principal_axes=False,
        convergence=(10, 1e-6, 10),
        verbose=True,
        fixed_image=tpl_ref,
        fixed_image_mask=tpl_ref_mask,
        moving_image=tpl_mov,
        moving_image_mask=tpl_mov_mask,
        environ=ants_env,
    ), name='init_aff', n_procs=omp_nthreads)

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
    flow.inputs.environ = ants_env

    fssource = pe.Node(
        FreeSurferSource(subjects_dir=str(fs_subjects_dir)),
        name='fssource', run_without_submitting=True)
    tonative = pe.Node(fs.Label2Vol(subjects_dir=str(fs_subjects_dir)),
                       name='tonative')
    tonii = pe.Node(
        fs.MRIConvert(out_type='niigz', resample_type='nearest'),
        name='tonii')

    ref_aparc = pe.Node(
        ApplyTransforms(interpolation='MultiLabel', float=True,
                        reference_image=tpl_ref, environ=ants_env),
        name='ref_aparc', mem_gb=1, n_procs=omp_nthreads
    )

    mov_aparc = pe.Node(
        ApplyTransforms(interpolation='MultiLabel', float=True,
                        reference_image=tpl_mov, environ=ants_env),
        name='mov_aparc', mem_gb=1, n_procs=omp_nthreads
    )

    ref_aparc_buffer = pe.JoinNode(
        niu.IdentityInterface(fields=['aparc']),
        joinsource='inputnode', joinfield='aparc', name='ref_aparc_buffer')

    ref_join_labels = pe.Node(
        AntsJointFusion(
            target_image=[tpl_ref],
            out_label_fusion='merged_aparc.nii.gz',
            out_intensity_fusion_name_format='merged_aparc_intensity_%d.nii.gz',
            out_label_post_prob_name_format='merged_aparc_posterior_%d.nii.gz',
            out_atlas_voting_weight_name_format='merged_aparc_weight_%d.nii.gz',
            environ=ants_env,
        ),
        name='ref_join_labels', n_procs=omp_nthreads)

    ref_join_labels_ds = pe.Node(
        DerivativesDataSink(
            base_directory=str(output_dir.parent),
            out_path_base=output_dir.name,
            suffix='dtissue', desc='aparc', keep_dtype=False,
            source_file='group/tpl-{0}_T1w.nii.gz'.format(ref_template)),
        name='ref_join_labels_ds', run_without_submitting=True)

    ref_join_probs_ds = pe.Node(
        DerivativesDataSink(
            base_directory=str(output_dir.parent),
            out_path_base=output_dir.name,
            suffix='probtissue', desc='aparc', keep_dtype=False,
            source_file='group/tpl-{0}_T1w.nii.gz'.format(ref_template)),
        name='ref_join_probs_ds', run_without_submitting=True)

    # ref_join_voting_ds = pe.Node(
    #     DerivativesDataSink(
    #         base_directory=str(output_dir.parent),
    #         out_path_base=output_dir.name, space=ref_template,
    #         suffix='probtissue', desc='aparcvoting', keep_dtype=False,
    #         source_file='group/tpl-{0}_T1w.nii.gz'.format(ref_template)),
    #     name='ref_join_voting_ds', run_without_submitting=True)

    mov_aparc_buffer = pe.JoinNode(
        niu.IdentityInterface(fields=['aparc']),
        joinsource='inputnode', joinfield='aparc', name='mov_aparc_buffer')

    mov_join_labels = pe.Node(
        AntsJointFusion(
            target_image=[tpl_mov],
            out_label_fusion='merged_aparc.nii.gz',
            out_intensity_fusion_name_format='merged_aparc_intensity_%d.nii.gz',
            out_label_post_prob_name_format='merged_aparc_posterior_%d.nii.gz',
            out_atlas_voting_weight_name_format='merged_aparc_weight_%d.nii.gz',
            environ=ants_env,
        ),
        name='mov_join_labels', n_procs=omp_nthreads)

    mov_join_labels_ds = pe.Node(
        DerivativesDataSink(
            base_directory=str(output_dir.parent),
            out_path_base=output_dir.name,
            suffix='dtissue', desc='aparc', keep_dtype=False,
            source_file='group/tpl-{0}_T1w.nii.gz'.format(mov_template)),
        name='mov_join_labels_ds', run_without_submitting=True)

    mov_join_probs_ds = pe.Node(
        DerivativesDataSink(
            base_directory=str(output_dir.parent),
            out_path_base=output_dir.name,
            suffix='probtissue', desc='aparc', keep_dtype=False,
            source_file='group/tpl-{0}_T1w.nii.gz'.format(mov_template)),
        name='mov_join_probs_ds', run_without_submitting=True)

    # Datasinking
    ref_norm_ds = pe.Node(
        DerivativesDataSink(base_directory=str(output_dir.parent),
                            out_path_base=output_dir.name, space=ref_template,
                            desc='preproc', keep_dtype=True),
        name='ref_norm_ds', run_without_submitting=True
    )

    mov_norm_ds = pe.Node(
        DerivativesDataSink(base_directory=str(output_dir.parent),
                            out_path_base=output_dir.name, space=mov_template,
                            desc='preproc', keep_dtype=True),
        name='mov_norm_ds', run_without_submitting=True
    )

    ref_aparc_ds = pe.Node(
        DerivativesDataSink(base_directory=str(output_dir.parent),
                            out_path_base=output_dir.name, space=ref_template,
                            suffix='dtissue', desc='aparc', keep_dtype=False),
        name='ref_aparc_ds', run_without_submitting=True
    )

    mov_aparc_ds = pe.Node(
        DerivativesDataSink(base_directory=str(output_dir.parent),
                            out_path_base=output_dir.name, space=mov_template,
                            suffix='dtissue', desc='aparc', keep_dtype=False),
        name='mov_aparc_ds', run_without_submitting=True
    )

    # Extract surfaces
    cifti_wf = init_gifti_surface_wf(
        name='cifti_surfaces',
        subjects_dir=str(fs_subjects_dir))

    # Move surfaces to template spaces
    gii2csv = pe.MapNode(GiftiToCSV(itk_lps=True),
                         iterfield=['in_file'], name='gii2csv')
    ref_map_surf = pe.MapNode(
        ApplyTransformsToPoints(dimension=3, environ=ants_env),
        n_procs=omp_nthreads, name='ref_map_surf', iterfield=['input_file'])
    ref_csv2gii = pe.MapNode(
        CSVToGifti(itk_lps=True),
        name='ref_csv2gii', iterfield=['in_file', 'gii_file'])
    ref_surfs_buffer = pe.JoinNode(
        niu.IdentityInterface(fields=['surfaces']),
        joinsource='inputnode', joinfield='surfaces', name='ref_surfs_buffer')
    ref_surfs_unzip = pe.Node(UnzipJoinedSurfaces(), name='ref_surfs_unzip',
                              run_without_submitting=True)
    ref_ply = pe.MapNode(SurfacesToPointCloud(), name='ref_ply',
                         iterfield=['in_files'])
    ref_recon = pe.MapNode(PoissonRecon(), name='ref_recon',
                           iterfield=['in_file'])
    ref_avggii = pe.MapNode(PLYtoGifti(), name='ref_avggii',
                            iterfield=['in_file', 'surf_key'])
    ref_smooth = pe.MapNode(fs.SmoothTessellation(), name='ref_smooth',
                            iterfield=['in_file'])

    ref_surfs_ds = pe.Node(
        DerivativesDataSink(
            base_directory=str(output_dir.parent),
            out_path_base=output_dir.name, space=ref_template,
            keep_dtype=False, compress=False),
        name='ref_surfs_ds', run_without_submitting=True)
    ref_avg_ds = pe.Node(
        DerivativesDataSink(
            base_directory=str(output_dir.parent),
            out_path_base=output_dir.name, space=ref_template,
            keep_dtype=False, compress=False,
            source_file='group/tpl-{0}_T1w.nii.gz'.format(ref_template)),
        name='ref_avg_ds', run_without_submitting=True)

    mov_map_surf = pe.MapNode(
        ApplyTransformsToPoints(dimension=3, environ=ants_env),
        n_procs=omp_nthreads, name='mov_map_surf', iterfield=['input_file'])
    mov_csv2gii = pe.MapNode(
        CSVToGifti(itk_lps=True),
        name='mov_csv2gii', iterfield=['in_file', 'gii_file'])
    mov_surfs_buffer = pe.JoinNode(
        niu.IdentityInterface(fields=['surfaces']),
        joinsource='inputnode', joinfield='surfaces', name='mov_surfs_buffer')
    mov_surfs_unzip = pe.Node(UnzipJoinedSurfaces(), name='mov_surfs_unzip',
                              run_without_submitting=True)
    mov_ply = pe.MapNode(SurfacesToPointCloud(), name='mov_ply',
                         iterfield=['in_files'])
    mov_recon = pe.MapNode(PoissonRecon(), name='mov_recon',
                           iterfield=['in_file'])
    mov_avggii = pe.MapNode(PLYtoGifti(), name='mov_avggii',
                            iterfield=['in_file', 'surf_key'])
    mov_smooth = pe.MapNode(fs.SmoothTessellation(), name='mov_smooth',
                            iterfield=['in_file'])

    mov_surfs_ds = pe.Node(
        DerivativesDataSink(
            base_directory=str(output_dir.parent),
            out_path_base=output_dir.name, space=mov_template,
            keep_dtype=False, compress=False),
        name='mov_surfs_ds', run_without_submitting=True)
    mov_avg_ds = pe.Node(
        DerivativesDataSink(
            base_directory=str(output_dir.parent),
            out_path_base=output_dir.name, space=mov_template,
            keep_dtype=False, compress=False,
            source_file='group/tpl-{0}_T1w.nii.gz'.format(mov_template)),
        name='mov_avg_ds', run_without_submitting=True)

    wf.connect([
        (inputnode, pick_file, [('participant_label', 'participant_label')]),
        (inputnode, fssource, [(('participant_label', _sub_decorate), 'subject_id')]),
        (inputnode, cifti_wf, [
            (('participant_label', _sub_decorate), 'inputnode.subject_id')]),
        (pick_file, cifti_wf, [('out', 'inputnode.in_t1w')]),
        (pick_file, ref_bex, [('out', 'inputnode.in_files')]),
        (pick_file, mov_bex, [('out', 'inputnode.in_files')]),
        (pick_file, ref_inu, [('out', 'input_image')]),
        (pick_file, tonii, [('out', 'reslice_like')]),
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
        (ref_buffer, flow, [('fixed_image', 'fixed_image')]),
        (mov_buffer, flow, [('moving_image', 'moving_image')]),
        # Select DKT aparc
        (fssource, tonative, [(('aparc_aseg', _last), 'seg_file'),
                              ('rawavg', 'template_file'),
                              ('aseg', 'reg_header')]),
        (tonative, tonii, [('vol_label_file', 'in_file')]),
        (tonii, ref_aparc, [('out_file', 'input_image')]),
        (tonii, mov_aparc, [('out_file', 'input_image')]),
        (ref_norm, ref_aparc, [('composite_transform', 'transforms')]),
        (mov_norm, mov_aparc, [('composite_transform', 'transforms')]),
        (ref_buffer, ref_join_labels, [
            ('fixed_image', 'atlas_image')]),
        (ref_aparc, ref_aparc_buffer, [('output_image', 'aparc')]),
        (ref_aparc_buffer, ref_join_labels, [
            ('aparc', 'atlas_segmentation_image')]),
        (mov_buffer, mov_join_labels, [
            ('moving_image', 'atlas_image')]),
        (mov_aparc, mov_aparc_buffer, [('output_image', 'aparc')]),
        (mov_aparc_buffer, mov_join_labels, [
            ('aparc', 'atlas_segmentation_image')]),
        # Datasinks
        (ref_join_labels, ref_join_labels_ds, [('out_label_fusion', 'in_file')]),
        (ref_join_labels, ref_join_probs_ds, [
            ('out_label_post_prob', 'in_file'),
            (('out_label_post_prob', _get_extra), 'extra_values')]),
        # (ref_join_labels, ref_join_voting_ds, [
        #     ('out_atlas_voting_weight_name_format', 'in_file')]),
        (mov_join_labels, mov_join_labels_ds, [('out_label_fusion', 'in_file')]),
        (mov_join_labels, mov_join_probs_ds, [
            ('out_label_post_prob', 'in_file'),
            (('out_label_post_prob', _get_extra), 'extra_values')]),
        (pick_file, ref_norm_ds, [('out', 'source_file')]),
        (ref_norm, ref_norm_ds, [('warped_image', 'in_file')]),
        (pick_file, mov_norm_ds, [('out', 'source_file')]),
        (mov_norm, mov_norm_ds, [('warped_image', 'in_file')]),
        (pick_file, ref_aparc_ds, [('out', 'source_file')]),
        (ref_aparc, ref_aparc_ds, [('output_image', 'in_file')]),
        (pick_file, mov_aparc_ds, [('out', 'source_file')]),
        (mov_aparc, mov_aparc_ds, [('output_image', 'in_file')]),
        # Mapping ref surfaces
        (cifti_wf, gii2csv, [
            (('outputnode.surf_norm', _discard_inflated), 'in_file')]),
        (gii2csv, ref_map_surf, [('out_file', 'input_file')]),
        (ref_norm, ref_map_surf, [
            (('inverse_composite_transform', _ensure_list), 'transforms')]),
        (ref_map_surf, ref_csv2gii, [('output_file', 'in_file')]),
        (cifti_wf, ref_csv2gii, [
            (('outputnode.surf_norm', _discard_inflated), 'gii_file')]),
        (pick_file, ref_surfs_ds, [('out', 'source_file')]),
        (ref_csv2gii, ref_surfs_ds, [
            ('out_file', 'in_file'),
            (('out_file', _get_surf_extra), 'extra_values')]),
        (ref_csv2gii, ref_surfs_buffer, [('out_file', 'surfaces')]),
        (ref_surfs_buffer, ref_surfs_unzip, [('surfaces', 'in_files')]),
        (ref_surfs_unzip, ref_ply, [('out_files', 'in_files')]),
        (ref_ply, ref_recon, [('out_file', 'in_file')]),
        (ref_recon, ref_avggii, [('out_file', 'in_file')]),
        (ref_surfs_unzip, ref_avggii, [('surf_keys', 'surf_key')]),
        (ref_avggii, ref_smooth, [('out_file', 'in_file')]),
        (ref_smooth, ref_avg_ds, [
            ('surface', 'in_file'),
            (('surface', _get_surf_extra), 'extra_values')]),

        # Mapping mov surfaces
        (gii2csv, mov_map_surf, [('out_file', 'input_file')]),
        (mov_norm, mov_map_surf, [
            (('inverse_composite_transform', _ensure_list), 'transforms')]),
        (mov_map_surf, mov_csv2gii, [('output_file', 'in_file')]),
        (cifti_wf, mov_csv2gii, [
            (('outputnode.surf_norm', _discard_inflated), 'gii_file')]),
        (pick_file, mov_surfs_ds, [('out', 'source_file')]),
        (mov_csv2gii, mov_surfs_ds, [
            ('out_file', 'in_file'),
            (('out_file', _get_surf_extra), 'extra_values')]),
        (mov_csv2gii, mov_surfs_buffer, [('out_file', 'surfaces')]),
        (mov_surfs_buffer, mov_surfs_unzip, [('surfaces', 'in_files')]),
        (mov_surfs_unzip, mov_ply, [('out_files', 'in_files')]),
        (mov_ply, mov_recon, [('out_file', 'in_file')]),
        (mov_recon, mov_avggii, [('out_file', 'in_file')]),
        (mov_surfs_unzip, mov_avggii, [('surf_keys', 'surf_key')]),
        (mov_avggii, mov_smooth, [('out_file', 'in_file')]),
        (mov_smooth, mov_avg_ds, [
            ('surface', 'in_file'),
            (('surface', _get_surf_extra), 'extra_values')]),
    ])
    return wf


def _ensure_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, (str, bytes)):
        return [value]
    return list(value)


def _last(inlist):
    return inlist[-1]


def _flatten(inlist):
    return [item for sublist in inlist for item in sublist]


def _bids_pick(participant_label, bids_root):
    return str(bids_root / ('sub-%s' % participant_label) / 'anat' /
               ('sub-%s_T1w.nii.gz' % participant_label))


def _sub_decorate(label):
    return "sub-%s" % label


def _get_extra(inlist):
    return ['class-%s' % s.rstrip(
        '.gz').rstrip('.nii').split('_')[-1] for s in inlist]


def _get_surf_extra(inlist):
    newlist = []
    for item in inlist:
        desc = None
        hemi = 'L'
        if 'rh' in item:
            hemi = 'R'
        if 'smoothwm' in item:
            desc = 'smoothwm'
        elif 'pial' in item:
            desc = 'pial'
        elif 'midthickness' in item:
            desc = 'midthickness'
        else:
            raise RuntimeError("Unknown surface %s" % item)
        newlist.append('hemi-%s_%s.surf' % (hemi, desc))
    return newlist


def _discard_inflated(inlist):
    return [s for s in inlist if 'inflated' not in s]


def cli():
    """Run templateflow on commandline"""
    from pathlib import Path
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter

    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument('bids_dir', action='store', help='BIDS directory root',
                        type=Path)
    parser.add_argument('--participant-label', nargs='*', action='store',
                        help='list of participants to be processed')
    parser.add_argument('--output-dir', action='store', help='output directory',
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
    parser.add_argument('-w', '--work-dir', action='store', type=Path,
                        default=Path() / 'work', help='work directory')
    parser.add_argument('--freesurfer', action='store', type=Path,
                        help='path to precomputed freesurfer results',
                        default=Path(getenv('SUBJECTS_DIR', 'freesurfer/')).resolve())
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

    bids_dir = opts.bids_dir.resolve()
    if not bids_dir.exists():
        raise RuntimeError(
            'Input BIDS directory "%s" does not exist.' % bids_dir)

    if not opts.output_dir:
        output_dir = bids_dir / 'derivatives' / 'templateflow_0.0.1'
    else:
        output_dir = opts.output_dir.resolve()

    all_subjects = set([Path(s).name.split('_')[0].split('-')[-1]
                        for s in bids_dir.glob('sub-*')])

    if not all_subjects:
        raise RuntimeError(
            'Input BIDS directory "%s" does not look like a valid BIDS '
            'tree.' % bids_dir)

    if not opts.participant_label:
        participants = all_subjects
    else:
        part_label = [s[4:] if s.startswith('sub-') else s
                      for s in opts.participant_label]
        participants = all_subjects.intersection(part_label)

    if not participants:
        raise RuntimeError(
            'No subject with the specified participant labels (%s) matched.' %
            ', '.join(opts.participant_label))

    tf = init_templateflow_wf(
        bids_dir,
        output_dir,
        participants,
        opts.moving,
        ref_template=opts.reference,
        omp_nthreads=opts.omp_nthreads,
        normalization_quality='precise' if not opts.testing else 'testing',
        fs_subjects_dir=opts.freesurfer,
    )
    tf.base_dir = str(opts.work_dir.resolve())
    tf.run(**plugin_settings)
