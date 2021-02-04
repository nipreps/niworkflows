# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflow for the generation of EPI (echo-planar imaging) references."""

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from ...engine.workflows import LiterateWorkflow as Workflow
from ...interfaces.fixes import (
    FixHeaderRegistration as Registration,
    FixHeaderApplyTransforms as ApplyTransforms,
    FixN4BiasFieldCorrection as N4BiasFieldCorrection,
)
from ...interfaces.images import ValidateImage, EstimateReferenceImage


DEFAULT_MEMORY_MIN_GB = 0.01


def init_func_reference_wf(bold_files, omp_nthreads, multiecho=False, name='func_reference_wf'):
    """
    IN: list of bold files with matching PE/readout.
    OUT: reference images
    """

    wf = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['t1w_preproc', 't1w_mask']), name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['bold_masked', 'reference_images']), name='outputnode',
    )

    val_img = pe.MapNode(ValidateImage(), name='val_img', iterfield=['in_file'])

    est_ref = pe.MapNode(
        EstimateReferenceImage(multiecho=multiecho),
        name='est_ref',
        iterfield=['in_file']
    )

    n4_ref = pe.MapNode(
        N4BiasFieldCorrection(dimension=3, copy_header=True),
        name='n4_ref',
        iterfield=['in_file'],
        n_procs=omp_nthreads,
    )

    if len(bold_files) > 1:
        # inspect all BOLD references and get the one with the highest SNR
        get_best_snr = pe.Node(
            niu.Function(function=_max_snr, output_names=["reference", "out_files", "fidx"]),
            name='get_best_snr',
        )

        # register other reference images to best SNR
        norm_ref = pe.MapNode(
            Registration(),
            name='norm_ref',
            iterfield=['in_file'],
            n_procs=omp_nthreads
        )
        apply_ref = pe.MapNode(
            ApplyTransforms(),
            name='apply_ref',
            iterfield=['transforms', 'invert_transform_flags']
        )

        # fmt:off
        wf.connect([
            (get_best_snr, norm_ref, [("out_files", "in_file")]),
            (get_best_snr, )
            (norm_ref, apply_ref, [
                ("reverse_transforms", "transforms"),
                ("reverse_invert_flags", "invert_transform_flags")]),
        ])
        # fmt:on

        # TODO: register with t1w mask + output
    else:
        # sole bold file
        pass


def _max_snr(in_files, ddof=0):
    """
    Quick and dirty assessment of a list of images' signal-to-noise ratio.
    This is largely inpired by scipy's deprecated ``signaltonoise`` function.
    https://github.com/scipy/scipy/issues/9097#issuecomment-409413907
    """
    import nibabel as nb
    import numpy as np

    m_snr = None
    filename = None

    for fl in in_files:
        data = nb.load(fl).get_fdata()
        snr = np.where(sd == 0, 0, data.mean() / data.std(ddof=ddof))
        if m_snr is None or snr > m_snr:
            m_snr = snr
            filename = fl

    if filename is None:
        raise RuntimeError("Could not calculate SNR.")

    # save location and remove future reference target from list
    file_idx = in_files.index(filename)
    in_files.remove(filename)

    return filename, in_files, file_idx
