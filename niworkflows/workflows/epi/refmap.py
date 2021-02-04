# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflow for the generation of EPI (echo-planar imaging) references."""

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from ...engine.workflows import LiterateWorkflow as Workflow


DEFAULT_MEMORY_MIN_GB = 0.01


def init_func_reference_wf(omp_nthreads, name="func_reference_wf"):
    """


    Inputs
    ------
    in_files : :obj:`list` of :obj:`str`
        List of paths of the input EPI images from which reference volumes will be
        selected, aligned and averaged.
        IMPORTANT: All input files MUST have the same shimming configuration.
        At the very least, make sure all input EPI images are acquired within the
        same session, and have the same PE direction and total readout time.

    Outputs
    -------
    out_epiref : :obj:`str`
        Path of the generated EPI reference file.

    See Also
    --------
    Discussion and original flowchart at `nipreps/niworkflows#601
    <https://github.com/nipreps/niworkflows/issues/601>`__.

    """
    from ...utils.connections import listify

    from ...interfaces.bold import NonsteadyStatesDetector
    from ...interfaces.fixes import (
        FixN4BiasFieldCorrection as N4BiasFieldCorrection,
    )
    from ...interfaces.freesurfer import StructuralReference
    from ...interfaces.images import ValidateImage, RobustAverage
    from ...interfaces.nibabel import IntensityClip

    wf = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=["in_files"]), name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["epiref", "xfms", "volumes"]), name="outputnode"
    )

    validate_nii = pe.MapNode(
        ValidateImage(), name="validate_nii", iterfield=["in_file"]
    )

    # TODO: allow b=0 selection for DWI datasets instead
    select_volumes = pe.MapNode(
        NonsteadyStatesDetector(), name="select_volumes", iterfield=["in_file"]
    )
    run_avgs = pe.MapNode(
        RobustAverage(), name="run_avgs", mem_gb=1, iterfield=["in_file"]
    )

    clip_avgs = pe.MapNode(IntensityClip(), name="clip_avgs", iterfield=["in_file"])

    # de-gradient the fields ("bias/illumination artifact")
    n4_avgs = pe.MapNode(
        N4BiasFieldCorrection(
            dimension=3,
            copy_header=True,
            n_iterations=[50] * 5,
            convergence_threshold=1e-7,
            shrink_factor=4,
        ),
        n_procs=omp_nthreads,
        name="n4_avgs",
    )

    epi_merge = pe.Node(
        StructuralReference(
            auto_detect_sensitivity=True,
            initial_timepoint=1,  # For deterministic behavior
            intensity_scaling=True,  # 7-DOF (rigid + intensity)
            subsample_threshold=200,
            fixed_timepoint=True,
            no_iteration=True,
            transform_outputs=True,
        ),
        name="epi_merge",
    )

    # fmt:off
    wf.connect([
        (inputnode, validate_nii, [(("in_files", listify), "in_file")]),
        (validate_nii, select_volumes, [("out_file", "in_file")]),
        (validate_nii, run_avgs, [("out_file", "in_file")]),
        (select_volumes, run_avgs, [("t_mask", "t_mask")]),
        (run_avgs, clip_avgs, [("out_file", "in_file")]),
        (clip_avgs, n4_avgs, [("out_file", "input_image")]),
        (n4_avgs, epi_merge, [("output_image", "in_files")]),
        (epi_merge, outputnode, [("out_file", "epiref"),
                                 ("transform_outputs", "xfms")]),
        (n4_avgs, outputnode, [("output_image", "volumes")]),

    ])
    # fmt:on

    return wf


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
