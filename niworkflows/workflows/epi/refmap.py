# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflow for the generation of EPI (echo-planar imaging) references."""

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from ...engine.workflows import LiterateWorkflow as Workflow


DEFAULT_MEMORY_MIN_GB = 0.01


def init_epi_reference_wf(omp_nthreads, name="epi_reference_wf"):
    """
    Build a workflow that generates a reference map from a set of EPI images.

    .. danger ::

        All input files MUST have the same shimming configuration.
        At the very least, make sure all input EPI images are acquired within the
        same session, and have the same PE direction and total readout time.

    Parameters
    ----------
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    name : :obj:`str`
        Name of workflow (default: ``epi_reference_wf``)

    Inputs
    ------
    in_files : :obj:`list` of :obj:`str`
        List of paths of the input EPI images from which reference volumes will be
        selected, aligned and averaged.

    Outputs
    -------
    out_epiref : :obj:`str`
        Path of the generated EPI reference file.

    See Also
    --------
    Discussion and original flowchart at `nipreps/niworkflows#601
    <https://github.com/nipreps/niworkflows/issues/601>`__.

    """
    from nipype.interfaces.ants import N4BiasFieldCorrection
    from ...utils.connections import listify

    from ...interfaces.bold import NonsteadyStatesDetector
    from ...interfaces.freesurfer import StructuralReference
    from ...interfaces.header import ValidateImage
    from ...interfaces.images import RobustAverage
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
        RobustAverage(), name="run_avgs", mem_gb=1, iterfield=["in_file", "t_mask"]
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
        iterfield=["input_image"],
    )
    clipper_post = pe.MapNode(
        IntensityClip(p_max=100.0), name="clipper_post", iterfield=["in_file"]
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

    tonii = pe.Node(niu.Function(function=_tonii), name="tonii")

    def _set_threads(in_list, maximum):
        return min(len(in_list), maximum)

    # fmt:off
    wf.connect([
        (inputnode, validate_nii, [(("in_files", listify), "in_file")]),
        (validate_nii, select_volumes, [("out_file", "in_file")]),
        (validate_nii, run_avgs, [("out_file", "in_file")]),
        (select_volumes, run_avgs, [("t_mask", "t_mask")]),
        (run_avgs, clip_avgs, [("out_file", "in_file")]),
        (clip_avgs, n4_avgs, [("out_file", "input_image")]),
        (n4_avgs, clipper_post, [("output_image", "in_file")]),
        (clipper_post, epi_merge, [
            ("out_file", "in_files"),
            (("out_file", _set_threads, omp_nthreads), "num_threads"),
        ]),
        (epi_merge, tonii, [("out_file", "in_file")]),
        (tonii, outputnode, [("out", "epiref")]),
        (epi_merge, outputnode, [("transform_outputs", "xfms")]),
        (n4_avgs, outputnode, [("output_image", "volumes")]),

    ])
    # fmt:on

    return wf


def _tonii(in_file):
    if in_file.endswith((".nii", ".nii.gz")):
        return in_file

    import nibabel as nb
    from pathlib import Path
    out_file = Path() / Path(in_file).name.replace(".mgz", ".nii.gz")
    img = nb.load(in_file)
    nb.Nifti1Image(img.dataobj, img.affine, None).to_filename(out_file)
    return str(out_file.absolute())
