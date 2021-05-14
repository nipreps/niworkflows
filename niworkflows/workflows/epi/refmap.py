# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Workflow for the generation of EPI (echo-planar imaging) references."""

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from ...engine.workflows import LiterateWorkflow as Workflow


DEFAULT_MEMORY_MIN_GB = 0.01


def init_epi_reference_wf(
    omp_nthreads,
    auto_bold_nss=False,
    name="epi_reference_wf",
):
    """
    Build a workflow that generates a reference map from a set of EPI images.

    .. danger ::

        All input files MUST have the same shimming configuration.
        At the very least, make sure all input EPI images are acquired within the
        same session, and have the same PE direction and total readout time.

    Inputs to this workflow might be a list of :abbr:`SBRefs (single-band references)`,
    a list of fieldmapping :abbr:`EPIs (echo-planar images)`, a list of
    :abbr:`BOLD (blood-oxygen level-dependent)` images, or a list of
    :abbr:`DWI (diffusion-weighted imaging)` datasets.
    Please note that these different modalities should not be mixed together in any
    case for this particular workflow.

    For BOLD datasets, the workflow may be set up to execute an algorithm that determines
    the nonsteady states in the beginning of the timeseries (also called *dummy scans*),
    and uses those for generating a reference of the particular run, since the nonsteady
    states are known to yield better T1 contrast (and hence perhaps better signal for
    image registration).

    Relatedly, the workflow also provides a global signal drift estimation per run.
    This global signal drift is typically interesting for DWIs: because *b=0*
    volumes are typically scattered throughout the scan, this drift can be
    fit an exponential decay to model the signal drop caused by the increasing
    temperature of the device (this is closely related to BOLD *nonsteady states*
    described above, as these are just the few initial instants when the exponential
    decay is much faster).

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from niworkflows.workflows.epi.refmap import init_epi_reference_wf
            wf = init_epi_reference_wf(omp_nthreads=1)

    Parameters
    ----------
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    name : :obj:`str`
        Name of workflow (default: ``epi_reference_wf``)
    auto_bold_nss : :obj:`bool`
        If ``True``, determines nonsteady states in the beginning of the timeseries
        and selects them for the averaging of each run.
        IMPORTANT: this option applies only to BOLD EPIs.

    Inputs
    ------
    in_files : :obj:`list` of :obj:`str`
        List of paths of the input EPI images from which reference volumes will be
        selected, aligned and averaged.

    Outputs
    -------
    epi_ref_file : :obj:`str`
        Path of the generated EPI reference file.
    xfm_files : :obj:`list` of :obj:`str`
        List of rigid-body transforms in LTA format to resample from
        the reference volume of each run into the ``epi_ref_file`` reference.
    per_run_ref_files : :obj:`list` of :obj:`str`
        List of paths to the reference volume generated per input run.
    drift_factors : :obj:`list` of :obj:`list` of :obj:`float`
        A list of global signal drift factors for the set of volumes selected
        for averaging, per run.
    n_dummy_scans : :obj:`list` of :obj:`int`
        Number of nonsteady states at the beginning of each run (only BOLD with
        ``auto_bold_nss=True``)
    validate_report : :obj:`str`
        HTML reportlet(s) indicating whether the input files had a valid affine

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

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["in_files", "t_masks"]), name="inputnode"
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "epi_ref_file",
                "xfm_files",
                "per_run_ref_files",
                "drift_factors",
                "n_dummy",
                "validation_report",
            ]
        ),
        name="outputnode",
    )

    validate_nii = pe.MapNode(
        ValidateImage(), name="validate_nii", iterfield=["in_file"]
    )

    per_run_avgs = pe.MapNode(
        RobustAverage(), name="per_run_avgs", mem_gb=1, iterfield=["in_file", "t_mask"]
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
    clip_bg_noise = pe.MapNode(
        IntensityClip(p_min=2.0, p_max=100.0),
        name="clip_bg_noise",
        iterfield=["in_file"],
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

    post_merge = pe.Node(niu.Function(function=_post_merge), name="post_merge")

    def _set_threads(in_list, maximum):
        return min(len(in_list), maximum)

    # fmt:off
    wf.connect([
        (inputnode, validate_nii, [(("in_files", listify), "in_file")]),
        (validate_nii, per_run_avgs, [("out_file", "in_file")]),
        (per_run_avgs, clip_avgs, [("out_file", "in_file")]),
        (clip_avgs, n4_avgs, [("out_file", "input_image")]),
        (n4_avgs, clip_bg_noise, [("output_image", "in_file")]),
        (clip_bg_noise, epi_merge, [
            ("out_file", "in_files"),
            (("out_file", _set_threads, omp_nthreads), "num_threads"),
        ]),
        (epi_merge, post_merge, [("out_file", "in_file"),
                                 ("transform_outputs", "in_xfms")]),
        (post_merge, outputnode, [("out", "epi_ref_file")]),
        (epi_merge, outputnode, [("transform_outputs", "xfm_files")]),
        (per_run_avgs, outputnode, [("out_drift", "drift_factors")]),
        (n4_avgs, outputnode, [("output_image", "per_run_ref_files")]),
        (validate_nii, outputnode, [("out_report", "validation_report")]),
    ])
    # fmt:on

    if auto_bold_nss:
        select_volumes = pe.MapNode(
            NonsteadyStatesDetector(), name="select_volumes", iterfield=["in_file"]
        )
        # fmt:off
        wf.connect([
            (validate_nii, select_volumes, [("out_file", "in_file")]),
            (select_volumes, per_run_avgs, [("t_mask", "t_mask")]),
            (select_volumes, outputnode, [("n_dummy", "n_dummy")])
        ])
        # fmt:on
    else:
        wf.connect(inputnode, "t_masks", per_run_avgs, "t_mask")

    return wf


def _post_merge(in_file, in_xfms):
    """
    Massage output from ``SpatialReference``.

    If the previous ``SpatialReference`` node by-passed the execution of
    ``mri_robust_template`` (hence, there was only one input file), the
    single-file is forwarded to the output.

    Otherwise (``mri_robust_template`` was indeed executed), the output
    is converted from mgz to NIfTI and the datatype of the reference
    normalized to int16, with an intensity range of 0-255 (ideal for
    ANTs registrations)

    """
    from niworkflows.utils.connections import listify

    in_xfms = listify(in_xfms)
    if len(in_xfms) == 1 and in_file.endswith((".nii", ".nii.gz")):
        return in_file

    if len(in_xfms) == 1:
        raise RuntimeError("Output format and number of transforms do not match")

    from pathlib import Path
    import nibabel as nb
    from niworkflows.interfaces.nibabel import _advanced_clip

    out_file = Path() / Path(in_file).name.replace(".mgz", ".nii.gz")
    img = nb.load(in_file)
    nb.Nifti1Image(img.dataobj, img.affine, None).to_filename(out_file)
    return _advanced_clip(out_file, p_min=0.0, p_max=100.0)
