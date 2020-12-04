"""Workflow for the registration of EPI datasets to anatomical space via reconstructed surfaces."""
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype import logging

LOGGER = logging.getLogger("workflow")


def init_bbreg_wf(
    *,
    omp_nthreads,
    debug=False,
    epi2t1w_init="register",
    epi2t1w_dof=6,
    name="bbreg_wf",
    use_bbr=None,
):
    """
    Build a workflow to run FreeSurfer's ``bbregister``.

    This workflow uses FreeSurfer's ``bbregister`` to register a EPI image to
    a T1-weighted structural image.
    It is a counterpart to :py:func:`~fmriprep.workflows.bold.registration.init_fsl_bbr_wf`,
    which performs the same task using FSL's FLIRT with a BBR cost function.
    The ``use_bbr`` option permits a high degree of control over registration.
    If ``False``, standard, affine coregistration will be performed using
    FreeSurfer's ``mri_coreg`` tool.
    If ``True``, ``bbregister`` will be seeded with the initial transform found
    by ``mri_coreg`` (equivalent to running ``bbregister --init-coreg``).
    If ``None``, after ``bbregister`` is run, the resulting affine transform
    will be compared to the initial transform found by ``mri_coreg``.
    Excessive deviation will result in rejecting the BBR refinement and
    accepting the original, affine registration.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from niworkflows.anat.coregistration import init_bbreg_wf
            wf = init_bbreg_wf(omp_nthreads=1)

    Parameters
    ----------
    use_bbr : :obj:`bool` or None
        Enable/disable boundary-based registration refinement.
        If ``None``, test BBR result for distortion before accepting.
    epi2t1w_dof : 6, 9 or 12
        Degrees-of-freedom for EPI-T1w registration
    epi2t1w_init : :obj:`str`, ``"header"`` or ``"register"``
        If ``"header"``, use header information for initialization of EPI and T1 images.
        If ``"register"``, align volumes by their centers.
    name : :obj:`str`, optional
        Workflow name (default: ``bbreg_wf``)

    Inputs
    ------
    in_file
        Reference EPI image to be registered
    fsnative2t1w_xfm
        FSL-style affine matrix translating from FreeSurfer T1.mgz to T1w
    subjects_dir
        Sets FreeSurfer's ``$SUBJECTS_DIR``
    subject_id
        FreeSurfer subject ID (must have a corresponding folder in ``$SUBJECTS_DIR``)
    t1w_brain
        Unused (see :py:func:`~fmriprep.workflows.bold.registration.init_fsl_bbr_wf`)
    t1w_dseg
        Unused (see :py:func:`~fmriprep.workflows.bold.registration.init_fsl_bbr_wf`)

    Outputs
    -------
    itk_epi_to_t1w
        Affine transform from the reference EPI to T1w space (ITK format)
    itk_t1w_to_epi
        Affine transform from T1w space to EPI space (ITK format)
    out_report
        Reportlet for assessing registration quality
    fallback
        Boolean indicating whether BBR was rejected (mri_coreg registration returned)

    """
    from ..engine.workflows import LiterateWorkflow as Workflow

    # See https://github.com/nipreps/fmriprep/issues/768
    from ..interfaces.freesurfer import (
        PatchedBBRegisterRPT as BBRegisterRPT,
        PatchedMRICoregRPT as MRICoregRPT,
        PatchedLTAConvert as LTAConvert,
    )
    from ..interfaces.nitransforms import ConcatenateXFMs

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
The EPI reference was then co-registered to the T1w reference using
`bbregister` (FreeSurfer) which implements boundary-based registration [@bbr].
Co-registration was configured with {dof} degrees of freedom{reason}.
""".format(
        dof={6: "six", 9: "nine", 12: "twelve"}[epi2t1w_dof],
        reason=""
        if epi2t1w_dof == 6
        else "to account for distortions remaining in the EPI reference",
    )

    inputnode = pe.Node(
        niu.IdentityInterface(
            [
                "in_file",
                "fsnative2t1w_xfm",
                "subjects_dir",
                "subject_id",  # BBRegister
                "t1w_dseg",  # FLIRT BBR
                "t1w_brain",  # FLIRT BBR
            ]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            ["itk_epi_to_t1w", "itk_t1w_to_epi", "out_report", "fallback"]
        ),
        name="outputnode",
    )

    if epi2t1w_init not in ("register", "header"):
        raise ValueError(f"Unknown EPI-T1w initialization option: {epi2t1w_init}")

    # For now make BBR unconditional - in the future, we can fall back to identity,
    # but adding the flexibility without testing seems a bit dangerous
    if epi2t1w_init == "header":
        if use_bbr is False:
            raise ValueError("Cannot disable BBR and use header registration")
        if use_bbr is None:
            LOGGER.warning("Initializing BBR with header; affine fallback disabled")
            use_bbr = True

    merge_ltas = pe.Node(niu.Merge(2), name="merge_ltas", run_without_submitting=True)
    concat_xfm = pe.Node(ConcatenateXFMs(inverse=True), name="concat_xfm")

    # fmt:off
    workflow.connect([
        # Output ITK transforms
        (inputnode, merge_ltas, [("fsnative2t1w_xfm", "in2")]),
        (merge_ltas, concat_xfm, [("out", "in_xfms")]),
        (concat_xfm, outputnode, [("out_xfm", "itk_epi_to_t1w")]),
        (concat_xfm, outputnode, [("out_inv", "itk_t1w_to_epi")]),
    ])
    # fmt:on

    if debug is True:
        from ..interfaces.images import RegridToZooms

        downsample = pe.Node(
            RegridToZooms(zooms=(4.0, 4.0, 4.0), smooth=True), name="downsample"
        )
        workflow.connect([(inputnode, downsample, [("in_file", "in_file")])])

    mri_coreg = pe.Node(
        MRICoregRPT(
            dof=epi2t1w_dof,
            sep=[4],
            ftol=0.0001,
            linmintol=0.01,
            generate_report=not use_bbr,
        ),
        name="mri_coreg",
        n_procs=omp_nthreads,
        mem_gb=5,
    )

    # Use mri_coreg
    if epi2t1w_init == "register":
        # fmt:off
        workflow.connect([
            (inputnode, mri_coreg, [("subjects_dir", "subjects_dir"),
                                    ("subject_id", "subject_id")]),
        ])
        # fmt:on

        if not debug:
            workflow.connect(inputnode, "in_file", mri_coreg, "source_file")
        else:
            workflow.connect(downsample, "out_file", mri_coreg, "source_file")

        # Short-circuit workflow building, use initial registration
        if use_bbr is False:
            # fmt:off
            workflow.connect([
                (mri_coreg, outputnode, [("out_report", "out_report")]),
                (mri_coreg, merge_ltas, [("out_lta_file", "in1")]),
            ])
            # fmt:on
            outputnode.inputs.fallback = True
            return workflow

    # Use bbregister
    bbregister = pe.Node(
        BBRegisterRPT(
            dof=epi2t1w_dof,
            contrast_type="t2",
            registered_file=True,
            out_lta_file=True,
            generate_report=True,
        ),
        name="bbregister",
        mem_gb=12,
    )

    # fmt:off
    workflow.connect([
        (inputnode, bbregister, [("subjects_dir", "subjects_dir"),
                                 ("subject_id", "subject_id")]),
    ])
    # fmt:on

    if not debug:
        workflow.connect(inputnode, "in_file", bbregister, "source_file")
    else:
        workflow.connect(downsample, "out_file", bbregister, "source_file")

    if epi2t1w_init == "header":
        bbregister.inputs.init = "header"
    else:
        workflow.connect([(mri_coreg, bbregister, [("out_lta_file", "init_reg_file")])])

    # Short-circuit workflow building, use boundary-based registration
    if use_bbr is True:
        # fmt:off
        workflow.connect([
            (bbregister, outputnode, [("out_report", "out_report")]),
            (bbregister, merge_ltas, [("out_lta_file", "in1")]),
        ])
        # fmt:on

        outputnode.inputs.fallback = False
        return workflow

    # Only reach this point if epi2t1w_init is "register" and use_bbr is None
    transforms = pe.Node(niu.Merge(2), run_without_submitting=True, name="transforms")
    reports = pe.Node(niu.Merge(2), run_without_submitting=True, name="reports")

    lta_ras2ras = pe.MapNode(
        LTAConvert(out_lta=True), iterfield=["in_lta"], name="lta_ras2ras", mem_gb=2
    )
    compare_transforms = pe.Node(
        niu.Function(function=compare_xforms), name="compare_transforms"
    )

    select_transform = pe.Node(
        niu.Select(), run_without_submitting=True, name="select_transform"
    )
    select_report = pe.Node(
        niu.Select(), run_without_submitting=True, name="select_report"
    )

    # fmt:off
    workflow.connect([
        (bbregister, transforms, [("out_lta_file", "in1")]),
        (mri_coreg, transforms, [("out_lta_file", "in2")]),
        # Normalize LTA transforms to RAS2RAS (inputs are VOX2VOX) and compare
        (transforms, lta_ras2ras, [("out", "in_lta")]),
        (lta_ras2ras, compare_transforms, [("out_lta", "lta_list")]),
        (compare_transforms, outputnode, [("out", "fallback")]),
        # Select output transform
        (transforms, select_transform, [("out", "inlist")]),
        (compare_transforms, select_transform, [("out", "index")]),
        (select_transform, merge_ltas, [("out", "in1")]),
        # Select output report
        (bbregister, reports, [("out_report", "in1")]),
        (mri_coreg, reports, [("out_report", "in2")]),
        (reports, select_report, [("out", "inlist")]),
        (compare_transforms, select_report, [("out", "index")]),
        (select_report, outputnode, [("out", "out_report")]),
    ])
    # fmt:on

    return workflow


def compare_xforms(lta_list, norm_threshold=15):
    """
    Determine a distance between two affine transformations.

    Computes a normalized displacement between two affine transforms as the
    maximum overall displacement of the midpoints of the faces of a cube, when
    each transform is applied to the cube.
    This combines displacement resulting from scaling, translation and rotation.
    Although the norm is in mm, in a scaling context, it is not necessarily
    equivalent to that distance in translation.
    We choose a default threshold of 15mm as a rough heuristic.
    Normalized displacement above 20mm showed clear signs of distortion, while
    "good" BBR refinements were frequently below 10mm displaced from the rigid
    transform.
    The 10-20mm range was more ambiguous, and 15mm chosen as a compromise.
    This is open to revisiting in either direction.
    See discussion in
    `GitHub issue #681 <https://github.com/nipreps/fmriprep/issues/681>`__
    and the `underlying implementation
    <https://github.com/nipy/nipype/blob/56b7c81eedeeae884ba47c80096a5f66bd9f8116/nipype/algorithms/rapidart.py#L108-L159>`__.

    Parameters
    ----------
      lta_list : :obj:`list` or :obj:`tuple` of :obj:`str`
          the two given affines in LTA format
      norm_threshold : :obj:`float`
          the upper bound limit to the normalized displacement caused by the
          second transform relative to the first (default: `15`)

    """
    from niworkflows.interfaces.surf import load_transform
    from nipype.algorithms.rapidart import _calc_norm_affine

    bbr_affine = load_transform(lta_list[0])
    fallback_affine = load_transform(lta_list[1])

    norm, _ = _calc_norm_affine([fallback_affine, bbr_affine], use_differences=True)

    return norm[1] > norm_threshold
