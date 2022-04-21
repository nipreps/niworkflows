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
"""Utility workflows."""
from packaging.version import parse as parseversion, Version
from pkg_resources import resource_filename as pkgr_fn

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu, fsl, afni

from templateflow.api import get as get_template

from ..engine.workflows import LiterateWorkflow as Workflow
from ..interfaces.fixes import (
    FixHeaderRegistration as Registration,
    FixHeaderApplyTransforms as ApplyTransforms,
    FixN4BiasFieldCorrection as N4BiasFieldCorrection,
)
from ..interfaces.header import CopyXForm, ValidateImage, MatchHeader
from ..interfaces.reportlets.masks import SimpleShowMaskRPT
from ..utils.connections import listify
from ..utils.misc import pass_dummy_scans as _pass_dummy_scans


DEFAULT_MEMORY_MIN_GB = 0.01


def init_bold_reference_wf(
    omp_nthreads,
    bold_file=None,
    sbref_files=None,
    brainmask_thresh=0.85,
    pre_mask=False,
    multiecho=False,
    name="bold_reference_wf",
    gen_report=False,
):
    """
    Build a workflow that generates reference BOLD images for a series.

    The raw reference image is the target of :abbr:`HMC (head motion correction)`, and a
    contrast-enhanced reference is the subject of distortion correction, as well as
    boundary-based registration to T1w and template spaces.

    LIMITATION: If one wants to extract the reference from several SBRefs
    with several echoes each, the first echo should be selected elsewhere
    and run this interface in ``multiecho = False`` mode. In other words,
    SBRef inputs are assumed to be single-echo.

    LIMITATION: If a list of SBRefs is provided, each can be 3D or 4D, but they
    are assumed to be sampled in the exact same 3D-grid and have the same orientation
    information in their headers so that they can directly be merged into a 4D volume.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from niworkflows.func.util import init_bold_reference_wf
            wf = init_bold_reference_wf(omp_nthreads=1)

    Parameters
    ----------
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    bold_file : :obj:`str`
        BOLD series NIfTI file
    sbref_files : :obj:`list` or :obj:`bool`
        Single band (as opposed to multi band) reference NIfTI file.
        If ``True`` is passed, the workflow is built to accommodate SBRefs,
        but the input is left undefined (i.e., it is left open for connection)
    brainmask_thresh: :obj:`float`
        Lower threshold for the probabilistic brainmask to obtain
        the final binary mask (default: 0.85).
    pre_mask : :obj:`bool`
        Indicates whether the ``pre_mask`` input will be set (and thus, step 1
        should be skipped).
    multiecho : :obj:`bool`
        If multiecho data was supplied, data from the first echo will be selected
    name : :obj:`str`
        Name of workflow (default: ``bold_reference_wf``)
    gen_report : :obj:`bool`
        Whether a mask report node should be appended in the end

    Inputs
    ------
    bold_file : str
        BOLD series NIfTI file
    all_bold_files : str
        Validated and header-corrected BOLD series; multiple if multi-echo
    bold_mask : bool
        A tentative brain mask to initialize the workflow (requires ``pre_mask``
        parameter set ``True``).
    dummy_scans : int or None
        Number of non-steady-state volumes specified by user at beginning of ``bold_file``
    sbref_file : str
        single band (as opposed to multi band) reference NIfTI file

    Outputs
    -------
    bold_file : str
        Validated BOLD series NIfTI file
    raw_ref_image : str
        Reference image to which BOLD series is motion corrected
    skip_vols : int
        Number of non-steady-state volumes selected at beginning of ``bold_file``
    algo_dummy_scans : int
        Number of non-steady-state volumes agorithmically detected at
        beginning of ``bold_file``
    ref_image : str
        Contrast-enhanced reference image
    ref_image_brain : str
        Skull-stripped reference image
    bold_mask : str
        Skull-stripping mask of reference image
    validation_report : str
        HTML reportlet indicating whether ``bold_file`` had a valid affine


    Subworkflows
        * :py:func:`~niworkflows.func.util.init_enhance_and_skullstrip_wf`

    """
    from ..utils.connections import pop_file as _pop
    from ..interfaces.bold import NonsteadyStatesDetector
    from ..interfaces.images import RobustAverage

    workflow = Workflow(name=name)
    workflow.__desc__ = f"""\
First, a reference volume and its skull-stripped version were generated
{'from the shortest echo of the BOLD run' * multiecho} using a custom
methodology of *fMRIPrep*.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=["bold_file", "bold_mask", "dummy_scans", "sbref_file"]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_file",
                "all_bold_files",
                "raw_ref_image",
                "skip_vols",
                "algo_dummy_scans",
                "ref_image",
                "ref_image_brain",
                "bold_mask",
                "validation_report",
                "mask_report",
            ]
        ),
        name="outputnode",
    )

    # Simplify manually setting input image
    if bold_file is not None:
        inputnode.inputs.bold_file = bold_file

    val_bold = pe.MapNode(
        ValidateImage(),
        name="val_bold",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
        iterfield=["in_file"],
    )

    get_dummy = pe.Node(NonsteadyStatesDetector(), name="get_dummy")
    gen_avg = pe.Node(RobustAverage(), name="gen_avg", mem_gb=1)

    enhance_and_skullstrip_bold_wf = init_enhance_and_skullstrip_bold_wf(
        brainmask_thresh=brainmask_thresh,
        omp_nthreads=omp_nthreads,
        pre_mask=pre_mask,
    )

    calc_dummy_scans = pe.Node(
        niu.Function(function=_pass_dummy_scans, output_names=["skip_vols_num"]),
        name="calc_dummy_scans",
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    # fmt: off
    workflow.connect([
        (inputnode, val_bold, [(("bold_file", listify), "in_file")]),
        (inputnode, get_dummy, [(("bold_file", _pop), "in_file")]),
        (inputnode, enhance_and_skullstrip_bold_wf, [("bold_mask", "inputnode.pre_mask")]),
        (inputnode, calc_dummy_scans, [("dummy_scans", "dummy_scans")]),
        (gen_avg, enhance_and_skullstrip_bold_wf, [("out_file", "inputnode.in_file")]),
        (get_dummy, calc_dummy_scans, [("n_dummy", "algo_dummy_scans")]),
        (calc_dummy_scans, outputnode, [("skip_vols_num", "skip_vols")]),
        (gen_avg, outputnode, [("out_file", "raw_ref_image")]),
        (get_dummy, outputnode, [("n_dummy", "algo_dummy_scans")]),
        (val_bold, outputnode, [(("out_file", _pop), "bold_file"),
                                ("out_file", "all_bold_files"),
                                (("out_report", _pop), "validation_report")]),
        (enhance_and_skullstrip_bold_wf, outputnode, [
            ("outputnode.bias_corrected_file", "ref_image"),
            ("outputnode.mask_file", "bold_mask"),
            ("outputnode.skull_stripped_file", "ref_image_brain"),
        ]),
    ])
    # fmt: on

    if gen_report:
        mask_reportlet = pe.Node(SimpleShowMaskRPT(), name="mask_reportlet")
        # fmt: off
        workflow.connect([
            (enhance_and_skullstrip_bold_wf, mask_reportlet, [
                ("outputnode.bias_corrected_file", "background_file"),
                ("outputnode.mask_file", "mask_file"),
            ]),
        ])
        # fmt: on

    if not sbref_files:
        # fmt: off
        workflow.connect([
            (val_bold, gen_avg, [(("out_file", _pop), "in_file")]),  # pop first echo of ME-EPI
            (get_dummy, gen_avg, [("t_mask", "t_mask")]),
        ])
        # fmt: on
        return workflow

    from ..interfaces.nibabel import MergeSeries

    nsbrefs = 0
    if sbref_files is not True:
        # If not boolean, then it is a list-of or pathlike.
        inputnode.inputs.sbref_file = sbref_files
        nsbrefs = 1 if isinstance(sbref_files, str) else len(sbref_files)

    val_sbref = pe.MapNode(
        ValidateImage(),
        name="val_sbref",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
        iterfield=["in_file"],
    )
    merge_sbrefs = pe.Node(MergeSeries(), name="merge_sbrefs")

    # fmt: off
    workflow.connect([
        (inputnode, val_sbref, [(("sbref_file", listify), "in_file")]),
        (val_sbref, merge_sbrefs, [("out_file", "in_files")]),
        (merge_sbrefs, gen_avg, [("out_file", "in_file")]),
    ])
    # fmt: on

    # Edit the boilerplate as the SBRef will be the reference
    workflow.__desc__ = f"""\
First, a reference volume and its skull-stripped version were generated
by aligning and averaging {nsbrefs or ''} single-band references (SBRefs).
"""

    return workflow


def init_enhance_and_skullstrip_bold_wf(
    brainmask_thresh=0.5,
    name="enhance_and_skullstrip_bold_wf",
    omp_nthreads=1,
    pre_mask=False,
):
    """
    Enhance and run brain extraction on a BOLD EPI image.

    This workflow takes in a :abbr:`BOLD (blood-oxygen level-dependant)`
    :abbr:`fMRI (functional MRI)` average/summary (e.g., a reference image
    averaging non-steady-state timepoints), and sharpens the histogram
    with the application of the N4 algorithm for removing the
    :abbr:`INU (intensity non-uniformity)` bias field and calculates a signal
    mask.

    Steps of this workflow are:

      1. Calculate a tentative mask by registering (9-parameters) to *fMRIPrep*'s
         :abbr:`EPI (echo-planar imaging)` -*boldref* template, which
         is in MNI space.
         The tentative mask is obtained by resampling the MNI template's
         brainmask into *boldref*-space.
      2. Binary dilation of the tentative mask with a sphere of 3mm diameter.
      3. Run ANTs' ``N4BiasFieldCorrection`` on the input
         :abbr:`BOLD (blood-oxygen level-dependant)` average, using the
         mask generated in 1) instead of the internal Otsu thresholding.
      4. Calculate a loose mask using FSL's ``bet``, with one mathematical morphology
         dilation of one iteration and a sphere of 6mm as structuring element.
      5. Mask the :abbr:`INU (intensity non-uniformity)`-corrected image
         with the latest mask calculated in 3), then use AFNI's ``3dUnifize``
         to *standardize* the T2* contrast distribution.
      6. Calculate a mask using AFNI's ``3dAutomask`` after the contrast
         enhancement of 4).
      7. Calculate a final mask as the intersection of 4) and 6).
      8. Apply final mask on the enhanced reference.

    Step 1 can be skipped if the ``pre_mask`` argument is set to ``True`` and
    a tentative mask is passed in to the workflow throught the ``pre_mask``
    Nipype input.


    Workflow graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from niworkflows.func.util import init_enhance_and_skullstrip_bold_wf
            wf = init_enhance_and_skullstrip_bold_wf(omp_nthreads=1)

    .. _N4BiasFieldCorrection: https://hdl.handle.net/10380/3053

    Parameters
    ----------
    brainmask_thresh: :obj:`float`
        Lower threshold for the probabilistic brainmask to obtain
        the final binary mask (default: 0.5).
    name : str
        Name of workflow (default: ``enhance_and_skullstrip_bold_wf``)
    omp_nthreads : int
        number of threads available to parallel nodes
    pre_mask : bool
        Indicates whether the ``pre_mask`` input will be set (and thus, step 1
        should be skipped).

    Inputs
    ------
    in_file : str
        BOLD image (single volume)
    pre_mask : bool
        A tentative brain mask to initialize the workflow (requires ``pre_mask``
        parameter set ``True``).


    Outputs
    -------
    bias_corrected_file : str
        the ``in_file`` after `N4BiasFieldCorrection`_
    skull_stripped_file : str
        the ``bias_corrected_file`` after skull-stripping
    mask_file : str
        mask of the skull-stripped input file
    out_report : str
        reportlet for the skull-stripping

    """
    from niworkflows.interfaces.nibabel import ApplyMask, BinaryDilation

    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=["in_file", "pre_mask"]), name="inputnode"
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["mask_file", "skull_stripped_file", "bias_corrected_file"]
        ),
        name="outputnode",
    )

    # Run N4 normally, force num_threads=1 for stability (images are small, no need for >1)
    n4_correct = pe.Node(
        N4BiasFieldCorrection(
            dimension=3, copy_header=True, bspline_fitting_distance=200
        ),
        shrink_factor=2,
        name="n4_correct",
        n_procs=1,
    )
    n4_correct.inputs.rescale_intensities = True

    # Create a generous BET mask out of the bias-corrected EPI
    skullstrip_first_pass = pe.Node(
        fsl.BET(frac=0.2, mask=True), name="skullstrip_first_pass"
    )
    first_dilate = pe.Node(BinaryDilation(radius=6), name="first_dilate")
    first_mask = pe.Node(ApplyMask(), name="first_mask")

    # Use AFNI's unifize for T2 constrast & fix header
    unifize = pe.Node(
        afni.Unifize(
            t2=True,
            outputtype="NIFTI_GZ",
            # Default -clfrac is 0.1, 0.4 was too conservative
            # -rbt because I'm a Jedi AFNI Master (see 3dUnifize's documentation)
            args="-clfrac 0.2 -rbt 18.3 65.0 90.0",
            out_file="uni.nii.gz",
        ),
        name="unifize",
    )
    fixhdr_unifize = pe.Node(CopyXForm(), name="fixhdr_unifize", mem_gb=0.1)

    # Run ANFI's 3dAutomask to extract a refined brain mask
    skullstrip_second_pass = pe.Node(
        afni.Automask(dilate=1, outputtype="NIFTI_GZ"), name="skullstrip_second_pass"
    )
    fixhdr_skullstrip2 = pe.Node(CopyXForm(), name="fixhdr_skullstrip2", mem_gb=0.1)

    # Take intersection of both masks
    combine_masks = pe.Node(fsl.BinaryMaths(operation="mul"), name="combine_masks")

    # Compute masked brain
    apply_mask = pe.Node(ApplyMask(), name="apply_mask")

    if not pre_mask:
        from nipype.interfaces.ants.utils import AI

        bold_template = get_template(
            "MNI152NLin2009cAsym", resolution=2, desc="fMRIPrep", suffix="boldref"
        )
        brain_mask = get_template(
            "MNI152NLin2009cAsym", resolution=2, desc="brain", suffix="mask"
        )

        # Initialize transforms with antsAI
        init_aff = pe.Node(
            AI(
                fixed_image=str(bold_template),
                fixed_image_mask=str(brain_mask),
                metric=("Mattes", 32, "Regular", 0.2),
                transform=("Affine", 0.1),
                search_factor=(20, 0.12),
                principal_axes=False,
                convergence=(10, 1e-6, 10),
                verbose=True,
            ),
            name="init_aff",
            n_procs=omp_nthreads,
        )

        # Registration().version may be None
        if parseversion(Registration().version or "0.0.0") > Version("2.2.0"):
            init_aff.inputs.search_grid = (40, (0, 40, 40))

        # Set up spatial normalization
        norm = pe.Node(
            Registration(
                from_file=pkgr_fn("niworkflows.data", "epi_atlasbased_brainmask.json")
            ),
            name="norm",
            n_procs=omp_nthreads,
        )
        norm.inputs.fixed_image = str(bold_template)
        map_brainmask = pe.Node(
            ApplyTransforms(
                interpolation="Linear",
                # Use the higher resolution and probseg for numerical stability in rounding
                input_image=str(
                    get_template(
                        "MNI152NLin2009cAsym",
                        resolution=1,
                        label="brain",
                        suffix="probseg",
                    )
                ),
            ),
            name="map_brainmask",
        )
        # Ensure mask's header matches reference's
        check_hdr = pe.Node(MatchHeader(), name="check_hdr", run_without_submitting=True)

        # fmt: off
        workflow.connect([
            (inputnode, check_hdr, [("in_file", "reference")]),
            (inputnode, init_aff, [("in_file", "moving_image")]),
            (inputnode, map_brainmask, [("in_file", "reference_image")]),
            (inputnode, norm, [("in_file", "moving_image")]),
            (init_aff, norm, [("output_transform", "initial_moving_transform")]),
            (norm, map_brainmask, [
                ("reverse_invert_flags", "invert_transform_flags"),
                ("reverse_transforms", "transforms"),
            ]),
            (map_brainmask, check_hdr, [("output_image", "in_file")]),
            (check_hdr, n4_correct, [("out_file", "weight_image")]),
        ])
        # fmt: on
    else:
        # fmt: off
        workflow.connect([
            (inputnode, n4_correct, [("pre_mask", "weight_image")]),
        ])
        # fmt: on

    # fmt: off
    workflow.connect([
        (inputnode, n4_correct, [("in_file", "input_image")]),
        (inputnode, fixhdr_unifize, [("in_file", "hdr_file")]),
        (inputnode, fixhdr_skullstrip2, [("in_file", "hdr_file")]),
        (n4_correct, skullstrip_first_pass, [("output_image", "in_file")]),
        (skullstrip_first_pass, first_dilate, [("mask_file", "in_file")]),
        (first_dilate, first_mask, [("out_file", "in_mask")]),
        (skullstrip_first_pass, first_mask, [("out_file", "in_file")]),
        (first_mask, unifize, [("out_file", "in_file")]),
        (unifize, fixhdr_unifize, [("out_file", "in_file")]),
        (fixhdr_unifize, skullstrip_second_pass, [("out_file", "in_file")]),
        (skullstrip_first_pass, combine_masks, [("mask_file", "in_file")]),
        (skullstrip_second_pass, fixhdr_skullstrip2, [("out_file", "in_file")]),
        (fixhdr_skullstrip2, combine_masks, [("out_file", "operand_file")]),
        (fixhdr_unifize, apply_mask, [("out_file", "in_file")]),
        (combine_masks, apply_mask, [("out_file", "in_mask")]),
        (combine_masks, outputnode, [("out_file", "mask_file")]),
        (apply_mask, outputnode, [("out_file", "skull_stripped_file")]),
        (n4_correct, outputnode, [("output_image", "bias_corrected_file")]),
    ])
    # fmt: on

    return workflow


def init_skullstrip_bold_wf(name="skullstrip_bold_wf"):
    """
    Apply skull-stripping to a BOLD image.

    It is intended to be used on an image that has previously been
    bias-corrected with
    :py:func:`~niworkflows.func.util.init_enhance_and_skullstrip_bold_wf`

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from niworkflows.func.util import init_skullstrip_bold_wf
            wf = init_skullstrip_bold_wf()


    Inputs
    ------
    in_file : str
        BOLD image (single volume)

    Outputs
    -------
    skull_stripped_file : str
        the ``in_file`` after skull-stripping
    mask_file : str
        mask of the skull-stripped input file
    out_report : str
        reportlet for the skull-stripping

    """
    from niworkflows.interfaces.nibabel import ApplyMask

    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=["in_file"]), name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["mask_file", "skull_stripped_file", "out_report"]
        ),
        name="outputnode",
    )
    skullstrip_first_pass = pe.Node(
        fsl.BET(frac=0.2, mask=True), name="skullstrip_first_pass"
    )
    skullstrip_second_pass = pe.Node(
        afni.Automask(dilate=1, outputtype="NIFTI_GZ"), name="skullstrip_second_pass"
    )
    combine_masks = pe.Node(fsl.BinaryMaths(operation="mul"), name="combine_masks")
    apply_mask = pe.Node(ApplyMask(), name="apply_mask")
    mask_reportlet = pe.Node(SimpleShowMaskRPT(), name="mask_reportlet")

    # fmt: off
    workflow.connect([
        (inputnode, skullstrip_first_pass, [("in_file", "in_file")]),
        (skullstrip_first_pass, skullstrip_second_pass, [("out_file", "in_file")]),
        (skullstrip_first_pass, combine_masks, [("mask_file", "in_file")]),
        (skullstrip_second_pass, combine_masks, [("out_file", "operand_file")]),
        (combine_masks, outputnode, [("out_file", "mask_file")]),
        # Masked file
        (inputnode, apply_mask, [("in_file", "in_file")]),
        (combine_masks, apply_mask, [("out_file", "in_mask")]),
        (apply_mask, outputnode, [("out_file", "skull_stripped_file")]),
        # Reportlet
        (inputnode, mask_reportlet, [("in_file", "background_file")]),
        (combine_masks, mask_reportlet, [("out_file", "mask_file")]),
        (mask_reportlet, outputnode, [("out_report", "out_report")]),
    ])
    # fmt: on

    return workflow
