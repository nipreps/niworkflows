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
"""Nipype translation of ANTs' workflows."""

# general purpose
from collections import OrderedDict
from multiprocessing import cpu_count
from pkg_resources import resource_filename as pkgr_fn
from warnings import warn

# nipype
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces.ants import (
    AI,
    Atropos,
    ImageMath,
    MultiplyImages,
    N4BiasFieldCorrection,
    ThresholdImage,
)

from ..utils.misc import get_template_specs
from ..utils.connections import pop_file as _pop

# niworkflows
from ..interfaces.fixes import (
    FixHeaderRegistration as Registration,
    FixHeaderApplyTransforms as ApplyTransforms,
)
from ..interfaces.nibabel import ApplyMask, RegridToZooms
from ..interfaces.header import CopyXForm


ATROPOS_MODELS = {
    "T1w": OrderedDict([("nclasses", 3), ("csf", 1), ("gm", 2), ("wm", 3)]),
    "T2w": OrderedDict([("nclasses", 3), ("csf", 3), ("gm", 2), ("wm", 1)]),
    "FLAIR": OrderedDict([("nclasses", 3), ("csf", 1), ("gm", 3), ("wm", 2)]),
}


def init_brain_extraction_wf(
    name="brain_extraction_wf",
    in_template="OASIS30ANTs",
    template_spec=None,
    use_float=True,
    normalization_quality="precise",
    omp_nthreads=None,
    mem_gb=3.0,
    bids_suffix="T1w",
    atropos_refine=True,
    atropos_use_random_seed=True,
    atropos_model=None,
    use_laplacian=True,
    bspline_fitting_distance=200,
):
    """
    Build a workflow for atlas-based brain extraction on anatomical MRI data.

    This is a Nipype implementation of atlas-based brain extraction inspired by
    the official ANTs' ``antsBrainExtraction.sh`` workflow (only for 3D images).

    The workflow follows the following structure:

      1. Step 1 performs several clerical tasks (preliminary INU correction,
         calculating the Laplacian of inputs, affine initialization) and the
         core spatial normalization.
      2. Maps the brain mask into target space using the normalization
         calculated in 1.
      3. Superstep 1b: binarization of the brain mask
      4. Maps the WM (white matter) probability map from the template, if such prior exists.
         Combines the BS (brainstem) probability map before mapping if the WM
         and BS are given separately (as it is the case for ``OASIS30ANTs``.)
      5. Run a second N4 INU correction round, using the prior mapped into
         individual step in step 4 if available.
      6. Superstep 6: apply ATROPOS on the INU-corrected result of step 5, and
         massage its outputs
      7. Superstep 7: use results from 4 to refine the brain mask
      8. If exist, use priors from step 4, calculate the overlap of the posteriors
         estimated in step 4 to select that overlapping the most with the WM+BS
         prior from the template. Combine that posterior with the refined brain
         mask and pass it on to the next step.
      9. Apply a final N4 using the refined brain mask (or the map calculated in
         step 8 if priors were found) as weights map for the algorithm.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from niworkflows.anat.ants import init_brain_extraction_wf
            wf = init_brain_extraction_wf()

    Parameters
    ----------
    in_template : str
        Name of the skull-stripping template ('OASIS30ANTs', 'NKI', or
        path).
        The brain template from which regions will be projected
        Anatomical template created using e.g. LPBA40 data set with
        ``buildtemplateparallel.sh`` in ANTs.
        The workflow will automatically search for a brain probability
        mask created using e.g. LPBA40 data set which have brain masks
        defined, and warped to anatomical template and
        averaged resulting in a probability image.
    use_float : bool
        Whether single precision should be used
    normalization_quality : str
        Use more precise or faster registration parameters
        (default: ``precise``, other possible values: ``testing``)
    omp_nthreads : int
        Maximum number of threads an individual process may use
    mem_gb : float
        Estimated peak memory consumption of the most hungry nodes
        in the workflow
    bids_suffix : str
        Sequence type of the first input image. For a list of acceptable values
        see https://bids-specification.readthedocs.io/en/latest/\
04-modality-specific-files/01-magnetic-resonance-imaging-data.html#anatomy-imaging-data
    atropos_refine : bool
        Enables or disables the whole ATROPOS sub-workflow
    atropos_use_random_seed : bool
        Whether ATROPOS should generate a random seed based on the
        system's clock
    atropos_model : tuple or None
        Allows to specify a particular segmentation model, overwriting
        the defaults based on ``bids_suffix``
    use_laplacian : bool
        Enables or disables alignment of the Laplacian as an additional
        criterion for image registration quality (default: True)
    bspline_fitting_distance : float
        The size of the b-spline mesh grid elements, in mm (default: 200)
    name : str, optional
        Workflow name (default: antsBrainExtraction)

    Inputs
    ------
    in_files : list
        List of input anatomical images to be brain-extracted,
        typically T1-weighted.
        If a list of anatomical images is provided, subsequently
        specified images are used during the segmentation process.
        However, only the first image is used in the registration
        of priors.
        Our suggestion would be to specify the T1w as the first image.
    in_mask : list, optional
        Mask used for registration to limit the metric
        computation to a specific region.

    Outputs
    -------
    out_file : str
        Skull-stripped and :abbr:`INU (intensity non-uniformity)`-corrected ``in_files``
    out_mask : str
        Calculated brain mask
    bias_corrected : str
        The ``in_files`` input images, after :abbr:`INU (intensity non-uniformity)`
        correction, before skull-stripping.
    bias_image : str
        The :abbr:`INU (intensity non-uniformity)` field estimated for each
        input in ``in_files``
    out_segm : str
        Output segmentation by ATROPOS
    out_tpms : str
        Output :abbr:`TPMs (tissue probability maps)` by ATROPOS

    """
    from packaging.version import parse as parseversion, Version
    from templateflow.api import get as get_template

    wf = pe.Workflow(name)

    template_spec = template_spec or {}

    # suffix passed via spec takes precedence
    template_spec["suffix"] = template_spec.get("suffix", bids_suffix)

    tpl_target_path, common_spec = get_template_specs(
        in_template, template_spec=template_spec
    )

    # Get probabilistic brain mask if available
    tpl_mask_path = get_template(
        in_template, label="brain", suffix="probseg", **common_spec
    ) or get_template(in_template, desc="brain", suffix="mask", **common_spec)

    if omp_nthreads is None or omp_nthreads < 1:
        omp_nthreads = cpu_count()

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["in_files", "in_mask"]), name="inputnode"
    )

    # Try to find a registration mask, set if available
    tpl_regmask_path = get_template(
        in_template, desc="BrainCerebellumExtraction", suffix="mask", **common_spec
    )
    if tpl_regmask_path:
        inputnode.inputs.in_mask = str(tpl_regmask_path)

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "out_file",
                "out_mask",
                "bias_corrected",
                "bias_image",
                "out_segm",
                "out_tpms",
            ]
        ),
        name="outputnode",
    )

    trunc = pe.MapNode(
        ImageMath(
            operation="TruncateImageIntensity", op2="0.01 0.999 256", copy_header=True
        ),
        name="truncate_images",
        iterfield=["op1"],
    )
    inu_n4 = pe.MapNode(
        N4BiasFieldCorrection(
            dimension=3,
            save_bias=False,
            copy_header=True,
            n_iterations=[50] * 4,
            convergence_threshold=1e-7,
            shrink_factor=4,
            bspline_fitting_distance=bspline_fitting_distance,
        ),
        n_procs=omp_nthreads,
        name="inu_n4",
        iterfield=["input_image"],
    )

    res_tmpl = pe.Node(
        RegridToZooms(in_file=tpl_target_path, zooms=(4, 4, 4), smooth=True),
        name="res_tmpl",
    )
    res_target = pe.Node(RegridToZooms(zooms=(4, 4, 4), smooth=True), name="res_target")

    lap_tmpl = pe.Node(
        ImageMath(operation="Laplacian", op2="1.5 1", copy_header=True), name="lap_tmpl"
    )
    lap_tmpl.inputs.op1 = tpl_target_path
    lap_target = pe.Node(
        ImageMath(operation="Laplacian", op2="1.5 1", copy_header=True),
        name="lap_target",
    )
    mrg_tmpl = pe.Node(niu.Merge(2), name="mrg_tmpl")
    mrg_tmpl.inputs.in1 = tpl_target_path
    mrg_target = pe.Node(niu.Merge(2), name="mrg_target")

    # Initialize transforms with antsAI
    init_aff = pe.Node(
        AI(
            metric=("Mattes", 32, "Regular", 0.25),
            transform=("Affine", 0.1),
            search_factor=(15, 0.1),
            principal_axes=False,
            convergence=(10, 1e-6, 10),
            verbose=True,
        ),
        name="init_aff",
        n_procs=omp_nthreads,
    )

    # Tolerate missing ANTs at construction time
    try:
        init_aff.inputs.search_grid = (40, (0, 40, 40))
    except ValueError:
        warn(
            "antsAI's option --search-grid was added in ANTS 2.3.0 "
            f"({init_aff.interface.version} found.)"
        )

    # Set up spatial normalization
    settings_file = (
        "antsBrainExtraction_%s.json"
        if use_laplacian
        else "antsBrainExtractionNoLaplacian_%s.json"
    )
    norm = pe.Node(
        Registration(
            from_file=pkgr_fn("niworkflows.data", settings_file % normalization_quality)
        ),
        name="norm",
        n_procs=omp_nthreads,
        mem_gb=mem_gb,
    )
    norm.inputs.float = use_float
    fixed_mask_trait = "fixed_image_mask"

    if norm.interface.version and parseversion(norm.interface.version) >= Version(
        "2.2.0"
    ):
        fixed_mask_trait += "s"

    map_brainmask = pe.Node(
        ApplyTransforms(interpolation="Gaussian"), name="map_brainmask", mem_gb=1,
    )
    map_brainmask.inputs.input_image = str(tpl_mask_path)

    thr_brainmask = pe.Node(
        ThresholdImage(
            dimension=3,
            th_low=0.5,
            th_high=1.0,
            inside_value=1,
            outside_value=0,
            copy_header=True,
        ),
        name="thr_brainmask",
    )

    # Refine INU correction
    inu_n4_final = pe.MapNode(
        N4BiasFieldCorrection(
            dimension=3,
            save_bias=True,
            copy_header=True,
            n_iterations=[50] * 5,
            convergence_threshold=1e-7,
            shrink_factor=4,
            bspline_fitting_distance=bspline_fitting_distance,
        ),
        n_procs=omp_nthreads,
        name="inu_n4_final",
        iterfield=["input_image"],
    )
    try:
        inu_n4_final.inputs.rescale_intensities = True
    except ValueError:
        warn(
            "N4BiasFieldCorrection's --rescale-intensities option was added in ANTS 2.1.0 "
            f"({inu_n4_final.interface.version} found.) Please consider upgrading.",
            UserWarning,
        )

    # Apply mask
    apply_mask = pe.MapNode(ApplyMask(), iterfield=["in_file"], name="apply_mask")

    # fmt: off
    wf.connect([
        (inputnode, trunc, [("in_files", "op1")]),
        (inputnode, inu_n4_final, [("in_files", "input_image")]),
        (inputnode, init_aff, [("in_mask", "fixed_image_mask")]),
        (inputnode, norm, [("in_mask", fixed_mask_trait)]),
        (inputnode, map_brainmask, [(("in_files", _pop), "reference_image")]),
        (trunc, inu_n4, [("output_image", "input_image")]),
        (inu_n4, res_target, [(("output_image", _pop), "in_file")]),
        (res_tmpl, init_aff, [("out_file", "fixed_image")]),
        (res_target, init_aff, [("out_file", "moving_image")]),
        (init_aff, norm, [("output_transform", "initial_moving_transform")]),
        (norm, map_brainmask, [
            ("reverse_transforms", "transforms"),
            ("reverse_invert_flags", "invert_transform_flags"),
        ]),
        (map_brainmask, thr_brainmask, [("output_image", "input_image")]),
        (map_brainmask, inu_n4_final, [("output_image", "weight_image")]),
        (inu_n4_final, apply_mask, [("output_image", "in_file")]),
        (thr_brainmask, apply_mask, [("output_image", "in_mask")]),
        (thr_brainmask, outputnode, [("output_image", "out_mask")]),
        (inu_n4_final, outputnode, [("output_image", "bias_corrected"),
                                    ("bias_image", "bias_image")]),
        (apply_mask, outputnode, [("out_file", "out_file")]),
    ])
    # fmt: on

    wm_tpm = (
        get_template(in_template, label="WM", suffix="probseg", **common_spec) or None
    )
    if wm_tpm:
        map_wmmask = pe.Node(
            ApplyTransforms(interpolation="Gaussian"), name="map_wmmask", mem_gb=1,
        )

        # Add the brain stem if it is found.
        bstem_tpm = (
            get_template(in_template, label="BS", suffix="probseg", **common_spec)
            or None
        )
        if bstem_tpm:
            full_wm = pe.Node(niu.Function(function=_imsum), name="full_wm")
            full_wm.inputs.op1 = str(wm_tpm)
            full_wm.inputs.op2 = str(bstem_tpm)
            # fmt: off
            wf.connect([
                (full_wm, map_wmmask, [("out", "input_image")])
            ])
            # fmt: on
        else:
            map_wmmask.inputs.input_image = str(wm_tpm)
        # fmt: off
        wf.disconnect([
            (map_brainmask, inu_n4_final, [("output_image", "weight_image")]),
        ])
        wf.connect([
            (inputnode, map_wmmask, [(("in_files", _pop), "reference_image")]),
            (norm, map_wmmask, [
                ("reverse_transforms", "transforms"),
                ("reverse_invert_flags", "invert_transform_flags"),
            ]),
            (map_wmmask, inu_n4_final, [("output_image", "weight_image")]),
        ])
        # fmt: on

    if use_laplacian:
        lap_tmpl = pe.Node(
            ImageMath(operation="Laplacian", op2="1.5 1", copy_header=True),
            name="lap_tmpl",
        )
        lap_tmpl.inputs.op1 = tpl_target_path
        lap_target = pe.Node(
            ImageMath(operation="Laplacian", op2="1.5 1", copy_header=True),
            name="lap_target",
        )
        mrg_tmpl = pe.Node(niu.Merge(2), name="mrg_tmpl")
        mrg_tmpl.inputs.in1 = tpl_target_path
        mrg_target = pe.Node(niu.Merge(2), name="mrg_target")
        # fmt: off
        wf.connect([
            (inu_n4, lap_target, [(("output_image", _pop), "op1")]),
            (lap_tmpl, mrg_tmpl, [("output_image", "in2")]),
            (inu_n4, mrg_target, [("output_image", "in1")]),
            (lap_target, mrg_target, [("output_image", "in2")]),
            (mrg_tmpl, norm, [("out", "fixed_image")]),
            (mrg_target, norm, [("out", "moving_image")]),
        ])
        # fmt: on

    else:
        norm.inputs.fixed_image = tpl_target_path
        # fmt: off
        wf.connect([
            (inu_n4, norm, [(("output_image", _pop), "moving_image")]),
        ])
        # fmt: on

    if atropos_refine:
        atropos_model = atropos_model or list(ATROPOS_MODELS[bids_suffix].values())
        atropos_wf = init_atropos_wf(
            use_random_seed=atropos_use_random_seed,
            omp_nthreads=omp_nthreads,
            mem_gb=mem_gb,
            in_segmentation_model=atropos_model,
            bspline_fitting_distance=bspline_fitting_distance,
            wm_prior=bool(wm_tpm),
        )

        # fmt: off
        wf.disconnect([
            (thr_brainmask, outputnode, [("output_image", "out_mask")]),
            (inu_n4_final, outputnode, [("output_image", "bias_corrected"),
                                        ("bias_image", "bias_image")]),
            (apply_mask, outputnode, [("out_file", "out_file")]),
        ])
        wf.connect([
            (inputnode, atropos_wf, [("in_files", "inputnode.in_files")]),
            (inu_n4_final, atropos_wf, [("output_image", "inputnode.in_corrected")]),
            (thr_brainmask, atropos_wf, [("output_image", "inputnode.in_mask")]),
            (atropos_wf, outputnode, [
                ("outputnode.out_file", "out_file"),
                ("outputnode.bias_corrected", "bias_corrected"),
                ("outputnode.bias_image", "bias_image"),
                ("outputnode.out_mask", "out_mask"),
                ("outputnode.out_segm", "out_segm"),
                ("outputnode.out_tpms", "out_tpms"),
            ]),
        ])
        # fmt: on
        if wm_tpm:
            # fmt: off
            wf.connect([
                (map_wmmask, atropos_wf, [("output_image", "inputnode.wm_prior")]),
            ])
            # fmt: on
    return wf


def init_atropos_wf(
    name="atropos_wf",
    use_random_seed=True,
    omp_nthreads=None,
    mem_gb=3.0,
    padding=10,
    in_segmentation_model=tuple(ATROPOS_MODELS["T1w"].values()),
    bspline_fitting_distance=200,
    wm_prior=False,
):
    """
    Create an ANTs' ATROPOS workflow for brain tissue segmentation.

    Re-interprets supersteps 6 and 7 of ``antsBrainExtraction.sh``,
    which refine the mask previously computed with the spatial
    normalization to the template.
    The workflow also executes steps 8 and 9 of the brain extraction
    workflow.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from niworkflows.anat.ants import init_atropos_wf
            wf = init_atropos_wf()

    Parameters
    ----------
    name : str, optional
        Workflow name (default: "atropos_wf").
    use_random_seed : bool
        Whether ATROPOS should generate a random seed based on the
        system's clock
    omp_nthreads : int
        Maximum number of threads an individual process may use
    mem_gb : float
        Estimated peak memory consumption of the most hungry nodes
        in the workflow
    padding : int
        Pad images with zeros before processing
    in_segmentation_model : tuple
        A k-means segmentation is run to find gray or white matter
        around the edge of the initial brain mask warped from the
        template.
        This produces a segmentation image with :math:`$K$` classes,
        ordered by mean intensity in increasing order.
        With this option, you can control  :math:`$K$` and tell the script which
        classes represent CSF, gray and white matter.
        Format (K, csfLabel, gmLabel, wmLabel).
        Examples:
        ``(3,1,2,3)`` for T1 with K=3, CSF=1, GM=2, WM=3 (default),
        ``(3,3,2,1)`` for T2 with K=3, CSF=3, GM=2, WM=1,
        ``(3,1,3,2)`` for FLAIR with K=3, CSF=1 GM=3, WM=2,
        ``(4,4,2,3)`` uses K=4, CSF=4, GM=2, WM=3.
    bspline_fitting_distance : float
        The size of the b-spline mesh grid elements, in mm (default: 200)
    wm_prior : :obj:`bool`
        Whether the WM posterior obtained with ATROPOS should be regularized with a prior
        map (typically, mapped from the template). When ``wm_prior`` is ``True`` the input
        field ``wm_prior`` of the input node must be connected.

    Inputs
    ------
    in_files : list
        The original anatomical images passed in to the brain-extraction workflow.
    in_corrected : list
        :abbr:`INU (intensity non-uniformity)`-corrected files.
    in_mask : str
        Brain mask calculated previously.
    wm_prior : :obj:`str`
        Path to the WM prior probability map, aligned with the individual data.

    Outputs
    -------
    out_file : :obj:`str`
        Path of the corrected and brain-extracted result, using the ATROPOS refinement.
    bias_corrected : :obj:`str`
        Path of the corrected and result, using the ATROPOS refinement.
    bias_image : :obj:`str`
        Path of the estimated INU bias field, using the ATROPOS refinement.
    out_mask : str
        Refined brain mask
    out_segm : str
        Output segmentation
    out_tpms : str
        Output :abbr:`TPMs (tissue probability maps)`


    """
    wf = pe.Workflow(name)

    out_fields = ["bias_corrected", "bias_image", "out_mask", "out_segm", "out_tpms"]

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=["in_files", "in_corrected", "in_mask", "wm_prior"]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["out_file"] + out_fields), name="outputnode"
    )

    copy_xform = pe.Node(
        CopyXForm(fields=out_fields), name="copy_xform", run_without_submitting=True
    )

    # Morphological dilation, radius=2
    dil_brainmask = pe.Node(
        ImageMath(operation="MD", op2="2", copy_header=True), name="dil_brainmask"
    )
    # Get largest connected component
    get_brainmask = pe.Node(
        ImageMath(operation="GetLargestComponent", copy_header=True),
        name="get_brainmask",
    )

    # Run atropos (core node)
    atropos = pe.Node(
        Atropos(
            convergence_threshold=0.0,
            dimension=3,
            initialization="KMeans",
            likelihood_model="Gaussian",
            mrf_radius=[1, 1, 1],
            mrf_smoothing_factor=0.1,
            n_iterations=3,
            number_of_tissue_classes=in_segmentation_model[0],
            save_posteriors=True,
            use_random_seed=use_random_seed,
        ),
        name="01_atropos",
        n_procs=omp_nthreads,
        mem_gb=mem_gb,
    )

    # massage outputs
    pad_segm = pe.Node(
        ImageMath(operation="PadImage", op2=f"{padding}", copy_header=False),
        name="02_pad_segm",
    )
    pad_mask = pe.Node(
        ImageMath(operation="PadImage", op2=f"{padding}", copy_header=False),
        name="03_pad_mask",
    )

    # Split segmentation in binary masks
    sel_labels = pe.Node(
        niu.Function(
            function=_select_labels, output_names=["out_wm", "out_gm", "out_csf"]
        ),
        name="04_sel_labels",
    )
    sel_labels.inputs.labels = list(reversed(in_segmentation_model[1:]))

    # Select largest components (GM, WM)
    # ImageMath ${DIMENSION} ${EXTRACTION_WM} GetLargestComponent ${EXTRACTION_WM}
    get_wm = pe.Node(ImageMath(operation="GetLargestComponent"), name="05_get_wm")
    get_gm = pe.Node(ImageMath(operation="GetLargestComponent"), name="06_get_gm")

    # Fill holes and calculate intersection
    # ImageMath ${DIMENSION} ${EXTRACTION_TMP} FillHoles ${EXTRACTION_GM} 2
    # MultiplyImages ${DIMENSION} ${EXTRACTION_GM} ${EXTRACTION_TMP} ${EXTRACTION_GM}
    fill_gm = pe.Node(ImageMath(operation="FillHoles", op2="2"), name="07_fill_gm")
    mult_gm = pe.Node(
        MultiplyImages(dimension=3, output_product_image="08_mult_gm.nii.gz"),
        name="08_mult_gm",
    )

    # MultiplyImages ${DIMENSION} ${EXTRACTION_WM} ${ATROPOS_WM_CLASS_LABEL} ${EXTRACTION_WM}
    # ImageMath ${DIMENSION} ${EXTRACTION_TMP} ME ${EXTRACTION_CSF} 10
    relabel_wm = pe.Node(
        MultiplyImages(
            dimension=3,
            second_input=in_segmentation_model[-1],
            output_product_image="09_relabel_wm.nii.gz",
        ),
        name="09_relabel_wm",
    )
    me_csf = pe.Node(ImageMath(operation="ME", op2="10"), name="10_me_csf")

    # ImageMath ${DIMENSION} ${EXTRACTION_GM} addtozero ${EXTRACTION_GM} ${EXTRACTION_TMP}
    # MultiplyImages ${DIMENSION} ${EXTRACTION_GM} ${ATROPOS_GM_CLASS_LABEL} ${EXTRACTION_GM}
    # ImageMath ${DIMENSION} ${EXTRACTION_SEGMENTATION} addtozero ${EXTRACTION_WM} ${EXTRACTION_GM}
    add_gm = pe.Node(ImageMath(operation="addtozero"), name="11_add_gm")
    relabel_gm = pe.Node(
        MultiplyImages(
            dimension=3,
            second_input=in_segmentation_model[-2],
            output_product_image="12_relabel_gm.nii.gz",
        ),
        name="12_relabel_gm",
    )
    add_gm_wm = pe.Node(ImageMath(operation="addtozero"), name="13_add_gm_wm")

    # Superstep 7
    # Split segmentation in binary masks
    sel_labels2 = pe.Node(
        niu.Function(function=_select_labels, output_names=["out_gm", "out_wm"]),
        name="14_sel_labels2",
    )
    sel_labels2.inputs.labels = in_segmentation_model[2:]

    # ImageMath ${DIMENSION} ${EXTRACTION_MASK} addtozero ${EXTRACTION_MASK} ${EXTRACTION_TMP}
    add_7 = pe.Node(ImageMath(operation="addtozero"), name="15_add_7")
    # ImageMath ${DIMENSION} ${EXTRACTION_MASK} ME ${EXTRACTION_MASK} 2
    me_7 = pe.Node(ImageMath(operation="ME", op2="2"), name="16_me_7")
    # ImageMath ${DIMENSION} ${EXTRACTION_MASK} GetLargestComponent ${EXTRACTION_MASK}
    comp_7 = pe.Node(ImageMath(operation="GetLargestComponent"), name="17_comp_7")
    # ImageMath ${DIMENSION} ${EXTRACTION_MASK} MD ${EXTRACTION_MASK} 4
    md_7 = pe.Node(ImageMath(operation="MD", op2="4"), name="18_md_7")
    # ImageMath ${DIMENSION} ${EXTRACTION_MASK} FillHoles ${EXTRACTION_MASK} 2
    fill_7 = pe.Node(ImageMath(operation="FillHoles", op2="2"), name="19_fill_7")
    # ImageMath ${DIMENSION} ${EXTRACTION_MASK} addtozero ${EXTRACTION_MASK} \
    # ${EXTRACTION_MASK_PRIOR_WARPED}
    add_7_2 = pe.Node(ImageMath(operation="addtozero"), name="20_add_7_2")
    # ImageMath ${DIMENSION} ${EXTRACTION_MASK} MD ${EXTRACTION_MASK} 5
    md_7_2 = pe.Node(ImageMath(operation="MD", op2="5"), name="21_md_7_2")
    # ImageMath ${DIMENSION} ${EXTRACTION_MASK} ME ${EXTRACTION_MASK} 5
    me_7_2 = pe.Node(ImageMath(operation="ME", op2="5"), name="22_me_7_2")

    # De-pad
    depad_mask = pe.Node(
        ImageMath(operation="PadImage", op2="-%d" % padding), name="23_depad_mask"
    )
    depad_segm = pe.Node(
        ImageMath(operation="PadImage", op2="-%d" % padding), name="24_depad_segm"
    )
    depad_gm = pe.Node(
        ImageMath(operation="PadImage", op2="-%d" % padding), name="25_depad_gm"
    )
    depad_wm = pe.Node(
        ImageMath(operation="PadImage", op2="-%d" % padding), name="26_depad_wm"
    )
    depad_csf = pe.Node(
        ImageMath(operation="PadImage", op2="-%d" % padding), name="27_depad_csf"
    )

    msk_conform = pe.Node(niu.Function(function=_conform_mask), name="msk_conform")
    merge_tpms = pe.Node(niu.Merge(in_segmentation_model[0]), name="merge_tpms")

    sel_wm = pe.Node(niu.Select(), name="sel_wm", run_without_submitting=True)
    if not wm_prior:
        sel_wm.inputs.index = in_segmentation_model[-1] - 1

    copy_xform_wm = pe.Node(
        CopyXForm(fields=["wm_map"]), name="copy_xform_wm", run_without_submitting=True
    )

    # Refine INU correction
    inu_n4_final = pe.MapNode(
        N4BiasFieldCorrection(
            dimension=3,
            save_bias=True,
            copy_header=True,
            n_iterations=[50] * 5,
            convergence_threshold=1e-7,
            shrink_factor=4,
            bspline_fitting_distance=bspline_fitting_distance,
        ),
        n_procs=omp_nthreads,
        name="inu_n4_final",
        iterfield=["input_image"],
    )

    try:
        inu_n4_final.inputs.rescale_intensities = True
    except ValueError:
        warn(
            "N4BiasFieldCorrection's --rescale-intensities option was added in ANTS 2.1.0 "
            f"({inu_n4_final.interface.version} found.) Please consider upgrading.",
            UserWarning,
        )

    # Apply mask
    apply_mask = pe.MapNode(ApplyMask(), iterfield=["in_file"], name="apply_mask")

    # fmt: off
    wf.connect([
        (inputnode, dil_brainmask, [("in_mask", "op1")]),
        (inputnode, copy_xform, [(("in_files", _pop), "hdr_file")]),
        (inputnode, copy_xform_wm, [(("in_files", _pop), "hdr_file")]),
        (inputnode, pad_mask, [("in_mask", "op1")]),
        (inputnode, atropos, [("in_corrected", "intensity_images")]),
        (inputnode, inu_n4_final, [("in_files", "input_image")]),
        (inputnode, msk_conform, [(("in_files", _pop), "in_reference")]),
        (dil_brainmask, get_brainmask, [("output_image", "op1")]),
        (get_brainmask, atropos, [("output_image", "mask_image")]),
        (atropos, pad_segm, [("classified_image", "op1")]),
        (pad_segm, sel_labels, [("output_image", "in_segm")]),
        (sel_labels, get_wm, [("out_wm", "op1")]),
        (sel_labels, get_gm, [("out_gm", "op1")]),
        (get_gm, fill_gm, [("output_image", "op1")]),
        (get_gm, mult_gm, [("output_image", "first_input")]),
        (fill_gm, mult_gm, [("output_image", "second_input")]),
        (get_wm, relabel_wm, [("output_image", "first_input")]),
        (sel_labels, me_csf, [("out_csf", "op1")]),
        (mult_gm, add_gm, [("output_product_image", "op1")]),
        (me_csf, add_gm, [("output_image", "op2")]),
        (add_gm, relabel_gm, [("output_image", "first_input")]),
        (relabel_wm, add_gm_wm, [("output_product_image", "op1")]),
        (relabel_gm, add_gm_wm, [("output_product_image", "op2")]),
        (add_gm_wm, sel_labels2, [("output_image", "in_segm")]),
        (sel_labels2, add_7, [("out_wm", "op1"), ("out_gm", "op2")]),
        (add_7, me_7, [("output_image", "op1")]),
        (me_7, comp_7, [("output_image", "op1")]),
        (comp_7, md_7, [("output_image", "op1")]),
        (md_7, fill_7, [("output_image", "op1")]),
        (fill_7, add_7_2, [("output_image", "op1")]),
        (pad_mask, add_7_2, [("output_image", "op2")]),
        (add_7_2, md_7_2, [("output_image", "op1")]),
        (md_7_2, me_7_2, [("output_image", "op1")]),
        (me_7_2, depad_mask, [("output_image", "op1")]),
        (add_gm_wm, depad_segm, [("output_image", "op1")]),
        (relabel_wm, depad_wm, [("output_product_image", "op1")]),
        (relabel_gm, depad_gm, [("output_product_image", "op1")]),
        (sel_labels, depad_csf, [("out_csf", "op1")]),
        (depad_csf, merge_tpms, [("output_image", "in1")]),
        (depad_gm, merge_tpms, [("output_image", "in2")]),
        (depad_wm, merge_tpms, [("output_image", "in3")]),
        (depad_mask, msk_conform, [("output_image", "in_mask")]),
        (msk_conform, copy_xform, [("out", "out_mask")]),
        (depad_segm, copy_xform, [("output_image", "out_segm")]),
        (merge_tpms, copy_xform, [("out", "out_tpms")]),
        (atropos, sel_wm, [("posteriors", "inlist")]),
        (sel_wm, copy_xform_wm, [("out", "wm_map")]),
        (copy_xform_wm, inu_n4_final, [("wm_map", "weight_image")]),
        (inu_n4_final, copy_xform, [("output_image", "bias_corrected"),
                                    ("bias_image", "bias_image")]),
        (copy_xform, apply_mask, [("bias_corrected", "in_file"),
                                  ("out_mask", "in_mask")]),
        (apply_mask, outputnode, [("out_file", "out_file")]),
        (copy_xform, outputnode, [
            ("bias_corrected", "bias_corrected"),
            ("bias_image", "bias_image"),
            ("out_mask", "out_mask"),
            ("out_segm", "out_segm"),
            ("out_tpms", "out_tpms"),
        ]),
    ])
    # fmt: on

    if wm_prior:
        from nipype.algorithms.metrics import FuzzyOverlap

        def _argmax(in_dice):
            import numpy as np

            return np.argmax(in_dice)

        match_wm = pe.Node(
            niu.Function(function=_matchlen),
            name="match_wm",
            run_without_submitting=True,
        )
        overlap = pe.Node(FuzzyOverlap(), name="overlap", run_without_submitting=True)

        apply_wm_prior = pe.Node(niu.Function(function=_improd), name="apply_wm_prior")

        # fmt: off
        wf.disconnect([
            (copy_xform_wm, inu_n4_final, [("wm_map", "weight_image")]),
        ])
        wf.connect([
            (inputnode, apply_wm_prior, [("in_mask", "in_mask"),
                                         ("wm_prior", "op2")]),
            (inputnode, match_wm, [("wm_prior", "value")]),
            (atropos, match_wm, [("posteriors", "reference")]),
            (atropos, overlap, [("posteriors", "in_ref")]),
            (match_wm, overlap, [("out", "in_tst")]),
            (overlap, sel_wm, [(("class_fdi", _argmax), "index")]),
            (copy_xform_wm, apply_wm_prior, [("wm_map", "op1")]),
            (apply_wm_prior, inu_n4_final, [("out", "weight_image")]),
        ])
        # fmt: on
    return wf


def init_n4_only_wf(
    atropos_model=None,
    atropos_refine=True,
    atropos_use_random_seed=True,
    bids_suffix="T1w",
    mem_gb=3.0,
    name="n4_only_wf",
    omp_nthreads=None,
):
    """
    Build a workflow to sidetrack brain extraction on skull-stripped datasets.

    An alternative workflow to "init_brain_extraction_wf", for anatomical
    images which have already been brain extracted.

      1. Creates brain mask assuming all zero voxels are outside the brain
      2. Applies N4 bias field correction
      3. (Optional) apply ATROPOS and massage its outputs
      4. Use results from 3 to refine N4 bias field correction

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from niworkflows.anat.ants import init_n4_only_wf
            wf = init_n4_only_wf()

    Parameters
    ----------
    omp_nthreads : int
        Maximum number of threads an individual process may use
    mem_gb : float
        Estimated peak memory consumption of the most hungry nodes
    bids_suffix : str
        Sequence type of the first input image. For a list of acceptable values see
        https://bids-specification.readthedocs.io/en/latest/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#anatomy-imaging-data
    atropos_refine : bool
        Enables or disables the whole ATROPOS sub-workflow
    atropos_use_random_seed : bool
        Whether ATROPOS should generate a random seed based on the
        system's clock
    atropos_model : tuple or None
        Allows to specify a particular segmentation model, overwriting
        the defaults based on ``bids_suffix``
    name : str, optional
        Workflow name (default: ``'n4_only_wf'``).

    Inputs
    ------
    in_files
        List of input anatomical images to be bias corrected,
        typically T1-weighted.
        If a list of anatomical images is provided, subsequently
        specified images are used during the segmentation process.
        However, only the first image is used in the registration
        of priors.
        Our suggestion would be to specify the T1w as the first image.

    Outputs
    -------
    out_file
        :abbr:`INU (intensity non-uniformity)`-corrected ``in_files``
    out_mask
        Calculated brain mask
    bias_corrected
        Same as "out_file", provided for consistency with brain extraction
    bias_image
        The :abbr:`INU (intensity non-uniformity)` field estimated for each
        input in ``in_files``
    out_segm
        Output segmentation by ATROPOS
    out_tpms
        Output :abbr:`TPMs (tissue probability maps)` by ATROPOS

    """
    from ..interfaces.nibabel import Binarize

    wf = pe.Workflow(name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["in_files", "in_mask"]), name="inputnode"
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "out_file",
                "out_mask",
                "bias_corrected",
                "bias_image",
                "out_segm",
                "out_tpms",
            ]
        ),
        name="outputnode",
    )

    # Create brain mask
    thr_brainmask = pe.Node(Binarize(thresh_low=2), name="binarize")

    # INU correction
    inu_n4_final = pe.MapNode(
        N4BiasFieldCorrection(
            dimension=3,
            save_bias=True,
            copy_header=True,
            n_iterations=[50] * 5,
            convergence_threshold=1e-7,
            shrink_factor=4,
            bspline_fitting_distance=200,
        ),
        n_procs=omp_nthreads,
        name="inu_n4_final",
        iterfield=["input_image"],
    )

    # Check ANTs version
    try:
        inu_n4_final.inputs.rescale_intensities = True
    except ValueError:
        warn(
            "N4BiasFieldCorrection's --rescale-intensities option was added in ANTS 2.1.0 "
            f"({inu_n4_final.interface.version} found.) Please consider upgrading.",
            UserWarning,
        )

    # fmt: off
    wf.connect([
        (inputnode, inu_n4_final, [("in_files", "input_image")]),
        (inputnode, thr_brainmask, [(("in_files", _pop), "in_file")]),
        (thr_brainmask, outputnode, [("out_mask", "out_mask")]),
        (inu_n4_final, outputnode, [("output_image", "out_file"),
                                    ("output_image", "bias_corrected"),
                                    ("bias_image", "bias_image")]),
    ])
    # fmt: on

    # If atropos refine, do in4 twice
    if atropos_refine:
        atropos_model = atropos_model or list(ATROPOS_MODELS[bids_suffix].values())
        atropos_wf = init_atropos_wf(
            use_random_seed=atropos_use_random_seed,
            omp_nthreads=omp_nthreads,
            mem_gb=mem_gb,
            in_segmentation_model=atropos_model,
        )

        # fmt: off
        wf.disconnect([
            (inu_n4_final, outputnode, [("output_image", "out_file"),
                                        ("output_image", "bias_corrected"),
                                        ("bias_image", "bias_image")]),
        ])
        wf.connect([
            (inputnode, atropos_wf, [("in_files", "inputnode.in_files")]),
            (inu_n4_final, atropos_wf, [("output_image", "inputnode.in_corrected")]),
            (thr_brainmask, atropos_wf, [("out_mask", "inputnode.in_mask")]),
            (atropos_wf, outputnode, [
                ("outputnode.out_file", "out_file"),
                ("outputnode.bias_corrected", "bias_corrected"),
                ("outputnode.bias_image", "bias_image"),
                ("outputnode.out_segm", "out_segm"),
                ("outputnode.out_tpms", "out_tpms"),
            ]),
        ])
        # fmt: on

    return wf


def _select_labels(in_segm, labels):
    from os import getcwd
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    out_files = []

    cwd = getcwd()
    nii = nb.load(in_segm)
    label_data = np.asanyarray(nii.dataobj).astype("uint8")
    for label in labels:
        newnii = nii.__class__(np.uint8(label_data == label), nii.affine, nii.header)
        newnii.set_data_dtype("uint8")
        out_file = fname_presuffix(in_segm, suffix="_class-%02d" % label, newpath=cwd)
        newnii.to_filename(out_file)
        out_files.append(out_file)
    return out_files


def _conform_mask(in_mask, in_reference):
    """Ensures the mask headers make sense and match those of the T1w"""
    from pathlib import Path
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    ref = nb.load(in_reference)
    nii = nb.load(in_mask)
    hdr = nii.header.copy()
    hdr.set_data_dtype("int16")
    hdr.set_slope_inter(1, 0)

    qform, qcode = ref.header.get_qform(coded=True)
    if qcode is not None:
        hdr.set_qform(qform, int(qcode))

    sform, scode = ref.header.get_sform(coded=True)
    if scode is not None:
        hdr.set_sform(sform, int(scode))

    if "_maths" in in_mask:  # Cut the name at first _maths occurrence
        ext = "".join(Path(in_mask).suffixes)
        basename = Path(in_mask).name
        in_mask = basename.split("_maths")[0] + ext

    out_file = fname_presuffix(in_mask, suffix="_mask", newpath=str(Path()))
    nii.__class__(
        np.asanyarray(nii.dataobj).astype("int16"), ref.affine, hdr
    ).to_filename(out_file)
    return out_file


def _matchlen(value, reference):
    return [value] * len(reference)


def _imsum(op1, op2, out_file=None):
    import nibabel as nb

    im1 = nb.load(op1)

    data = im1.get_fdata(dtype="float32") + nb.load(op2).get_fdata(dtype="float32")
    data /= data.max()
    nii = nb.Nifti1Image(data, im1.affine, im1.header)

    if out_file is None:
        from pathlib import Path

        out_file = str((Path() / "summap.nii.gz").absolute())

    nii.to_filename(out_file)
    return out_file


def _improd(op1, op2, in_mask, out_file=None):
    import nibabel as nb

    im1 = nb.load(op1)

    data = im1.get_fdata(dtype="float32") * nb.load(op2).get_fdata(dtype="float32")
    mskdata = nb.load(in_mask).get_fdata() > 0
    data[~mskdata] = 0
    data[data < 0] = 0
    data /= data.max()
    data = 0.5 * (data + mskdata)
    nii = nb.Nifti1Image(data, im1.affine, im1.header)

    if out_file is None:
        from pathlib import Path

        out_file = str((Path() / "prodmap.nii.gz").absolute())

    nii.to_filename(out_file)
    return out_file
