# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Brain extraction workflows."""
from nipype.interfaces import afni, utility as niu
from nipype.pipeline import engine as pe
from ..interfaces.nibabel import Binarize
from ..interfaces.fixes import FixN4BiasFieldCorrection as N4BiasFieldCorrection


def afni_wf(name="AFNISkullStripWorkflow", unifize=False, n4_nthreads=1):
    """
    Create a skull-stripping workflow based on AFNI's tools.

    Originally derived from the `codebase of the QAP
    <https://github.com/preprocessed-connectomes-project/quality-assessment-protocol/blob/master/qap/anatomical_preproc.py#L105>`_.
    Now, this workflow includes :abbr:`INU (intensity non-uniformity)` correction
    using the N4 algorithm and (optionally) intensity harmonization using
    ANFI's ``3dUnifize``.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from niworkflows.anat.skullstrip import afni_wf
            wf = afni_wf()

    Parameters
    ----------
    n4_nthreads : int
        number of cpus N4 bias field correction can utilize.
    unifize : bool
        whether AFNI's ``3dUnifize`` should be applied (default: ``False``).
    name : str
        name for the workflow hierarchy of Nipype

    Inputs
    ------
    in_file : str
        input T1w image.

    Outputs
    -------
    bias_corrected : str
        path to the bias corrected input MRI.
    out_file : str
        path to the skull-stripped image.
    out_mask : str
        path to the generated brain mask.
    bias_image : str
        path to the B1 inhomogeneity field.

    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=["in_file"]), name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["bias_corrected", "out_file", "out_mask", "bias_image"]
        ),
        name="outputnode",
    )

    inu_n4 = pe.Node(
        N4BiasFieldCorrection(
            dimension=3,
            save_bias=True,
            num_threads=n4_nthreads,
            rescale_intensities=True,
            copy_header=True,
        ),
        n_procs=n4_nthreads,
        name="inu_n4",
    )

    sstrip = pe.Node(afni.SkullStrip(outputtype="NIFTI_GZ"), name="skullstrip")
    sstrip_orig_vol = pe.Node(
        afni.Calc(expr="a*step(b)", outputtype="NIFTI_GZ"), name="sstrip_orig_vol"
    )
    binarize = pe.Node(Binarize(thresh_low=0.0), name="binarize")

    if unifize:
        # Add two unifize steps, pre- and post- skullstripping.
        inu_uni_0 = pe.Node(
            afni.Unifize(outputtype="NIFTI_GZ"), name="unifize_pre_skullstrip"
        )
        inu_uni_1 = pe.Node(
            afni.Unifize(gm=True, outputtype="NIFTI_GZ"), name="unifize_post_skullstrip"
        )
        # fmt: off
        workflow.connect([
            (inu_n4, inu_uni_0, [("output_image", "in_file")]),
            (inu_uni_0, sstrip, [("out_file", "in_file")]),
            (inu_uni_0, sstrip_orig_vol, [("out_file", "in_file_a")]),
            (sstrip_orig_vol, inu_uni_1, [("out_file", "in_file")]),
            (inu_uni_1, outputnode, [("out_file", "out_file")]),
            (inu_uni_0, outputnode, [("out_file", "bias_corrected")]),
        ])
        # fmt: on
    else:
        # fmt: off
        workflow.connect([
            (inputnode, sstrip_orig_vol, [("in_file", "in_file_a")]),
            (inu_n4, sstrip, [("output_image", "in_file")]),
            (sstrip_orig_vol, outputnode, [("out_file", "out_file")]),
            (inu_n4, outputnode, [("output_image", "bias_corrected")]),
        ])
        # fmt: on

    # Remaining connections
    # fmt: off
    workflow.connect([
        (sstrip, sstrip_orig_vol, [("out_file", "in_file_b")]),
        (inputnode, inu_n4, [("in_file", "input_image")]),
        (sstrip_orig_vol, binarize, [("out_file", "in_file")]),
        (binarize, outputnode, [("out_mask", "out_mask")]),
        (inu_n4, outputnode, [("bias_image", "bias_image")]),
    ])
    # fmt: on
    return workflow
