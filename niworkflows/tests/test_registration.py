# -*- coding: utf-8 -*-
""" Registration tests """

from __future__ import absolute_import, division, print_function, unicode_literals

import os
from shutil import copy
import pytest
from tempfile import TemporaryDirectory

from nipype.pipeline import engine as pe
from niworkflows.interfaces.registration import (
    FLIRTRPT,
    RobustMNINormalizationRPT,
    ANTSRegistrationRPT,
    BBRegisterRPT,
    MRICoregRPT,
    ApplyXFMRPT,
    SimpleBeforeAfterRPT,
)
from .conftest import _run_interface_mock, datadir, has_fsl, has_freesurfer


def _smoke_test_report(report_interface, artifact_name):
    with TemporaryDirectory() as tmpdir:
        res = pe.Node(report_interface, name="smoke_test", base_dir=tmpdir).run()
        out_report = res.outputs.out_report

        save_artifacts = os.getenv("SAVE_CIRCLE_ARTIFACTS", False)
        if save_artifacts:
            copy(out_report, os.path.join(save_artifacts, artifact_name))
        assert os.path.isfile(out_report), "Report does not exist"


@pytest.mark.skipif(not has_fsl, reason="No FSL")
def test_FLIRTRPT(reference, moving):
    """ the FLIRT report capable test """
    flirt_rpt = FLIRTRPT(generate_report=True, in_file=moving, reference=reference)
    _smoke_test_report(flirt_rpt, "testFLIRT.svg")


@pytest.mark.skipif(not has_freesurfer, reason="No FreeSurfer")
def test_MRICoregRPT(monkeypatch, reference, moving, nthreads):
    """ the MRICoreg report capable test """

    def _agg(objekt, runtime):
        outputs = objekt.output_spec()
        outputs.out_lta_file = os.path.join(datadir, "testMRICoregRPT-out_lta_file.lta")
        outputs.out_report = os.path.join(runtime.cwd, objekt.inputs.out_report)
        return outputs

    # Patch the _run_interface method
    monkeypatch.setattr(MRICoregRPT, "_run_interface", _run_interface_mock)
    monkeypatch.setattr(MRICoregRPT, "aggregate_outputs", _agg)

    mri_coreg_rpt = MRICoregRPT(
        generate_report=True,
        source_file=moving,
        reference_file=reference,
        num_threads=nthreads,
    )
    _smoke_test_report(mri_coreg_rpt, "testMRICoreg.svg")


@pytest.mark.skipif(not has_fsl, reason="No FSL")
def test_ApplyXFMRPT(reference, moving):
    """ the ApplyXFM report capable test """
    flirt_rpt = FLIRTRPT(generate_report=False, in_file=moving, reference=reference)

    applyxfm_rpt = ApplyXFMRPT(
        generate_report=True,
        in_file=moving,
        in_matrix_file=flirt_rpt.run().outputs.out_matrix_file,
        reference=reference,
        apply_xfm=True,
    )
    _smoke_test_report(applyxfm_rpt, "testApplyXFM.svg")


@pytest.mark.skipif(not has_fsl, reason="No FSL")
def test_SimpleBeforeAfterRPT(reference, moving):
    """ the SimpleBeforeAfterRPT report capable test """
    flirt_rpt = FLIRTRPT(generate_report=False, in_file=moving, reference=reference)

    ba_rpt = SimpleBeforeAfterRPT(
        generate_report=True, before=reference, after=flirt_rpt.run().outputs.out_file
    )
    _smoke_test_report(ba_rpt, "test_SimpleBeforeAfterRPT.svg")


@pytest.mark.skipif(not has_fsl, reason="No FSL")
def test_FLIRTRPT_w_BBR(reference, reference_mask, moving):
    """ test FLIRTRPT with input `wm_seg` set.
    For the sake of testing ONLY, `wm_seg` is set to the filename of a brain mask """
    flirt_rpt = FLIRTRPT(
        generate_report=True, in_file=moving, reference=reference, wm_seg=reference_mask
    )
    _smoke_test_report(flirt_rpt, "testFLIRTRPTBBR.svg")


@pytest.mark.skipif(not has_freesurfer, reason="No FreeSurfer")
def test_BBRegisterRPT(monkeypatch, moving):
    """ the BBRegister report capable test """

    def _agg(objekt, runtime):
        outputs = objekt.output_spec()
        outputs.out_lta_file = os.path.join(
            datadir, "testBBRegisterRPT-out_lta_file.lta"
        )
        outputs.out_report = os.path.join(runtime.cwd, objekt.inputs.out_report)
        return outputs

    # Patch the _run_interface method
    monkeypatch.setattr(BBRegisterRPT, "_run_interface", _run_interface_mock)
    monkeypatch.setattr(BBRegisterRPT, "aggregate_outputs", _agg)

    subject_id = "fsaverage"
    bbregister_rpt = BBRegisterRPT(
        generate_report=True,
        contrast_type="t1",
        init="fsl",
        source_file=moving,
        subject_id=subject_id,
        registered_file=True,
    )
    _smoke_test_report(bbregister_rpt, "testBBRegister.svg")


def test_RobustMNINormalizationRPT(monkeypatch, moving):
    """ the RobustMNINormalizationRPT report capable test """

    def _agg(objekt, runtime):
        outputs = objekt.output_spec()
        outputs.warped_image = os.path.join(
            datadir, "testRobustMNINormalizationRPTMovingWarpedImage.nii.gz"
        )
        outputs.out_report = os.path.join(runtime.cwd, objekt.inputs.out_report)
        return outputs

    # Patch the _run_interface method
    monkeypatch.setattr(
        RobustMNINormalizationRPT, "_run_interface", _run_interface_mock
    )
    monkeypatch.setattr(RobustMNINormalizationRPT, "aggregate_outputs", _agg)

    ants_rpt = RobustMNINormalizationRPT(
        generate_report=True, moving_image=moving, flavor="testing"
    )
    _smoke_test_report(ants_rpt, "testRobustMNINormalizationRPT.svg")


def test_RobustMNINormalizationRPT_masked(monkeypatch, moving, reference_mask):
    """ the RobustMNINormalizationRPT report capable test with masking """

    def _agg(objekt, runtime):
        outputs = objekt.output_spec()
        outputs.warped_image = os.path.join(
            datadir, "testRobustMNINormalizationRPTMovingWarpedImage.nii.gz"
        )
        outputs.out_report = os.path.join(runtime.cwd, objekt.inputs.out_report)
        return outputs

    # Patch the _run_interface method
    monkeypatch.setattr(
        RobustMNINormalizationRPT, "_run_interface", _run_interface_mock
    )
    monkeypatch.setattr(RobustMNINormalizationRPT, "aggregate_outputs", _agg)

    ants_rpt = RobustMNINormalizationRPT(
        generate_report=True,
        moving_image=moving,
        reference_mask=reference_mask,
        flavor="testing",
    )
    _smoke_test_report(ants_rpt, "testRobustMNINormalizationRPT_masked.svg")


def test_ANTSRegistrationRPT(monkeypatch, reference, moving):
    """ the RobustMNINormalizationRPT report capable test """
    import pkg_resources as pkgr

    def _agg(objekt, runtime):
        outputs = objekt.output_spec()
        outputs.warped_image = os.path.join(
            datadir, "testANTSRegistrationRPT-warped_image.nii.gz"
        )
        outputs.out_report = os.path.join(runtime.cwd, objekt.inputs.out_report)
        return outputs

    # Patch the _run_interface method
    monkeypatch.setattr(ANTSRegistrationRPT, "_run_interface", _run_interface_mock)
    monkeypatch.setattr(ANTSRegistrationRPT, "aggregate_outputs", _agg)

    ants_rpt = ANTSRegistrationRPT(
        generate_report=True,
        moving_image=moving,
        fixed_image=reference,
        from_file=pkgr.resource_filename(
            "niworkflows.data", "t1w-mni_registration_testing_000.json"
        ),
    )
    _smoke_test_report(ants_rpt, "testANTSRegistrationRPT.svg")
