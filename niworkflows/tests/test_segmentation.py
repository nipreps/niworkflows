# -*- coding: utf-8 -*-
""" Segmentation tests """

from __future__ import absolute_import, division, print_function, unicode_literals

import os
from shutil import copy
from tempfile import TemporaryDirectory
import pytest
from templateflow.api import get as get_template

from nipype.pipeline import engine as pe
from niworkflows.interfaces.segmentation import FASTRPT, ReconAllRPT
from niworkflows.interfaces.masks import (
    BETRPT,
    BrainExtractionRPT,
    SimpleShowMaskRPT,
    ROIsPlot,
)
from .conftest import _run_interface_mock, datadir, has_fsl, has_freesurfer


def _smoke_test_report(report_interface, artifact_name):
    with TemporaryDirectory() as tmpdir:
        res = pe.Node(report_interface, name="smoke_test", base_dir=tmpdir).run()
        out_report = res.outputs.out_report

        save_artifacts = os.getenv("SAVE_CIRCLE_ARTIFACTS", False)
        if save_artifacts:
            copy(out_report, os.path.join(save_artifacts, artifact_name))
        assert os.path.isfile(out_report), 'Report "%s" does not exist' % out_report


@pytest.mark.skipif(not has_fsl, reason="No FSL")
def test_BETRPT(moving):
    """ the BET report capable test """
    bet_rpt = BETRPT(generate_report=True, in_file=moving)
    _smoke_test_report(bet_rpt, "testBET.svg")


def test_ROIsPlot(tmp_path):
    """ the BET report capable test """
    import nibabel as nb
    import numpy as np

    im = nb.load(
        str(
            get_template(
                "OASIS30ANTs",
                resolution=1,
                desc="4",
                suffix="dseg",
                extension=[".nii", ".nii.gz"],
            )
        )
    )
    lookup = np.zeros(5, dtype=int)
    lookup[1] = 1
    lookup[2] = 4
    lookup[3] = 2
    lookup[4] = 3
    newdata = lookup[np.round(im.get_fdata()).astype(int)]
    hdr = im.header.copy()
    hdr.set_data_dtype("int16")
    hdr["scl_slope"] = 1
    hdr["scl_inter"] = 0
    out_file = str(tmp_path / "segments.nii.gz")
    nb.Nifti1Image(newdata, im.affine, hdr).to_filename(out_file)
    roi_rpt = ROIsPlot(
        generate_report=True,
        in_file=str(get_template("OASIS30ANTs", resolution=1, desc=None, suffix="T1w")),
        in_mask=str(
            get_template("OASIS30ANTs", resolution=1, desc="brain", suffix="mask")
        ),
        in_rois=[out_file],
        levels=[1.5, 2.5, 3.5],
        colors=["gold", "magenta", "b"],
    )
    _smoke_test_report(roi_rpt, "testROIsPlot.svg")


def test_ROIsPlot2(tmp_path):
    """ the BET report capable test """
    import nibabel as nb
    import numpy as np

    im = nb.load(
        str(
            get_template(
                "OASIS30ANTs",
                resolution=1,
                desc="4",
                suffix="dseg",
                extension=[".nii", ".nii.gz"],
            )
        )
    )
    lookup = np.zeros(5, dtype=int)
    lookup[1] = 1
    lookup[2] = 4
    lookup[3] = 2
    lookup[4] = 3
    newdata = lookup[np.round(im.get_fdata()).astype(int)]
    hdr = im.header.copy()
    hdr.set_data_dtype("int16")
    hdr["scl_slope"] = 1
    hdr["scl_inter"] = 0

    out_files = []
    for i in range(1, 5):
        seg = np.zeros_like(newdata, dtype="uint8")
        seg[(newdata > 0) & (newdata <= i)] = 1
        out_file = str(tmp_path / ("segments%02d.nii.gz" % i))
        nb.Nifti1Image(seg, im.affine, hdr).to_filename(out_file)
        out_files.append(out_file)
    roi_rpt = ROIsPlot(
        generate_report=True,
        in_file=str(get_template("OASIS30ANTs", resolution=1, desc=None, suffix="T1w")),
        in_mask=str(
            get_template("OASIS30ANTs", resolution=1, desc="brain", suffix="mask")
        ),
        in_rois=out_files,
        colors=["gold", "lightblue", "b", "g"],
    )
    _smoke_test_report(roi_rpt, "testROIsPlot2.svg")


def test_SimpleShowMaskRPT():
    """ the BET report capable test """

    msk_rpt = SimpleShowMaskRPT(
        generate_report=True,
        background_file=str(
            get_template("OASIS30ANTs", resolution=1, desc=None, suffix="T1w")
        ),
        mask_file=str(
            get_template(
                "OASIS30ANTs",
                resolution=1,
                desc="BrainCerebellumRegistration",
                suffix="mask",
            )
        ),
    )
    _smoke_test_report(msk_rpt, "testSimpleMask.svg")


def test_BrainExtractionRPT(monkeypatch, moving, nthreads):
    """ test antsBrainExtraction with reports"""

    def _agg(objekt, runtime):
        outputs = objekt.output_spec()
        outputs.BrainExtractionMask = os.path.join(
            datadir, "testBrainExtractionRPTBrainExtractionMask.nii.gz"
        )
        outputs.out_report = os.path.join(runtime.cwd, objekt.inputs.out_report)
        return outputs

    # Patch the _run_interface method
    monkeypatch.setattr(BrainExtractionRPT, "_run_interface", _run_interface_mock)
    monkeypatch.setattr(BrainExtractionRPT, "aggregate_outputs", _agg)

    bex_rpt = BrainExtractionRPT(
        generate_report=True,
        dimension=3,
        use_floatingpoint_precision=1,
        anatomical_image=moving,
        brain_template=str(
            get_template("OASIS30ANTs", resolution=1, desc=None, suffix="T1w")
        ),
        brain_probability_mask=str(
            get_template("OASIS30ANTs", resolution=1, label="brain", suffix="probseg")
        ),
        extraction_registration_mask=str(
            get_template(
                "OASIS30ANTs",
                resolution=1,
                desc="BrainCerebellumRegistration",
                suffix="mask",
            )
        ),
        out_prefix="testBrainExtractionRPT",
        debug=True,  # run faster for testing purposes
        num_threads=nthreads,
    )
    _smoke_test_report(bex_rpt, "testANTSBrainExtraction.svg")


@pytest.mark.skipif(not has_fsl, reason="No FSL")
@pytest.mark.parametrize("segments", [True, False])
def test_FASTRPT(monkeypatch, segments, reference, reference_mask):
    """ test FAST with the two options for segments """
    from nipype.interfaces.fsl.maths import ApplyMask

    def _agg(objekt, runtime):
        outputs = objekt.output_spec()
        outputs.out_report = os.path.join(runtime.cwd, objekt.inputs.out_report)
        outputs.tissue_class_map = os.path.join(
            datadir, "testFASTRPT-tissue_class_map.nii.gz"
        )
        outputs.tissue_class_files = [
            os.path.join(datadir, "testFASTRPT-tissue_class_files0.nii.gz"),
            os.path.join(datadir, "testFASTRPT-tissue_class_files1.nii.gz"),
            os.path.join(datadir, "testFASTRPT-tissue_class_files2.nii.gz"),
        ]
        return outputs

    # Patch the _run_interface method
    monkeypatch.setattr(FASTRPT, "_run_interface", _run_interface_mock)
    monkeypatch.setattr(FASTRPT, "aggregate_outputs", _agg)

    brain = (
        pe.Node(ApplyMask(in_file=reference, mask_file=reference_mask), name="brain")
        .run()
        .outputs.out_file
    )
    fast_rpt = FASTRPT(
        in_files=brain,
        generate_report=True,
        no_bias=True,
        probability_maps=True,
        segments=segments,
        out_basename="test",
    )
    _smoke_test_report(fast_rpt, "testFAST_%ssegments.svg" % ("no" * int(not segments)))


@pytest.mark.skipif(not has_freesurfer, reason="No FreeSurfer")
def test_ReconAllRPT(monkeypatch):
    # Patch the _run_interface method
    monkeypatch.setattr(ReconAllRPT, "_run_interface", _run_interface_mock)

    rall_rpt = ReconAllRPT(
        subject_id="fsaverage",
        directive="all",
        subjects_dir=os.getenv("SUBJECTS_DIR"),
        generate_report=True,
    )

    _smoke_test_report(rall_rpt, "testReconAll.svg")
