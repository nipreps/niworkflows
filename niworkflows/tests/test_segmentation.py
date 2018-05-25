# -*- coding: utf-8 -*-
""" Segmentation tests """

from __future__ import absolute_import, division, print_function, unicode_literals

import os
from shutil import copy
import pytest

from niworkflows.nipype.interfaces.base import Bunch
from niworkflows.interfaces.segmentation import FASTRPT, ReconAllRPT
from niworkflows.interfaces.masks import (
    BETRPT, BrainExtractionRPT, SimpleShowMaskRPT, ROIsPlot
)
from .conftest import _run_interface_mock, datadir


def _smoke_test_report(report_interface, artifact_name):
    report_interface.run()
    out_report = report_interface.inputs.out_report

    save_artifacts = os.getenv('SAVE_CIRCLE_ARTIFACTS', False)
    if save_artifacts:
        copy(out_report, os.path.join(save_artifacts, artifact_name))
    assert os.path.isfile(out_report), 'Report "%s" does not exist' % out_report


def test_BETRPT(moving):
    """ the BET report capable test """
    bet_rpt = BETRPT(generate_report=True, in_file=moving)
    _smoke_test_report(bet_rpt, 'testBET.svg')


def test_ROIsPlot(oasis_dir):
    """ the BET report capable test """
    import nibabel as nb
    import numpy as np

    labels = nb.load(os.path.join(oasis_dir, 'T_template0_glm_4labelsJointFusion.nii.gz'))
    data = labels.get_data()
    out_files = []
    ldata = np.zeros_like(data)
    for i, l in enumerate([1, 3, 4, 2]):
        ldata[data == l] = 1
        out_files.append(os.path.abspath('label%d.nii.gz' % i))
        lfile = nb.Nifti1Image(ldata, labels.affine, labels.header)
        lfile.to_filename(out_files[-1])

    roi_rpt = ROIsPlot(
        generate_report=True,
        in_file=os.path.join(oasis_dir, 'T_template0.nii.gz'),
        in_mask=out_files[-1],
        in_rois=out_files[:-1],
        colors=['g', 'y']
    )
    _smoke_test_report(roi_rpt, 'testROIsPlot.svg')


def test_SimpleShowMaskRPT(oasis_dir):
    """ the BET report capable test """

    msk_rpt = SimpleShowMaskRPT(
        generate_report=True,
        background_file=os.path.join(oasis_dir, 'T_template0.nii.gz'),
        mask_file=os.path.join(oasis_dir, 'T_template0_BrainCerebellumRegistrationMask.nii.gz')
    )
    _smoke_test_report(msk_rpt, 'testSimpleMask.svg')


def test_BrainExtractionRPT(monkeypatch, oasis_dir, moving, nthreads):
    """ test antsBrainExtraction with reports"""

    def _agg(objekt, runtime):
        outputs = Bunch(BrainExtractionMask=os.path.join(
            datadir, 'testBrainExtractionRPTBrainExtractionMask.nii.gz')
        )
        return outputs

    # Patch the _run_interface method
    monkeypatch.setattr(BrainExtractionRPT, '_run_interface',
                        _run_interface_mock)
    monkeypatch.setattr(BrainExtractionRPT, 'aggregate_outputs',
                        _agg)

    bex_rpt = BrainExtractionRPT(
        generate_report=True,
        dimension=3,
        use_floatingpoint_precision=1,
        anatomical_image=moving,
        brain_template=os.path.join(oasis_dir, 'T_template0.nii.gz'),
        brain_probability_mask=os.path.join(
            oasis_dir, 'T_template0_BrainCerebellumProbabilityMask.nii.gz'),
        extraction_registration_mask=os.path.join(
            oasis_dir, 'T_template0_BrainCerebellumRegistrationMask.nii.gz'),
        out_prefix='testBrainExtractionRPT',
        debug=True,  # run faster for testing purposes
        num_threads=nthreads
    )
    _smoke_test_report(bex_rpt, 'testANTSBrainExtraction.svg')


@pytest.mark.parametrize("segments", [True, False])
def test_FASTRPT(monkeypatch, segments, reference, reference_mask):
    """ test FAST with the two options for segments """
    from niworkflows.nipype.interfaces.fsl.maths import ApplyMask

    def _agg(objekt, runtime):
        outputs = Bunch(tissue_class_map=os.path.join(
            datadir, 'testFASTRPT-tissue_class_map.nii.gz'),
            tissue_class_files=[
                os.path.join(datadir, 'testFASTRPT-tissue_class_files0.nii.gz'),
                os.path.join(datadir, 'testFASTRPT-tissue_class_files1.nii.gz'),
                os.path.join(datadir, 'testFASTRPT-tissue_class_files2.nii.gz')]
        )
        return outputs

    # Patch the _run_interface method
    monkeypatch.setattr(FASTRPT, '_run_interface',
                        _run_interface_mock)
    monkeypatch.setattr(FASTRPT, 'aggregate_outputs',
                        _agg)

    brain = ApplyMask(
        in_file=reference, mask_file=reference_mask).run().outputs.out_file
    fast_rpt = FASTRPT(
        in_files=brain,
        generate_report=True,
        no_bias=True,
        probability_maps=True,
        segments=segments,
        out_basename='test'
    )
    _smoke_test_report(
        fast_rpt, 'testFAST_%ssegments.svg' % ('no' * int(not segments)))


def test_ReconAllRPT(monkeypatch):
    # Patch the _run_interface method
    monkeypatch.setattr(ReconAllRPT, '_run_interface',
                        _run_interface_mock)

    rall_rpt = ReconAllRPT(
        subject_id='fsaverage',
        directive='all',
        subjects_dir=os.getenv('SUBJECTS_DIR'),
        generate_report=True
    )

    _smoke_test_report(rall_rpt, 'testReconAll.svg')
