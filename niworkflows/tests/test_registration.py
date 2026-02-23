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
"""Registration tests"""

import os
from shutil import copy
from tempfile import TemporaryDirectory

import pytest
from nipype.pipeline import engine as pe

from ..interfaces.reportlets.registration import (
    ANTSRegistrationRPT,
    SpatialNormalizationRPT,
)
from .conftest import _run_interface_mock


def _smoke_test_report(report_interface, artifact_name):
    with TemporaryDirectory() as tmpdir:
        res = pe.Node(report_interface, name='smoke_test', base_dir=tmpdir).run()
        out_report = res.outputs.out_report

        save_artifacts = os.getenv('SAVE_CIRCLE_ARTIFACTS')
        if save_artifacts:
            copy(out_report, os.path.join(save_artifacts, artifact_name))
        assert os.path.isfile(out_report), 'Report does not exist'


def test_SpatialNormalizationRPT(monkeypatch, moving, datadir):
    """the SpatialNormalizationRPT report capable test"""

    def _agg(objekt, runtime):
        outputs = objekt.output_spec()
        outputs.warped_image = os.path.join(
            datadir, 'testSpatialNormalizationRPTMovingWarpedImage.nii.gz'
        )
        outputs.out_report = os.path.join(runtime.cwd, objekt.inputs.out_report)
        return outputs

    # Patch the _run_interface method
    monkeypatch.setattr(SpatialNormalizationRPT, '_run_interface', _run_interface_mock)
    monkeypatch.setattr(SpatialNormalizationRPT, 'aggregate_outputs', _agg)

    ants_rpt = SpatialNormalizationRPT(generate_report=True, moving_image=moving, flavor='testing')
    _smoke_test_report(ants_rpt, 'testSpatialNormalizationRPT.svg')


def test_SpatialNormalizationRPT_masked(monkeypatch, moving, reference_mask, datadir):
    """the SpatialNormalizationRPT report capable test with masking"""

    def _agg(objekt, runtime):
        outputs = objekt.output_spec()
        outputs.warped_image = os.path.join(
            datadir, 'testSpatialNormalizationRPTMovingWarpedImage.nii.gz'
        )
        outputs.out_report = os.path.join(runtime.cwd, objekt.inputs.out_report)
        return outputs

    # Patch the _run_interface method
    monkeypatch.setattr(SpatialNormalizationRPT, '_run_interface', _run_interface_mock)
    monkeypatch.setattr(SpatialNormalizationRPT, 'aggregate_outputs', _agg)

    ants_rpt = SpatialNormalizationRPT(
        generate_report=True,
        moving_image=moving,
        reference_mask=reference_mask,
        flavor='testing',
    )
    _smoke_test_report(ants_rpt, 'testSpatialNormalizationRPT_masked.svg')


def test_ANTSRegistrationRPT(monkeypatch, reference, moving, datadir):
    """the SpatialNormalizationRPT report capable test"""
    from niworkflows import data

    def _agg(objekt, runtime):
        outputs = objekt.output_spec()
        outputs.warped_image = os.path.join(datadir, 'testANTSRegistrationRPT-warped_image.nii.gz')
        outputs.out_report = os.path.join(runtime.cwd, objekt.inputs.out_report)
        return outputs

    # Patch the _run_interface method
    monkeypatch.setattr(ANTSRegistrationRPT, '_run_interface', _run_interface_mock)
    monkeypatch.setattr(ANTSRegistrationRPT, 'aggregate_outputs', _agg)

    ants_rpt = ANTSRegistrationRPT(
        generate_report=True,
        moving_image=moving,
        fixed_image=reference,
        from_file=data.load('t1w-mni_registration_testing_000.json'),
    )
    _smoke_test_report(ants_rpt, 'testANTSRegistrationRPT.svg')
