# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
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
"""Tests plotting interfaces."""

import os

import nibabel as nb

from niworkflows import viz
from niworkflows.interfaces.plotting import _get_tr
from niworkflows.tests.data import load_test_data
from niworkflows.utils.timeseries import _cifti_timeseries, _nifti_timeseries


def test_cifti_carpetplot():
    """Exercise extraction of timeseries from CIFTI2."""
    save_artifacts = os.getenv('SAVE_CIRCLE_ARTIFACTS')

    cifti_file = load_test_data(
        'sub-01_task-mixedgamblestask_run-02_space-fsLR_den-91k_bold.dtseries.nii'
    )
    data, segments = _cifti_timeseries(str(cifti_file))
    viz.plot_carpet(
        data,
        segments,
        tr=_get_tr(nb.load(cifti_file)),
        output_file=(
            os.path.join(save_artifacts, 'carpetplot_cifti.svg') if save_artifacts else None
        ),
        drop_trs=0,
        cmap='paired',
    )


def test_nifti_carpetplot():
    """Exercise extraction of timeseries from CIFTI2."""
    save_artifacts = os.getenv('SAVE_CIRCLE_ARTIFACTS')

    nifti_file = load_test_data('sub-ds205s03_task-functionallocalizer_run-01_bold_volreg.nii.gz')
    seg_file = load_test_data('sub-ds205s03_task-functionallocalizer_run-01_bold_parc.nii.gz')
    data, segments = _nifti_timeseries(str(nifti_file), str(seg_file))
    viz.plot_carpet(
        data,
        segments,
        tr=_get_tr(nb.load(nifti_file)),
        output_file=(
            os.path.join(save_artifacts, 'carpetplot_nifti.svg') if save_artifacts else None
        ),
        drop_trs=0,
    )
