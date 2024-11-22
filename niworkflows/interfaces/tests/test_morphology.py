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
"""Test morphology module."""

import shutil
from pathlib import Path

import nibabel as nb
import numpy as np

from niworkflows.interfaces.morphology import (
    BinaryDilation,
    BinarySubtraction,
)


def test_BinaryDilation_interface(tmpdir):
    """Check the dilation interface."""

    data = np.zeros((80, 80, 80), dtype='uint8')
    data[30:-30, 35:-35, 20:-20] = 1

    nb.Nifti1Image(data, np.eye(4), None).to_filename('mask.nii.gz')

    out1 = (
        BinaryDilation(
            in_mask=str(Path('mask.nii.gz').absolute()),
            radius=4,
        )
        .run()
        .outputs.out_mask
    )
    shutil.move(out1, 'large_radius.nii.gz')

    out2 = (
        BinaryDilation(
            in_mask=str(Path('mask.nii.gz').absolute()),
            radius=1,
        )
        .run()
        .outputs.out_mask
    )
    shutil.move(out2, 'small_radius.nii.gz')

    out_final = (
        BinarySubtraction(
            in_base=str(Path('large_radius.nii.gz').absolute()),
            in_subtract=str(Path('small_radius.nii.gz').absolute()),
        )
        .run()
        .outputs.out_mask
    )

    out_data = np.asanyarray(nb.load(out_final).dataobj, dtype='uint8')

    assert np.all(out_data[data] == 0)
