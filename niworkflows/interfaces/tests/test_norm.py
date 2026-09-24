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
import nibabel as nb
import numpy as np
from nibabel.affines import apply_affine
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform

from ..norm import SpatialNormalization, create_cfm


def test_get_settings():
    norm = SpatialNormalization(moving='T1w', flavor='fast')
    settings = norm._get_settings()
    assert len(settings) == 1
    assert settings[0].split('/')[-1] == 't1w-mni_registration_fast_000.json'

    norm = SpatialNormalization(moving='T1w', flavor='testing')
    settings = norm._get_settings()
    assert len(settings) == 3


def test_create_cfm_lesion_orientation(tmp_path):
    """A lesion stored in RAS must be excluded at the correct world location
    even when in_file is stored in LPS (regression test for the lesion mask
    orientation bug)."""
    shape = (4, 5, 6)
    ras_affine = np.eye(4)

    # in_file: all-ones brain mask, stored in LPS orientation.
    ras_img = nb.Nifti1Image(np.ones(shape, dtype=np.uint8), ras_affine)
    xfm = ornt_transform(io_orientation(ras_img.affine), axcodes2ornt(('L', 'P', 'S')))
    in_img = ras_img.as_reoriented(xfm)
    in_file = str(tmp_path / 'in_lps.nii.gz')
    in_img.to_filename(in_file)

    # lesion: single voxel in RAS at world coordinate (1, 2, 3).
    lesion_data = np.zeros(shape, dtype=np.uint8)
    lesion_data[1, 2, 3] = 1
    lesion_file = str(tmp_path / 'lesion_ras.nii.gz')
    nb.Nifti1Image(lesion_data, ras_affine).to_filename(lesion_file)

    out = create_cfm(
        in_file,
        lesion_mask=lesion_file,
        global_mask=True,
        out_path=str(tmp_path / 'cfm.nii.gz'),
    )

    cfm = nb.load(out)
    zeros = np.argwhere(np.asanyarray(cfm.dataobj) == 0)
    # Exactly one voxel excluded, at world coordinate (1, 2, 3).
    assert zeros.shape[0] == 1
    assert np.allclose(apply_affine(cfm.affine, zeros[0]), [1, 2, 3])
