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
from pathlib import Path

import nibabel as nb
import numpy as np

from ..freesurfer import FSInjectBrainExtracted


def test_inject_skullstrip(tmp_path):
    t1_mgz = tmp_path / 'sub-01' / 'mri' / 'T1.mgz'
    t1_mgz.parent.mkdir(parents=True)
    # T1.mgz images are uint8
    nb.MGHImage(np.ones((5, 5, 5), dtype=np.uint8), np.eye(4)).to_filename(str(t1_mgz))

    mask_nii = tmp_path / 'mask.nii.gz'
    # Masks may be in a different space (and need resampling), but should be boolean,
    # or uint8 in NIfTI
    nb.Nifti1Image(np.ones((6, 6, 6), dtype=np.uint8), np.eye(4)).to_filename(str(mask_nii))

    FSInjectBrainExtracted(
        subjects_dir=str(tmp_path), subject_id='sub-01', in_brain=str(mask_nii)
    ).run()

    assert Path.exists(tmp_path / 'sub-01' / 'mri' / 'brainmask.auto.mgz')
    assert Path.exists(tmp_path / 'sub-01' / 'mri' / 'brainmask.mgz')

    # Run a second time to hit "already exists" condition
    FSInjectBrainExtracted(
        subjects_dir=str(tmp_path), subject_id='sub-01', in_brain=str(mask_nii)
    ).run()
