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
"""Exercise interface.header."""

from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from nipype.pipeline import engine as pe

from .. import header


@pytest.mark.parametrize(
    ('qform_add', 'sform_add', 'expectation'),
    [
        (0, 0, 'no_warn'),
        (0, 1e-14, 'no_warn'),
        (0, 1e-09, 'no_warn'),
        (1e-6, 0, 'warn'),
        (0, 1e-6, 'warn'),
        (1e-5, 0, 'warn'),
        (0, 1e-5, 'warn'),
        (1e-3, 1e-3, 'no_warn'),
    ],
)
# just a diagonal of ones in qform and sform and see that this doesn't warn
# only look at the 2 areas of images.py that I added and get code coverage of those
def test_qformsform_warning(tmp_path, qform_add, sform_add, expectation):
    fname = str(tmp_path / 'test.nii')

    # make a random image
    random_data = np.random.random(size=(5, 5, 5) + (5,))
    img = nb.Nifti1Image(random_data, np.eye(4) + sform_add)
    # set the qform of the image before calling it
    img.set_qform(np.eye(4) + qform_add)
    img.to_filename(fname)

    validate = pe.Node(header.ValidateImage(), name='validate', base_dir=str(tmp_path))
    validate.inputs.in_file = fname
    res = validate.run()
    out_report = Path(res.outputs.out_report).read_text()
    if expectation == 'warn':
        assert 'Note on' in out_report
    elif expectation == 'no_warn':
        assert len(out_report) == 0


@pytest.mark.parametrize(
    ('qform_code', 'warning_text'),
    [(0, 'Note on orientation'), (1, 'WARNING - Invalid qform')],
)
def test_bad_qform(tmp_path, qform_code, warning_text):
    fname = str(tmp_path / 'test.nii')

    # make a random image
    random_data = np.random.random(size=(5, 5, 5) + (5,))
    img = nb.Nifti1Image(random_data, np.eye(4))

    # Some magic terms from a bad qform in the wild
    img.header['qform_code'] = qform_code
    img.header['quatern_b'] = 0
    img.header['quatern_c'] = 0.998322
    img.header['quatern_d'] = -0.0579125
    img.to_filename(fname)

    validate = pe.Node(header.ValidateImage(), name='validate', base_dir=str(tmp_path))
    validate.inputs.in_file = fname
    res = validate.run()
    assert warning_text in Path(res.outputs.out_report).read_text()


def test_no_good_affines(tmp_path):
    fname = str(tmp_path / 'test.nii')

    # make a random image
    random_data = np.random.random(size=(5, 5, 5) + (5,))
    img = nb.Nifti1Image(random_data, None)
    img.header['qform_code'] = 0
    img.header['sform_code'] = 0
    img.to_filename(fname)

    validate = pe.Node(header.ValidateImage(), name='validate', base_dir=str(tmp_path))
    validate.inputs.in_file = fname
    res = validate.run()
    assert 'WARNING - Missing orientation information' in Path(res.outputs.out_report).read_text()
