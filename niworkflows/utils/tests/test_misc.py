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
"""Test misc module."""

import os
import shutil
from unittest import mock

import pytest

from niworkflows.testing import has_freesurfer

from ..misc import check_valid_fs_license, pass_dummy_scans


@pytest.mark.parametrize(
    ('algo_dummy_scans', 'dummy_scans', 'expected_out'), [(2, 1, 1), (2, None, 2), (2, 0, 0)]
)
def test_pass_dummy_scans(algo_dummy_scans, dummy_scans, expected_out):
    """Check dummy scans passing."""
    skip_vols = pass_dummy_scans(algo_dummy_scans, dummy_scans)

    assert skip_vols == expected_out


@pytest.mark.parametrize(
    ('stdout', 'rc', 'valid'),
    [
        (b'Successful command', 0, True),
        (b'', 0, True),
        (b'ERROR: FreeSurfer license file /made/up/license.txt not found', 1, False),
        (b'Failed output', 1, False),
        (b'ERROR: Systems running GNU glibc version greater than 2.15', 0, False),
    ],
)
def test_fs_license_check(stdout, rc, valid):
    with mock.patch('subprocess.run') as mocked_run:
        mocked_run.return_value.stdout = stdout
        mocked_run.return_value.returncode = rc
        assert check_valid_fs_license() is valid


@pytest.mark.skipif(not has_freesurfer, reason='Needs FreeSurfer')
@pytest.mark.skipif(not os.getenv('FS_LICENSE'), reason='No FS license found')
def test_fs_license_check2(monkeypatch):
    """Execute the canary itself."""
    assert check_valid_fs_license() is True


@pytest.mark.skipif(shutil.which('mri_convert') is None, reason='FreeSurfer not installed')
def test_fs_license_check3(monkeypatch):
    with monkeypatch.context() as m:
        m.delenv('FS_LICENSE', raising=False)
        m.delenv('FREESURFER_HOME', raising=False)
        assert check_valid_fs_license() is False
