"""Test misc module."""
import os
import shutil
from unittest import mock

import pytest
from ..misc import pass_dummy_scans, check_valid_fs_license


@pytest.mark.parametrize(
    "algo_dummy_scans,dummy_scans,expected_out", [(2, 1, 1), (2, None, 2), (2, 0, 0)]
)
def test_pass_dummy_scans(algo_dummy_scans, dummy_scans, expected_out):
    """Check dummy scans passing."""
    skip_vols = pass_dummy_scans(algo_dummy_scans, dummy_scans)

    assert skip_vols == expected_out


@pytest.mark.parametrize(
    "stdout,rc,valid",
    [
        (b"Successful command", 0, True),
        (b"", 0, True),
        (b"ERROR: FreeSurfer license file /made/up/license.txt not found", 1, False),
        (b"Failed output", 1, False),
        (b"ERROR: Systems running GNU glibc version greater than 2.15", 0, False),
    ],
)
def test_fs_license_check(stdout, rc, valid):
    with mock.patch("subprocess.run") as mocked_run:
        mocked_run.return_value.stdout = stdout
        mocked_run.return_value.returncode = rc
        assert check_valid_fs_license() is valid


@pytest.mark.skipif(not os.getenv("FS_LICENSE"), reason="No FS license found")
def test_fs_license_check2(monkeypatch):
    """Execute the canary itself."""
    assert check_valid_fs_license() is True


@pytest.mark.skipif(shutil.which('mri_convert') is None, reason="FreeSurfer not installed")
def test_fs_license_check3(monkeypatch):
    with monkeypatch.context() as m:
        m.delenv("FS_LICENSE", raising=False)
        m.delenv("FREESURFER_HOME", raising=False)
        assert check_valid_fs_license() is False
