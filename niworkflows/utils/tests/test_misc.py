"""Test misc module."""
import os
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
def test_fs_license_check(tmp_path, stdout, rc, valid):
    with mock.patch("subprocess.run") as mocked_run:
        mocked_run.return_value.stdout = stdout
        mocked_run.return_value.returncode = rc
        assert check_valid_fs_license() is valid
