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
    "valid,lic,stderr",
    [
        (True, None, b""),
        (False, None, b"ERROR: FreeSurfer license file /made/up/license.txt not found"),
        (True, None, b"Non-license ERROR"),
        (True, "custom/license.txt", b""),
    ],
)
def test_fs_license_check(valid, lic, stderr):
    with mock.patch("subprocess.run") as mocked_run:
        mocked_run.return_value.stderr = stderr
        assert check_valid_fs_license(lic=lic) == valid
        if lic is not None:
            assert os.getenv("FS_LICENSE")
