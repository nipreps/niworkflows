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
    "valid,lic,stdout",
    [
        (True, None, b""),
        (True, None, b"Successful command"),
        (False, None, b"ERROR: FreeSurfer license file /made/up/license.txt not found"),
        (False, "dir", b""),
        (False, "invalid", b"ERROR: Systems running GNU glibc version greater than 2.15"),
    ],
)
def test_fs_license_check(tmp_path, valid, lic, stdout):
    if lic == "dir":
        # point to a directory as the license
        lic = tmp_path / "license.txt"
        lic.mkdir()
    elif lic == "invalid":
        # invalid license file
        lic = tmp_path / "license.txt"
        lic.write_text("Not a license")

    with mock.patch("subprocess.run") as mocked_run:
        mocked_run.return_value.stdout = stdout
        assert check_valid_fs_license(lic=lic) is valid
        if lic is not None:
            assert os.getenv("FS_LICENSE")
