"""Test misc module."""
import pytest
from ..misc import pass_dummy_scans


@pytest.mark.parametrize('algo_dummy_scans,dummy_scans,expected_out', [
    (2, 1, 1),
    (2, None, 2),
    (2, 0, 0),
])
def test_pass_dummy_scans(algo_dummy_scans, dummy_scans, expected_out):
    """Check dummy scans passing."""
    skip_vols = pass_dummy_scans(algo_dummy_scans, dummy_scans)

    assert skip_vols == expected_out
