"""Tests for the coregistration helpers."""

import pytest
from nipype.interfaces import utility as niu

from niworkflows.anat.coregistration import compare_xforms

test_data = pytest.importorskip('niworkflows.tests.data')


def test_compare_xforms_returns_a_builtin_bool():
    bbr = str(test_data.load_test_data('testBBRegisterRPT-out_lta_file.lta'))
    mri = str(test_data.load_test_data('testMRICoregRPT-out_lta_file.lta'))
    for ltas, expected in (([bbr, bbr], False), ([bbr, mri], True)):
        result = compare_xforms(ltas)
        assert result is expected
        # the bug: nipype's Int traits, e.g. Select.index, reject numpy's bool
        niu.Select().inputs.index = result
