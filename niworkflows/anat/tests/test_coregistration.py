"""Tests for the coregistration helpers."""

from pathlib import Path

from niworkflows.anat.coregistration import compare_xforms

DATA = Path(__file__).parent.parent.parent / 'tests' / 'data'


def test_compare_xforms_returns_a_builtin_bool():
    # nipype's Int traits, e.g. Select.index, reject numpy's bool
    lta = str(DATA / 'valid_transform.lta')
    assert type(compare_xforms([lta, lta])) is bool
