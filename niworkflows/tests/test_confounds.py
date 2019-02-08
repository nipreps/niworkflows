# -*- coding: utf-8 -*-
""" Utilities tests """

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import pandas as pd
#from ..interfaces.confounds import ExpandModel, SpikeRegressors
#from .conftest import datadir
from niworkflows.interfaces.confounds import ExpandModel, SpikeRegressors
from niworkflows.tests.conftest import datadir


def _expand_test(orig_data_file, model_formula):
    exp_data_file = ExpandModel(
        confounds_file=orig_data_file,
        model_formula=model_formula
    ).run().outputs.confounds_file
    return pd.read_csv(exp_data_file, sep='\t')


orig_data_file = os.path.join(datadir, 'confounds_test.tsv')
orig_data = pd.read_csv(orig_data_file, sep='\t')


def test_expansion_variable_selection():
    """Test model expansion: simple variable selection"""
    model_formula = 'a + b + c + d'
    expected_data = pd.DataFrame({
        'a': [-1, -2, -3, -4, -5],
        'b': [2, 2, 2, 2, 2],
        'c': [0, 1, 0, 1, 0],
        'd': [9, 7, 5, 3, 1],
    })
    exp_data = _expand_test(orig_data_file, model_formula)
    pd.testing.assert_frame_equal(exp_data, expected_data)


def test_expansion_derivatives_and_powers():
    """Temporal derivatives and quadratics"""
    model_formula = '(dd1(a) + d1(b))^^2 + d1-2((c)^2) + d + others'
    # b_derivative1_power2 is dropped as an exact duplicate of b_derivative1
    expected_data = pd.DataFrame({
        'a': [-1, -2, -3, -4, -5],
        'a_power2': [1, 4, 9, 16, 25],
        'a_derivative1': [np.NaN, -1, -1, -1, -1],
        'a_derivative1_power2': [np.NaN, 1, 1, 1, 1],
        'b_derivative1': [np.NaN, 0, 0, 0, 0],
        'c_power2_derivative1': [np.NaN, 1, -1, 1, -1],
        'c_power2_derivative2': [np.NaN, np.NaN, -2, 2, -2],
        'd': [9, 7, 5, 3, 1],
        'e': [0, 0, 0, 0, 0],
        'f': [np.NaN, 6, 4, 2, 0],
    })
    exp_data = _expand_test(orig_data_file, model_formula)
    assert set(exp_data.columns) == set(expected_data.columns)
    for col in expected_data.columns:
        pd.testing.assert_series_equal(expected_data[col], exp_data[col],
                               check_dtype=False)


def test_expansion_na_robustness():
    """NA robustness"""
    model_formula = '(dd1(f))^^2'
    expected_data = pd.DataFrame({
        'f': [np.NaN, 6, 4, 2, 0],
        'f_power2': [np.NaN, 36, 16, 4, 0],
        'f_derivative1': [np.NaN, np.NaN, -2, -2, -2],
        'f_derivative1_power2': [np.NaN, np.NaN, 4, 4, 4],
    })
    exp_data = _expand_test(orig_data_file, model_formula)
    assert set(exp_data.columns) == set(expected_data.columns)
    for col in expected_data.columns:
        pd.testing.assert_series_equal(expected_data[col], exp_data[col],
                               check_dtype=False)
