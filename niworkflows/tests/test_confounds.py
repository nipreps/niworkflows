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
"""Utilities tests"""

import os
from shutil import copy

import numpy as np
import pandas as pd
from nipype.pipeline import engine as pe

from ..interfaces.confounds import ExpandModel, SpikeRegressors
from ..interfaces.plotting import CompCorVariancePlot, ConfoundsCorrelationPlot
from .conftest import datadir


def _smoke_test_report(report_interface, artifact_name):
    out_report = report_interface.run().outputs.out_file

    save_artifacts = os.getenv('SAVE_CIRCLE_ARTIFACTS')
    if save_artifacts:
        copy(out_report, os.path.join(save_artifacts, artifact_name))
    assert os.path.isfile(out_report), f'Report "{out_report}" does not exist'


def _expand_test(model_formula):
    orig_data_file = os.path.join(datadir, 'confounds_test.tsv')
    exp_data_file = (
        pe.Node(
            ExpandModel(confounds_file=orig_data_file, model_formula=model_formula),
            name='expand_model',
        )
        .run()
        .outputs.confounds_file
    )
    return pd.read_csv(exp_data_file, sep='\t')


def _spikes_test(lags=None, mincontig=None, fmt='mask'):
    orig_data_file = os.path.join(datadir, 'spikes_test.tsv')
    lags = lags or [0]
    spk_data_file = (
        pe.Node(
            SpikeRegressors(
                confounds_file=orig_data_file,
                fd_thresh=4,
                dvars_thresh=6,
                lags=lags,
                minimum_contiguous=mincontig,
                output_format=fmt,
                concatenate=False,
            ),
            name='spike_regressors',
        )
        .run()
        .outputs.confounds_file
    )
    return pd.read_csv(spk_data_file, sep='\t')


def test_expansion_variable_selection():
    """Test model expansion: simple variable selection"""
    model_formula = 'a + b + c + d'
    expected_data = pd.DataFrame(
        {
            'a': [-1, -2, -3, -4, -5],
            'b': [2, 2, 2, 2, 2],
            'c': [0, 1, 0, 1, 0],
            'd': [9, 7, 5, 3, 1],
        }
    )
    exp_data = _expand_test(model_formula)
    pd.testing.assert_frame_equal(exp_data, expected_data)


def test_expansion_derivatives_and_powers():
    """Temporal derivatives and quadratics"""
    model_formula = '(dd1(a) + d1(b))^^2 + d1-2((c)^2) + d + others'
    # b_derivative1_power2 is dropped as an exact duplicate of b_derivative1
    expected_data = pd.DataFrame(
        {
            'a': [-1, -2, -3, -4, -5],
            'a_power2': [1, 4, 9, 16, 25],
            'a_derivative1': [np.nan, -1, -1, -1, -1],
            'a_derivative1_power2': [np.nan, 1, 1, 1, 1],
            'b_derivative1': [np.nan, 0, 0, 0, 0],
            'b_derivative1_power2': [np.nan, 0, 0, 0, 0],
            'c_power2_derivative1': [np.nan, 1, -1, 1, -1],
            'c_power2_derivative2': [np.nan, np.nan, -2, 2, -2],
            'd': [9, 7, 5, 3, 1],
            'e': [0, 0, 0, 0, 0],
            'f': [np.nan, 6, 4, 2, 0],
        }
    )
    exp_data = _expand_test(model_formula)
    assert set(exp_data.columns) == set(expected_data.columns)
    for col in expected_data.columns:
        pd.testing.assert_series_equal(expected_data[col], exp_data[col], check_dtype=False)


def test_expansion_na_robustness():
    """NA robustness"""
    model_formula = '(dd1(f))^^2'
    expected_data = pd.DataFrame(
        {
            'f': [np.nan, 6, 4, 2, 0],
            'f_power2': [np.nan, 36, 16, 4, 0],
            'f_derivative1': [np.nan, np.nan, -2, -2, -2],
            'f_derivative1_power2': [np.nan, np.nan, 4, 4, 4],
        }
    )
    exp_data = _expand_test(model_formula)
    assert set(exp_data.columns) == set(expected_data.columns)
    for col in expected_data.columns:
        pd.testing.assert_series_equal(expected_data[col], exp_data[col], check_dtype=False)


def test_spikes():
    """Test outlier flagging"""
    outliers = [1, 1, 0, 0, 1]
    spk_data = _spikes_test()
    assert np.all(np.isclose(outliers, spk_data['motion_outlier']))

    outliers_spikes = pd.DataFrame(
        {
            'motion_outlier00': [1, 0, 0, 0, 0],
            'motion_outlier01': [0, 1, 0, 0, 0],
            'motion_outlier02': [0, 0, 0, 0, 1],
        }
    )
    spk_data = _spikes_test(fmt='spikes')
    assert set(spk_data.columns) == set(outliers_spikes.columns)
    for col in outliers_spikes.columns:
        assert np.all(np.isclose(outliers_spikes[col], spk_data[col]))

    lags = [0, 1]
    outliers_lags = [1, 1, 1, 0, 1]
    spk_data = _spikes_test(lags=lags)
    assert np.all(np.isclose(outliers_lags, spk_data['motion_outlier']))

    mincontig = 2
    outliers_mc = [1, 1, 1, 1, 1]
    spk_data = _spikes_test(lags=lags, mincontig=mincontig)
    assert np.all(np.isclose(outliers_mc, spk_data['motion_outlier']))


def test_CompCorVariancePlot():
    """CompCor variance report test"""
    metadata_file = os.path.join(datadir, 'confounds_metadata_test.tsv')
    cc_rpt = CompCorVariancePlot(metadata_files=[metadata_file], metadata_sources=['aCompCor'])
    _smoke_test_report(cc_rpt, 'compcor_variance.svg')


def test_ConfoundsCorrelationPlot():
    """confounds correlation report test"""
    confounds_file = os.path.join(datadir, 'confounds_test.tsv')
    cc_rpt = ConfoundsCorrelationPlot(
        confounds_file=confounds_file,
        reference_column='a',
        ignore_initial_volumes=1,
    )
    _smoke_test_report(cc_rpt, 'confounds_correlation.svg')


def test_ConfoundsCorrelationPlotColumns():
    """confounds correlation report test"""
    confounds_file = os.path.join(datadir, 'confounds_test.tsv')
    cc_rpt = ConfoundsCorrelationPlot(
        confounds_file=confounds_file,
        reference_column='a',
        columns=['b', 'd', 'f'],
        ignore_initial_volumes=0,
    )
    _smoke_test_report(cc_rpt, 'confounds_correlation_cols.svg')
